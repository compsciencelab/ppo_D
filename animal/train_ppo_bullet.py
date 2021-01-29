

import copy
import glob
import os
import sys
import time
import random
import argparse

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/..')

from ppo import algo, utils
from ppo.envs import make_vec_envs
from ppo.model import Policy
from ppo.model import CNNBase,FixupCNNBase,ImpalaCNNBase,StateCNNBase, MLPBase
from ppo.storage import RolloutStorage
from ppo.algo.ppokl import ppo_rollout, ppo_update, ppo_save_model, ppo_rollout_imitate
from bullet.make_pybullet_env import make_pybullet_env
from vision_module import ImpalaCNNVision
from object_detection_module import ImpalaCNNObject
from wrappers import VecVisionState, VecObjectState 

CNN={'CNN':CNNBase,'Impala':ImpalaCNNBase,'Fixup':FixupCNNBase,'State':StateCNNBase, 'MLP': MLPBase}

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    utils.cleanup_log_dir(args.log_dir)

    env_make = make_pybullet_env(args.task, log_dir = args.log_dir, frame_skip=args.frame_skip, info_keywords=('ereward','reward_woD'), base_seed=args.seed, rho=args.rho, phi=args.phi, demo_dir = args.demo_dir, size_buffer= args.size_buffer, size_buffer_V = args.size_buffer_V, threshold_reward=args.threshold_reward)
    #spaces = ( gym.spaces.Box(low=0, high=0xff,shape=(3, 84, 84),dtype=np.uint8),
    #               gym.spaces.Discrete(9) )
    if args.cnn == 'State':
         #TODO: hugly hack
        state_shape = (13,) if args.reduced_actions else (15,)  
    else:
        state_shape = None

    envs = make_vec_envs(env_make, args.num_processes, args.log_dir, device, args.frame_stack, state_shape, args.state_stack)

    if args.vision_module:
        vision_module, _ = ImpalaCNNVision.load(args.vision_module,device=device)
        vision_module.to(device)
        envs = VecVisionState(envs, vision_module)

    if args.object_module:
        object_module, _ = ImpalaCNNObject.load(args.object_module ,device=device)
        object_module.to(device)
        envs = VecObjectState(envs, object_module)

    actor_critic = Policy(envs.observation_space,envs.action_space,base=CNN[args.cnn],
                            base_kwargs={'recurrent': args.recurrent_policy})

    if args.restart_model:
        actor_critic.load_state_dict(torch.load(args.restart_model, map_location=device))
    actor_critic.to(device)

    actor_behaviors = None
    if args.behavior: 
        actor_behaviors = []
        for a in args.behavior:
            actor = Policy(envs.observation_space, envs.action_space, base=CNN[args.cnn],
                            base_kwargs={'recurrent': args.behavior_recurrent})
            actor.load_state_dict(torch.load(a,map_location=device))
            actor.to(device)
            actor_behaviors.append(actor) 

    agent = algo.PPOKL(actor_critic,args.clip_param,args.ppo_epoch, args.num_mini_batch,args.value_loss_coef,
            args.entropy_coef,lr=args.lr,eps=args.eps,max_grad_norm=args.max_grad_norm,actor_behaviors=actor_behaviors,  log_dir= args.log_dir, behaviour_cloning = args.behaviour_cloning, ppo_bc = args.ppo_bc)

    obs = envs.reset()
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obs, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)


    rollouts.to(device)  #they live in GPU, converted to torch from the env wrapper # comment out

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    infos_in = [{'demo_value_in':[], 'demo_in': [], 'value': None} for i in range(args.num_processes)]

    for j in range(num_updates):

        
        infos_in = ppo_rollout_imitate(args.num_steps, envs, actor_critic, rollouts, infos_in)

        value_loss, action_loss, dist_entropy, kl_div, loss = ppo_update(agent, actor_critic, rollouts,
                                    args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
            
    

        if (j % args.save_interval == 0 or j == num_updates - 1) and args.log_dir != "":
            ppo_save_model(actor_critic, os.path.join(args.log_dir, "animal.state_dict"), j)

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            s =  "Update {}, num timesteps {}, FPS {} \n".format(j, total_num_steps,int(total_num_steps / ( time.time() - start)))
            s += "Loss {}, Entropy {}, value_loss {}, action_loss {}, kl_divergence {}".format(loss, dist_entropy, value_loss,action_loss,kl_div)
            print(s)
    

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',type=float,default=1e-5,help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',type=float,default=0.99,help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',type=float,default=0.99,help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',action='store_true',default=False,help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',type=float,default=0.95,help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',type=float,default=0.01,help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',type=float,default=0.5,help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',type=float,default=0.5,help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument(
        '--num-processes',type=int,default=16,help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',type=int,default=5,help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',type=int,default=4,help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',type=int,default=32,help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',type=float,default=0.2,help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',type=int,default=1,help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',type=int,default=100,help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',type=int,default=None,help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',type=int,default=10e7,help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--log-dir',default='/tmp/ppo/',help='directory to save agent logs (default: /tmp/ppo)')
    parser.add_argument(
        '--use-proper-time-limits',action='store_true',default=False,help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',action='store_true',default=False,help='use a recurrent policy')
    parser.add_argument(
        '--restart-model',default='',help='Restart training using the model given (Gianni)')  
    parser.add_argument(
        '--vision-module',default='',help='File to use to load the vision module ') 
    parser.add_argument(
        '--object-module',default='',help='File to use to load the object module ') 
    parser.add_argument(
        '--behavior',action='append',default=None,help='directory that contains expert policies for high-level actions')
    parser.add_argument(
        '--behavior-recurrent',action='store_true',default=False,help='if the behavior policy is recurrent')
    parser.add_argument(
        '--device',default='cpu',help='Device to run on') 
    parser.add_argument(
        '--frame-skip',type=int,default=0,help='Number of frame to skip for each action')
    parser.add_argument(
        '--frame-stack',type=int,default=4,help='Number of frame to stack in observation') 
    parser.add_argument(
        '--state-stack',type=int,default=4,help='Number of steps to stack in states')               
    parser.add_argument(
        '--realtime',action='store_true',default=False,help='If to plot in realtime. ')
    parser.add_argument(
        '--cnn',default='Fixup',help='Type of cnn. Options are CNN,Impala,Fixup,State') 
    parser.add_argument(
        '--arenas-dir',default=None,help='directory where the yamls files for the environemnt are (default: None)')   
    parser.add_argument(
        '--reduced-actions',action='store_true',default=False,help='Use reduced actions set')
    parser.add_argument(
        '--rho',default=0, type=float, help='ratio of demonstration replays during training')
    parser.add_argument(
        '--phi',default=0, type=float, help='ratio of value self-imitation replays during training')
    parser.add_argument(
        '--size-buffer',type=int,default=0,help='Max num in the reward buffer')
    parser.add_argument(
        '--size-buffer-V',type=int,default=0,help='Max num in the value buffer') 
    parser.add_argument(
        '--demo-dir',default='', help='directory where to get the demonstrations from')
    parser.add_argument(
        '--task',default='HalfCheetahPyBulletEnv-v0',help='which of the pybullet task')
    parser.add_argument(
        '--threshold-reward',type=float,default=None,help='clips the reward under a certain threshold to zero')
    parser.add_argument(
        '--behaviour-cloning', action='store_true',default=False ,help='Adds a behavioural cloning loss')
    parser.add_argument(
        '--ppo-bc', action='store_true',default=False ,help='Alternates between bc and normal PPO')
    
    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    args.state = args.cnn=='State'
    return args

if __name__ == "__main__":
    main()
