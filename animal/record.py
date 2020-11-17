import argparse
from pynput import keyboard
import threading
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/..')
import json
import tarfile
import tempfile

from ppo.model import Policy, CNNBase, FixupCNNBase, ImpalaCNNBase
from collections import deque
import gym
import torch
from ppo.envs import VecPyTorchFrameStack, TransposeImage, VecPyTorch
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np
from matplotlib import pyplot as plt

from ppo.envs import VecPyTorch, make_vec_envs
from animal import make_animal_env
from animalai.envs.arena_config import ArenaConfig

CNN={'CNN':CNNBase,'Impala':ImpalaCNNBase,'Fixup':FixupCNNBase}




parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--arenas-dir', default='', help='yaml dir')
parser.add_argument(
    '--load-model', default='', help='directory to save agent logs (default: )')
parser.add_argument(
    '--device', default='cuda', help='Cuda device  or cpu (default:cuda:0 )')
parser.add_argument(
    '--det', action='store_true', default=False, help='whether to use a non-deterministic policy')
parser.add_argument(
    '--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
parser.add_argument(
    '--realtime', action='store_true', default=False, help='If to plot in realtime. ') 
parser.add_argument(
    '--silent', action='store_true', default=False, help='stop plotting ') 
parser.add_argument(
    '--frame-skip', type=int, default=0, help='Number of frame to skip for each action')
parser.add_argument(
    '--frame-stack', type=int, default=4, help='Number of frame to stack')        
parser.add_argument(
    '--reduced-actions',action='store_true',default=False,help='Use reduced actions set')
parser.add_argument(
    '--vae-model',default=None,help='directory for the dictionary of the initial VAE model')
parser.add_argument(
    '--cnn',default='CNN',help='Type of cnn. Options are CNN,Impala,Fixup')     
parser.add_argument(
    '--replay-ratio',default=0.5, type=float, help='ratio of demonstration replays during training')
parser.add_argument(
    '--record-actions',default='', help='if not not none records actions, directory for the recordings to be stored in ' )
parser.add_argument(
    '--schedule-ratio',action='store_true',default=False ,help='Wether to schedule the replayer ratio')
parser.add_argument(
    '--demo-dir',default= '/workspace7/Unity3D/gabriele/Animal-AI/animal-ppo/RUNS/recorded_reason2', help='directory where to get the demonstrations from')


args = parser.parse_args()
device = torch.device(args.device)

maker = make_animal_env(log_dir = None, inference_mode=args.realtime,  frame_skip=args.frame_skip , 
            arenas_dir=args.arenas_dir, info_keywords=('ereward','max_reward','max_time','arena'), 
            reduced_actions=args.reduced_actions, seed= 1, state= False, replay_ratio= args.replay_ratio, record_actions = args.record_actions,schedule_ratio = args.schedule_ratio, demo_dir = args.demo_dir)

env = make_vec_envs(maker, 1, None, device=device, num_frame_stack=args.frame_stack, 
                        state_shape=None, num_state_stack=14 )


# Get a render function
#render_func = get_render_func(env)


base_kwargs={'recurrent': args.recurrent_policy}
actor_critic = Policy(env.observation_space,env.action_space,base=CNN[args.cnn],base_kwargs=base_kwargs)

if args.load_model:
    actor_critic.load_state_dict(torch.load(args.load_model,map_location=device))
actor_critic.to(device)

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
masks = torch.zeros(1, 1).to(device)

obs = env.reset()

step = 0
S = deque(maxlen = 100)
done = False
all_obs = []
episode_reward = 0


pressed_keys = set([])

def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        if key.name == "space":
            pressed_keys.add("space")

def on_release(key):
    try:
        pressed_keys.remove(key.char)
    except AttributeError:
        if key.name == "space":
            pressed_keys.remove("space")
    except:
        pass

def create_action():
    # forward back
    action = [0, 0]
    if "w" in pressed_keys:
        action[0] = 1
    elif "s" in pressed_keys:
        action[0] = 2

    if "d" in pressed_keys:
        action[1] = 1
    elif "a" in pressed_keys:
        action[1] = 2
    return action


action_lookup = {(0, 0): 0,(0, 1): 1, (0, 2): 2,(1, 0): 3, (1, 1): 4,(1, 2): 5, (2, 0): 6, (2, 1): 7, (2, 2): 8}



def check_act(action):
    found = False
    for e in action_lookup.values():
        if list(e) == list(action):
            found = True
    if found:
        return action


def unroll(acts):
    for act in acts:
        obs, reward, done, info = env.step(act)



with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    threading.Thread(target=listener.join)
    
    while not done:
        
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, dist_entropy = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        action_ = create_action()
        action_2 = action_lookup[tuple(action_)]
        if "k" in pressed_keys:
            action[0] = action_2

        """ if "i" in pressed_keys:
            unroll(acts)
            
        if "u" in pressed_keys:
            if recording == False:
                arena_name =  info[0]['arena'].split('/')[-1].split('.')[0]
                obs_rollouts = []
                rews_rollouts = []
                actions_rollouts = []

            obs_rollouts.append(obs)
            rews_rollouts.append(reward)
            actions_rollouts.append(action)
            recording = True """

        """ sum = 0.0
        for parameter in actor_critic.parameters():
            sum += parameter.sum()
        print(sum)"""
        #import ipdb; ipdb.set_trace()
        data = (action,[[]])
        obs, reward, done, info = env.step(data)

   

        masks.fill_(0.0 if done else 1.0)

        step +=1
        episode_reward += reward 

        """ if not args.silent:
            fig.clf()
            #plt.imshow(transform_view(vobs))
            S.append(dist_entropy.item())
            plt.plot(S)
            plt.draw()
            plt.pause(0.01)

        term = 'goal' in info[0].keys()

        print('Step {} Entropy {:3f} reward {:2f} value {:3f} done {}, bad_transition {} total reward {}'.format(
                    step,dist_entropy.item(),reward.item(),value.item(), done, term, episode_reward.item()))"""

        if done:
            #print("EPISODE: {} steps: ", episode_reward, step, flush=True)
            obs = env.reset()
            step = 0
            episode_reward = 0
            done = False
# drop redundant frames if needed
#all_obs = [x[0,:, :, -3:] for x in  all_obs]
#os.makedirs("./temp/", exist_ok=True)
#for i, x in enumerate(all_obs):
#     cv2.imwrite("./temp/{:05d}.png".format(i), x[:,:, ::-1])
