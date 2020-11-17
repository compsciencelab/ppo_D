import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/..')

from ppo.model import Policy, CNNBase, FixupCNNBase, ImpalaCNNBase, StateCNNBase
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

CNN={'CNN':CNNBase,'Impala':ImpalaCNNBase,'Fixup':FixupCNNBase,'State':StateCNNBase}


parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--arenas-dir', default='', help='yaml dir')
parser.add_argument(
    '--load-model', default='', help='directory to save agent logs (default: )')
parser.add_argument(
    '--device', default='cuda', help='Cuda device  or cpu (default:cuda:0 )')
parser.add_argument(
    '--non-det', action='store_true', default=True, help='whether to use a non-deterministic policy')
parser.add_argument(
    '--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
parser.add_argument(
    '--realtime', action='store_true', default=False, help='If to plot in realtime. ') 
parser.add_argument(
    '--silent', action='store_true', default=False, help='stop plotting ') 
parser.add_argument(
    '--frame-skip', type=int, default=2, help='Number of frame to skip for each action')
parser.add_argument(
    '--frame-stack', type=int, default=4, help='Number of frame to stack')        
parser.add_argument(
    '--reduced-actions',action='store_true',default=False,help='Use reduced actions set')
parser.add_argument(
    '--cnn',default='Fixup',help='Type of cnn. Options are CNN,Impala,Fixup,State')
parser.add_argument(
    '--state-stack',type=int,default=4,help='Number of steps to stack in states')    

args = parser.parse_args()
args.det = not args.non_det
args.state = args.cnn=='State'
device = torch.device(args.device)

maker = make_animal_env(log_dir = None, inference_mode=args.realtime, frame_skip=args.frame_skip,
                        arenas_dir=args.arenas_dir, info_keywords=(),reduced_actions=args.reduced_actions,
                        seed=1, state=args.state)

if args.state:
    #TODO: hugly hack
    state_shape = (13,) if args.reduced_actions else (15,)  
else:
    state_shape = None

env = make_vec_envs(maker, 1, None, device=device, num_frame_stack=args.frame_stack, 
                        state_shape=state_shape, num_state_stack=args.state_stack, )

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

fig = plt.figure()
step = 0
S = deque(maxlen = 100)
done = False
all_obs = []
episode_reward = 0 

while not done:
    
    with torch.no_grad():
        value, action, action_log_prob, recurrent_hidden_states, dist_entropy = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, info = env.step(action)
    masks.fill_(0.0 if done else 1.0)

    step +=1
    episode_reward += reward 

    if not args.silent:
        fig.clf()
        #plt.imshow(transform_view(vobs))
        S.append(dist_entropy.item())
        plt.plot(S)
        plt.draw()
        plt.pause(0.01)

    term = 'goal' in info[0].keys()

    print('Step {} Entropy {:3f} reward {:2f} value {:3f} done {}, bad_transition {} total reward {}'.format(
                step,dist_entropy.item(),reward.item(),value.item(), done, term, episode_reward.item()))

    if done:
        print("EPISODE: {} steps: ", episode_reward, step, flush=True)
        obs = env.reset()
        step = 0
        episode_reward = 0
        done = False
# drop redundant frames if needed
#all_obs = [x[0,:, :, -3:] for x in  all_obs]
#os.makedirs("./temp/", exist_ok=True)
#for i, x in enumerate(all_obs):
#     cv2.imwrite("./temp/{:05d}.png".format(i), x[:,:, ::-1])
