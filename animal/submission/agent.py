import sys
sys.path.append('.')
import torch
from ppo.model import Policy
from ppo.model import CNNBase,FixupCNNBase,ImpalaCNNBase,StateCNNBase
from ppo.envs import  VecPyTorch, VecPyTorchFrameStack, FrameSkipEnv, TransposeImage, VecPyTorchStateStack
from ppo.wrappers import RetroEnv,Stateful,FilterActionEnv
from animalai.envs.gym.environment import ActionFlattener
from PIL import Image
from ppo.envs import VecPyTorchFrameStack, TransposeImage, VecPyTorch, VecPyTorchState
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from ppo.vision_model import ImpalaCNNVision 
from ppo.object_model import ImpalaCNNObject
from ppo.wrappers import VecVisionState, VecObjectState
import numpy as np
from gym.spaces import Box
import gym

class FakeAnimalEnv(gym.Env):
 
    def set_step(self,obs,reward,done,info):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info

    def step(self, action_unused):
        self.steps += 1
        self.env_reward += self.reward
        return self.obs,self.reward,self.done,self.info

    def set_maxtime(self,max_time):
        self.max_time = max_time

    def reset(self):
        self.steps = 0
        self.env_reward = 0
        return (np.zeros((84,84,3),dtype=np.float32),None)


frame_skip = 2
frame_stack = 4
state_stack = 16
#CNN=FixupCNNBase
CNN=StateCNNBase
reduced_actions = True
recurrent = True 
vision_module_file = None 
#vision_module_file = '/aaio/data/model_249.ckpt'
object_module_file = None


def make_env():
    env = FakeAnimalEnv()
    env = RetroEnv(env)
    if reduced_actions:
       env = FilterActionEnv(env)
    if state_stack:
       env = Stateful(env)
    if frame_skip > 0:
        env = FrameSkipEnv(env, skip=frame_skip)
    env = TransposeImage(env, op=[2, 0, 1])
    return env


class Agent(object):

    def __init__(self, device='cpu'):
        """
         Load your agent here and initialize anything needed
        """
        print(device)
        envs = DummyVecEnv([make_env])
        envs = VecPyTorch(envs, device)
        envs = VecPyTorchFrameStack(envs, frame_stack, device)
        if CNN==StateCNNBase:
            state_shape = (13,) if reduced_actions else (15,)
            envs = VecPyTorchState(envs,state_shape)
            envs = VecPyTorchStateStack(envs,state_stack)
        else:
            state_shape = None

        if vision_module_file:
            vision_module, _ = ImpalaCNNVision.load(vision_module_file,device=device)
            vision_module.to(device)
            envs = VecVisionState(envs, vision_module)

        if object_module_file:
            object_module, _ = ImpalaCNNObject.load(object_module_file ,device=device)
            object_module.to(device)
            envs = VecObjectState(envs, object_module)

        self.envs = envs
        self.flattener = self.envs.unwrapped.envs[0].flattener
        # Load the configuration and model using *** ABSOLUTE PATHS ***
        self.model_path = '/aaio/data/animal.state_dict'
        base_kwargs={'recurrent': recurrent}
        self.policy = Policy(self.envs.observation_space,self.envs.action_space,base=CNN,base_kwargs=base_kwargs)
        self.policy.load_state_dict(torch.load(self.model_path,map_location=device))
        self.policy.to(device)
        self.recurrent_hidden_states = torch.zeros(1, self.policy.recurrent_hidden_state_size).to(device)
        self.masks = torch.zeros(1, 1).to(device)  # set to zero
        self.device = device

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.envs.reset()
        self.envs.unwrapped.envs[0].unwrapped.set_maxtime(t)
        self.recurrent_hidden_states = torch.zeros(1, self.policy.recurrent_hidden_state_size).to(self.device)
        self.masks = torch.zeros(1, 1).to(self.device)

    def step(self, obs, reward, done, info):
        """
        A single step the agent should take based on the current
        :param brain_info:  a single BrainInfo containing the observations and reward for a single step for one agent
        :return:            a list of actions to execute (of size 2)
        """
        self.envs.unwrapped.envs[0].unwrapped.set_step(obs,reward,done,info) #set obs,etc in fakenv
        obs, reward, done, info = self.envs.step( torch.LongTensor([[0]]) ) #apply transformations
        value, action, action_log_prob, self.recurrent_hidden_states, dist_entropy = self.policy.act(
            obs, self.recurrent_hidden_states, self.masks, deterministic=False)
        self.masks.fill_(1.0)
        action = self.flattener.lookup_action(int(action))
        return action


