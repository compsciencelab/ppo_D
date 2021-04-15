import os
import sys
import gym
import torch
import glob
from os.path import join
import random
import numpy as np
from gym import error, spaces
from baselines.bench import load_results
from baselines import bench
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper
import animalai
from animalai.envs.gym.environment import AnimalAIEnv
import time
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import ActionFlattener
from ppo.envs import FrameSkipEnv,TransposeImage
from PIL import Image

STATEFUL_BASE_SIZE = 1+3+1+1 # and hotbit for actions
class Stateful(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        # self.observation_space = spaces.Dict(
        #         {'obs': env.observation_space,
        #          'timeleft': spaces.Box(low=0, high=1, shape=()),
        #          'speed': spaces.Box(low=0, high=10, shape=()) ,
        #          'direction': spaces.Box(low=-1, high=1, shape=(3,))})

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        vel = info['vector_obs']
        mag = np.sqrt(vel.dot(vel))
        timeleft = (self.max_time - self.steps)/1000 #normalized to a fixed time unit (0.25, 0.5, 1.0)
        o = vel/mag if mag>0 else vel
        state = np.array([mag,o[0],o[1],o[2],timeleft,self.env_reward],dtype=np.float32) 
        actions = np.zeros(self.action_space.n,dtype=np.float32)
        actions[action] = 1  #hotbit
        state = np.concatenate((state,actions))
        info['states'] = state
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class RetroEnv(gym.Wrapper):
    def __init__(self,env):
        gym.Wrapper.__init__(self, env)
        self.flattener = ActionFlattener([3,3])
        self.action_space = self.flattener.action_space
        self.observation_space = gym.spaces.Box(0, 255,dtype=np.uint8,shape=(84, 84, 3))

    def step(self, action): 
        action = int(action)
        action = self.flattener.lookup_action(action) # convert to multi
        obs, reward, done, info = self.env.step(action)  #non-retro
        visual_obs, vector_obs = self._preprocess_obs(obs)
        info['vector_obs']=vector_obs
        return visual_obs,reward,done,info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        visual_obs, _ = self._preprocess_obs(obs)
        return visual_obs

    def _preprocess_obs(self,obs):
        visual_obs, vector_obs = obs
        visual_obs = self._preprocess_single(visual_obs)
        visual_obs = self._resize_observation(visual_obs)
        return visual_obs, vector_obs

    @staticmethod
    def _preprocess_single(single_visual_obs):
            return (255.0 * single_visual_obs).astype(np.uint8)

    @staticmethod
    def _resize_observation(observation):
        """
        Re-sizes visual observation to 84x84
        """
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        return np.array(obs_image)



#{0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], 4: [1, 1], 5: [1, 2], 6: [2, 0], 7: [2, 1], 8: [2, 2]}
class FilterActionEnv(gym.ActionWrapper):
    """
    An environment wrapper that limits the action space.
    """
    _ACTIONS = (0, 1, 2, 3, 4, 5, 6)

    def __init__(self, env):
        super().__init__(env)
        self.actions = self._ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]


class VecVisionState(VecEnvWrapper):
    def __init__(self, venv, visnet):
        wos = venv.observation_space[1]  # wrapped state space
        #output_size = visnet.output_size
        output_size = visnet.posangles_size 
        low = np.concatenate((wos.low,   np.full((output_size,), -np.inf,dtype=np.float32)) )
        high = np.concatenate((wos.high, np.full((output_size,),  np.inf,dtype=np.float32)) )
        observation_space = gym.spaces.Tuple( 
                (venv.observation_space[0],
                 gym.spaces.Box(low=low, high=high, dtype=np.float32)) 
            )

        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

        self.visnet = visnet

    def step_wait(self):
        (viz,states), rews, news, infos = self.venv.step_wait()
        with torch.no_grad():
            posangles,_,h = self.visnet(viz[:,-self.visnet.num_inputs:,:,:])  #match network viz take the last obs
        states = torch.cat((states,posangles),dim=1)
        return (viz,states), rews, news, infos

    def reset(self):
        (viz,states) = self.venv.reset()
        with torch.no_grad():
            posangles,_,h = self.visnet(viz[:,-self.visnet.num_inputs:,:,:])  #match network viz take the last obs
        states = torch.cat((states,posangles),dim=1)
        return (viz,states)
    

class VecObjectState(VecEnvWrapper):
    def __init__(self, venv, objnet):
        wos = venv.observation_space[1]  # wrapped state space
        output_size = objnet.num_classes 
        low = np.concatenate((wos.low,   np.full((output_size,), -np.inf,dtype=np.float32)) )
        high = np.concatenate((wos.high, np.full((output_size,),  np.inf,dtype=np.float32)) )
        observation_space = gym.spaces.Tuple( 
                (venv.observation_space[0],
                 gym.spaces.Box(low=low, high=high, dtype=np.float32)) 
            )

        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

        self.objnet = objnet

    def step_wait(self):
        (viz,states), rews, news, infos = self.venv.step_wait()
        with torch.no_grad():
            _,classes,_,h = self.objnet(viz[:,-self.objnet.num_inputs:,:,:])  #match network viz take the last obs
        states = torch.cat((states,classes),dim=1)
        return (viz,states), rews, news, infos

    def reset(self):
        (viz,states) = self.venv.reset()
        with torch.no_grad():
            _,classes,_,h = self.objnet(viz[:,-self.objnet.num_inputs:,:,:])  #match network viz take the last obs
        states = torch.cat((states,classes),dim=1)
        return (viz,states)
  