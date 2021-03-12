"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum

import numpy as np
import gym

import rrc_simulation
from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube
from rrc_simulation import visual_objects
from rrc_simulation import TriFingerPlatform


class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = [
            self.observation_space[name].low.flatten()
            for name in self.observation_names
        ]

        high = [
            self.observation_space[name].high.flatten()
            for name in self.observation_names
        ]

        self.observation_space = gym.spaces.Box(
            low=np.concatenate(low), high=np.concatenate(high)
        )

    def observation(self, obs):

        #obs = {
        #    "robot_position": np.array(obs['observation']['position']),
        #    "robot_velocity": np.array(obs['observation']['velocity']),
        #    "robot_tip_positions": np.array(self.env.platform.forward_kinematics(
        #        obs['observation']['position'])),
        #    "object_position": np.array(obs['achieved_goal']['position']),
        #    "object_orientation": np.array(obs['achieved_goal']['orientation']),
        #    "goal_object_position": np.array(obs['desired_goal']['position']),
        #}

        observation = [obs[name].flatten() for name in self.observation_names]

        observation = np.concatenate(observation)
        return observation


class ActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        tmp = {}
        for name in self.observation_names:
            tmp[name]=self.observation_space[name]
        tmp['previous_action']=self.action_space
        self.observation_names.append("previous_action") 
        # Dict cannot be extended, only created
        self.observation_space = gym.spaces.Dict(tmp)

    def step(self,action):
        observation, reward, is_done, info = self.env.step(action)
        observation['previous_action']=action 
        return observation, reward, is_done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        #other possibility is to use obs["robot_position"] as to say did not move
        obs["previous_action"]=np.zeros(self.action_space.shape,dtype=np.float32)
        return obs
    
class ActionClipper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        

    def step(self,action):
        action =    np.clip(action, self.action_low, self.action_high)
        observation, reward, is_done, info = self.env.step(action)

        return observation, reward, is_done, info

    
    