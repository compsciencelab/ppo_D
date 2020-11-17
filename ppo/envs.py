import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from ppo.subproc_vec_env import MySubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import  VecNormalize as VecNormalize_

def make_vec_envs(make,num_processes,log_dir,device,num_frame_stack,state_shape,num_state_stack,spaces=None):

    envs = [make(i)  for i in range(num_processes)    ]

    if len(envs) > 1:
        #envs = SubprocVecEnv(envs)
        envs = MySubprocVecEnv(envs)
        #envs = ShmemVecEnv(envs,spaces=spaces, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if num_frame_stack > 0:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

    if state_shape:
        #tupled obs
        envs = VecPyTorchState(envs,state_shape)
        if num_state_stack>0:
            envs = VecPyTorchStateStack(envs,num_state_stack)
    
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions, infos_in = actions
        if isinstance(actions, torch.Tensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        actions = (actions, infos_in)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecPyTorchState(VecEnvWrapper):
#   Convert obs to tuple (obs,states) and making all available to device as torch arrays
    def __init__(self, venv, state_shape):
        self.venv = venv
        self._state_shape = state_shape
        observation_space = gym.spaces.Tuple( 
                (venv.observation_space, 
                 gym.spaces.Box(low=-np.inf, high=np.inf, shape= state_shape, dtype=np.float32)) 
            )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def reset(self):
        obses = self.venv.reset()  #TODO> really, I should also return vector obs here because they are available
        states = torch.zeros((self.venv.num_envs,) + self._state_shape).to(self.device)
        return (obses,states)

    def step_wait(self):
        obses, rews, dones, infos = self.venv.step_wait()
        states = np.zeros((self.venv.num_envs,) + self._state_shape,dtype=np.float32)
        for i, done in enumerate(dones):
            if done:
                states[i] = 0
            else:
                states[i] = infos[i]['states'] 
        states = torch.from_numpy(states.copy()).float().to(self.device) 
        return (obses,states), rews, dones, infos


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
# Takes obs image and stack them
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()



class VecPyTorchStateStack(VecEnvWrapper):
#Take tupled obs=(vis,state) and stack states
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space[1]  # wrapped state space
        self.shape_dim0 = wos.shape[0]
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stacked = torch.zeros((venv.num_envs, ) + low.shape).to(self.device)

        observation_space = gym.spaces.Tuple( 
                (venv.observation_space[0],
                 gym.spaces.Box(low=low, high=high, dtype=np.float32)) 
            )

        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        (obs,states), rews, news, infos = self.venv.step_wait()
        self.stacked[:, :-self.shape_dim0] =  self.stacked[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked[i] = 0
        self.stacked[:, -self.shape_dim0:] = states
        return (obs,self.stacked), rews, news, infos

    def reset(self):
        obs,states = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked = torch.zeros(self.stacked.shape)
        else:
            self.stacked.zero_()
        self.stacked[:, -self.shape_dim0:] = states
        return (obs,self.stacked)


class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        #max_frame = self._obs_buffer.max(axis=0)
        last_frame = obs

        return last_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

