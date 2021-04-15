import os
import sys
import gym
import glob
from os.path import join
import random
import numpy as np
from gym import error, spaces
from baselines.bench import load_results
from baselines import bench
from gym.spaces.box import Box
import animalai
from animalai.envs.gym.environment import AnimalAIEnv
import time
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import ActionFlattener
from ppo.envs import FrameSkipEnv, TransposeImage
from PIL import Image
from wrappers import RetroEnv, Stateful, FilterActionEnv
import os.path as osp


def make_animal_env(log_dir, inference_mode, frame_skip, arenas_dir, 
                    info_keywords, reduced_actions, seed, state, rho, phi,
                    record_actions, demo_dir, size_buffer, size_buffer_V):
    base_port = 100*seed  # avoid collisions

    def make_env(rank):
        def _thunk():

            if 'DISPLAY' not in os.environ.keys():
                os.environ['DISPLAY'] = ':0'
            exe = os.path.join(os.path.dirname(animalai.__file__),'../../env/AnimalAI')
            env = AnimalAIEnv(environment_filename=exe, retro=False,
                              worker_id=base_port+rank, docker_training=False,
                              seed=seed, n_arenas=1, arenas_configurations=None,
                              greyscale=False, inference=inference_mode, 
                              resolution=None)
            env = RetroEnv(env)
            if reduced_actions:
                env = FilterActionEnv(env)
            if record_actions: 
                env = LabAnimalRecordAction(env, arenas_dir, rho, record_actions)
            else:
                env = LabAnimalReplayRecord(env, arenas_dir, rho, phi, demo_dir,
                                            size_buffer, size_buffer_V)
            if state:
                env = Stateful(env)

            if frame_skip > 0:
                env = FrameSkipEnv(env, skip=frame_skip)   #TODO:Is this wrong here? Are we double counting rewards? Infos?
                print("Frame skip: ", frame_skip, flush=True)

            if log_dir is not None:
                env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                    allow_early_resets=False,
                                    info_keywords=info_keywords)

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
               env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env


class ReplayAll():
    def __init__(self, arenas, rho, phi, demo_dir, size_buffer, size_buffer_V):
        self.rho = rho
        self.phi = phi
        self.phi_ = phi
        self.rho_ = rho
        self.replay = False

        self.demo_dir = demo_dir
        self.files = glob.glob("{}/*".format(demo_dir))

        self.recordings = [np.load(filename) for ii, filename in enumerate(self.files) if 100 >= ii ]
        self.size_V_buffer = size_buffer_V
        self.size_R_buffer = size_buffer
        self.size_buffer = size_buffer
        self.recordings_value = [] 
        self.min_value = 0
        self.value_list_index = None
        self.max_value = 0 
        self.mean_value = 0
        self.index_min = None
        self.deleted_index = []

    def replay_step(self, action, value):

        if self.replay == True:
            if self.num_steps > self.step:
                act = self.acts[self.step]
                obs = self.obs[self.step]
                reward = self.rews[self.step]
                if self.value_list_index:
                    if self.value_list_index not in self.deleted_index:
                        self.recordings_value[self.value_list_index]["values"][self.step] = value

                if self.step == (self.num_steps -1):
                    done = True
                    if self.value_list_index:
                        self.new_max_value = np.max(self.recordings_value[self.value_list_index]["values"])
                        self.max_value_error = self.new_max_value - self.old_max_value
                    else:
                        self.max_value_error = 0.0
                        self.new_max_value = 0.0     
                else: 
                    done = False
                    if self.value_list_index:
                        if self.value_list_index in self.deleted_index:
                            done = True
                            self.deleted_index = []
                            self.max_value_error = 0.0
                            self.new_max_value = 0.0      

                self.step += 1
                return [act, obs, reward, done]

            else:
                return [action]
        else:
            return [action]

    def reset(self, arena_name, average_performance):

        rho = self.rho
        phi = self.phi

        max_trajectory_value = np.array([np.max(record['values']) for record in self.recordings_value])
        self.ps_ = max_trajectory_value

        if len(self.ps_) == 0:
            ps = self.ps_
        else:
            self.max_value = np.max(self.ps_)
            ps = np.abs(np.min(self.ps_)) + self.ps_

        P = ps*10/np.sum(ps*10)

        if len(self.recordings) != 0:

            if len(self.recordings_value) == 0:
                coin_toss = random.choices([0,2], weights=[rho, 1 - rho])[0]
            else:
                coin_toss = random.choices([0,1, 2], weights=[rho, phi, 1- phi -rho])[0]

            if coin_toss == 0:
                recording = random.choice(self.recordings)
                self.replay = True

                self.acts = recording['actions']
                self.obs = recording['observations']
                self.rews = recording['rewards']
                self.num_steps = self.acts.shape[0]
                self.step = 0
                self.value_list_index = None

            elif coin_toss == 1:

                self.value_list_index = np.random.choice(np.arange(0,len(self.recordings_value)), p= P)

                recording = self.recordings_value[self.value_list_index]
                self.replay = True

                self.acts = recording['actions']
                self.obs = recording['observations']
                self.rews = recording['rewards']

                self.num_steps = self.acts.shape[0]
                self.step = 0
                self.old_max_value = np.max(recording['values'])

            else:
                self.value_list_index = None
                self.replay = False
        else:
            self.replay = False
            self.value_list_index = None

        return self.replay

    def add_demo(self, demo):

        self.recordings.insert(1,demo)

        if len(self.recordings) > self.size_R_buffer:
            self.recordings.pop()

        if self.size_V_buffer > 0:
            self.rho = self.rho + self.phi_/self.size_buffer
            self.phi = self.phi - self.phi_/self.size_buffer
            self.size_V_buffer = self.size_V_buffer - 1


        if (len(self.recordings_value) >= self.size_V_buffer) and len(self.recordings_value) > 0:
            max_trajectory_value = np.array([np.max(record['values']) for record in self.recordings_value])
            self.ps_ = max_trajectory_value
            self.min_value = np.min(self.ps_)
            self.index_min = np.argmin(self.ps_)
            self.recordings_value.pop(self.index_min)
            self.deleted_index.append(self.index_min)
            if self.value_list_index:
                if self.value_list_index > self.index_min:
                    self.value_list_index -= 1

    def add_demo_value(self, demo):
        if self.size_V_buffer > 0:
            if len(self.recordings_value) >= self.size_V_buffer:
                max_trajectory_value = np.array([np.max(record['values']) for record in self.recordings_value])
                self.ps_ = max_trajectory_value
                self.min_value = np.min(self.ps_)
                self.max_value = np.max(self.ps_)
                self.mean_value = np.mean(self.ps_)
                self.index_min = np.argmin(self.ps_)
                self.recordings_value[self.index_min] = demo
                self.deleted_index.append(self.index_min)

            else:
                self.recordings_value.append(demo)


class LabAnimalReplayRecord(gym.Wrapper):
    def __init__(self, env, arenas_dir, rho, phi, demo_dir, size_buffer, size_buffer_V):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            #files = glob.glob("{}/*/*.yaml".format(arenas_dir)) + glob.glob("{}/*.yaml".format(arenas_dir))
            files = glob.glob("{}/*.yml".format(arenas_dir)) + glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.replayer = ReplayAll(files,rho, phi, demo_dir, size_buffer, size_buffer_V)
        self.performance_tracker = np.zeros(1000)
        self.n_arenas = 0
        self.directory = demo_dir
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.value_rollouts = []
        self.demo = None
        self.demo_value = None

    def step(self, action_):

        action, info_in = action_

        out = self.replayer.replay_step(action, info_in['value'])
        info = {}
        if len(out) == 1:
            obs, reward, done, info = self.env.step(action)
            self.obs_rollouts.append(obs)
            self.rews_rollouts.append(reward)
            self.actions_rollouts.append(action)
            self.value_rollouts.append(info_in['value'])
            # get rid of negative rewards ! 
            if reward < 0:
                reward = 0 

            self.env_reward_no_D += reward
            self.len_real +=1

            vec_obs = info['vector_obs'].tolist()

            info['true_action'] = False
            info['action'] = action
            self.max_value = max(self.max_value, info_in['value'][0])
            info['value'] = self.max_value 
            info['max_value_error'] = 0.0
        else:
            action, obs, reward, done = out

            if (reward > -0.01) and (reward < 0):
                reward = 0 # get rid of the time reward
            info['true_action'] = True
            info['action'] = action

            if done:
                info['max_value_error'] = self.replayer.max_value_error
                info['value'] = self.replayer.new_max_value
            else:
                info['max_value_error'] = 0.0
                info['value'] = 0.0
            self.len_real = 0
        self.steps += 1

        self.env_reward += reward

        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        info['reward_woD'] = self.env_reward_no_D
        info['len_real'] = self.len_real
        info['min_value'] = self.replayer.min_value
        info['max_value'] = self.replayer.max_value
        info['mean_value'] = self.replayer.mean_value

        if self.demo:
            info['demo_out'] = self.demo
            self.demo = None
        else:
            info['demo_out'] = None

        for demo_in_ in info_in["demo_in"]:

            if len(demo_in_)> 0:
                self.replayer.add_demo(demo_in_)

        if self.demo_value:
            info['demo_value_out'] = self.demo_value
            self.demo_value = None
        else:
            info['demo_value_out'] = None

        for demo_in_value_ in info_in["demo_value_in"]:

            if len(demo_in_value_)> 0:
                self.replayer.add_demo_value(demo_in_value_)
        return obs, reward, done, info

    def reset(self, **kwargs):

        self.n_arenas += 1
        self.steps = 0
        self.env_reward = 0
        self.env_reward_no_D = 0
        self.len_real = 0
        self.max_value = 0

        if (len (self.actions_rollouts) > 0) and (sum(self.rews_rollouts) > 0.5):
            arena_name =  self._arena_file.split('/')[-1].split('.')[0]
            self.filename = '{}/{}_{}'.format(self.directory , arena_name, random.getrandbits(50))

            self.demo = {'name':self.filename,'observations':np.array(self.obs_rollouts),
                            'rewards':np.array(self.rews_rollouts),
                            'actions':np.array(self.actions_rollouts)}

        if (len (self.actions_rollouts) > 0) and (max(self.value_rollouts) > self.replayer.min_value) and (sum(self.rews_rollouts) <= 0):

            arena_name =  self._arena_file.split('/')[-1].split('.')[0]
            self.filename = '{}/{}_{}'.format(self.directory , arena_name, random.getrandbits(50))

            self.demo_value = {'name':self.filename, 'observations':np.array(self.obs_rollouts),
                            'rewards':np.array(self.rews_rollouts),
                            'actions':np.array(self.actions_rollouts),
                            'values': np.array(self.value_rollouts)}

        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.value_rollouts = []

        average_performance = np.average(self.performance_tracker)
        replay = self.replayer.reset(self._arena_file, average_performance)

        if self.demo:
            self.replayer.add_demo(self.demo)

        if self.demo_value:
            self.replayer.add_demo_value(self.demo_value)

        self._arena_file, arena = random.choice(self.env_list)

        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs) 


def random_size_reward():
    #according to docs it's 0.5-5
    s = random.randint(5, 50)/10
    return (s,s,s)

from animalai.envs.arena_config import Vector3

def set_reward_arena(arena, force_new_size = False):
    tot_reward = 0
    max_good = 0
    goods = []
    goodmultis = []
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal','GoodGoalBounce']:
            if len(i.sizes)==0 or force_new_size:
                x,y,z = random_size_reward() 
                i.sizes = [] #remove previous size if there
                i.sizes.append(Vector3(x,y,z))
            max_good = max(i.sizes[0].x,max_good)
            goods.append(i.sizes[0].x)
        if i.name in ['GoodGoalMulti','GoodGoalMultiBounce']:
            if len(i.sizes)==0 or force_new_size: 
                x,y,z = random_size_reward() 
                i.sizes = [] #remove previous size if there
                i.sizes.append(Vector3(x,y,z))
            tot_reward += i.sizes[0].x
            goodmultis.append(i.sizes[0].x)  

    tot_reward += max_good
    goods.sort()
    goodmultis.sort()
    return tot_reward


class LabAnimalReplayAll(gym.Wrapper):
    def __init__(self, env, arenas_dir, replay_ratio, demo_dir):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*/*.yaml".format(arenas_dir)) + glob.glob("{}/*.yaml".format(arenas_dir))
            
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.replayer = ReplayAll(replay_ratio,files, demo_dir)
        self.performance_tracker = np.zeros(1000)
        self.n_arenas = 0

    def step(self, action):
        out =self.replayer.replay_step(action)
        info = {}
        if len(out) == 1:
            obs, reward, done, info = self.env.step(action)
            if (reward > -0.01 ) and (reward < 0):
                reward = 0 # get rid of the time reward
            
            self.env_reward_no_D += reward
            info['action'] = 99
        else:
            action, obs, reward, done = out
            
            if (reward > -0.01 ) and (reward < 0):
                reward = 0 # get rid of the time reward
        
            info['action'] = action
                  
        self.steps += 1

        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        info['reward_woD'] = self.env_reward_no_D
        if done:
            self.performance_tracker[self.n_arenas % 1000] = max(self.env_reward_no_D, 0)/self.max_reward

        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.n_arenas += 1
        self.steps = 0
        self.env_reward = 0
        self.env_reward_no_D = 0
        
        """  while True:
            self._arena_file, arena = random.choice(self.env_list)
            replay = self.replayer.reset(self._arena_file)
            if replay:
                break """
        self._arena_file, arena = random.choice(self.env_list)

        average_performance = np.average(self.performance_tracker)
        replay = self.replayer.reset(self._arena_file, average_performance)
#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs)    

class LabAnimal(gym.Wrapper):
    def __init__(self, env, arenas_dir, replay_ratio):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.replayer = ReplayActions(replay_ratio)

    def step(self, action):
        action_ =self.replayer.replay_step(action)
        obs, reward, done, info = self.env.step(action_)
        self.steps += 1
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        info['action'] = action_ 
        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.steps = 0
        self.env_reward = 0
        self._arena_file, arena = random.choice(self.env_list)
        self.replayer.reset(self._arena_file)
#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs)


class LabAnimalRecordAction(gym.Wrapper):
    def __init__(self, env, arenas_dir, replay_ratio, record_actions):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir)) + glob.glob("{}/*.yml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.directory = record_actions
        self.arena_num = 0

    def step(self, action):
        
        #action_ =self.replayer.replay_step(action)
        action, other = action
        obs, reward, done, info = self.env.step(action)
        self.obs_rollouts.append(obs)
        self.rews_rollouts.append(reward)
        self.actions_rollouts.append(action)
        self.steps += 1
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        
        return obs, reward, done, info        

    def reset(self, **kwargs):
        
        if (len (self.actions_rollouts) > 0) and (self.env_reward > 0) :
            arena_name =  self._arena_file.split('/')[-1].split('.')[0]
            self.filename = '{}/{}'.format( self.directory ,arena_name)

            print(os.path.exists('{}.npz'.format(self.filename)))
            print(self.filename)
            if not os.path.exists('{}.npz'.format(self.filename)):
            
                np.savez(self.filename,observations=np.array(self.obs_rollouts),
                            rewards=np.array(self.rews_rollouts),
                            actions=np.array(self.actions_rollouts))
            self.arena_num += 1

        self._arena_file, arena = random.choice(self.env_list)
        #self._arena_file, arena   = self.env_list[self.arena_num % len(self.env_list)]
        self.steps = 0
        self.env_reward = 0
        
        #self.replayer.reset(self._arena_file)
        #self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t

        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []

        return self.env.reset(arenas_configurations=arena,**kwargs)


def analyze_arena(arena):
    tot_reward = 0
    max_good = 0
    goods = []
    goodmultis = []
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal','GoodGoalBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            max_good = max(i.sizes[0].x,max_good)
            goods.append(i.sizes[0].x)
        if i.name in ['GoodGoalMulti','GoodGoalMultiBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            tot_reward += i.sizes[0].x
            goodmultis.append(i.sizes[0].x)  

    tot_reward += max_good
    goods.sort()
    goodmultis.sort()
    return tot_reward