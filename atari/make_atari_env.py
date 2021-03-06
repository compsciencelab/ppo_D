import os
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from atari.wrappers import wrap_deepmind
from ppo.envs import TransposeImage
import gym
import sys
import numpy as np
import random
import glob
import bisect 

def make_atari_env(env_id, rho=0.1, phi=0.3, demo_dir= '', size_buffer=50, 
                   size_buffer_V=50, base_seed=0, log_dir=None, frame_skip=0, 
                   frame_stack=1, allow_early_resets=False, info_keywords=('ereward'), threshold_reward=None, test=False):

    def make_env(rank):

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        def _thunk():
            
            env = make_atari(env_id)
            
            env.seed(index_worker + rank) if base_seed is None else env.seed(
                base_seed + rank)
            # WARNING
           
            env = wrap_deepmind(
                env, episode_life=True if not test else False,
                clip_rewards=True if not test else False,
                scale=False)
            #env = TruncateAtari(env, threshold_reward)
            env = AtariReplayRecord(env,rho, phi, demo_dir, size_buffer, size_buffer_V, threshold_reward)
            
            if log_dir is not None:
                env = bench.Monitor(
                    env,
                    os.path.join(log_dir, str(rank)),
                    allow_early_resets=allow_early_resets,
                    info_keywords=('rr', 'rrr', 'lives') + info_keywords if not test else ())

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env

class TruncateAtari(gym.Wrapper):
    def __init__(self, env, threshold_reward):
        gym.Wrapper.__init__(self, env)
        self.sum_reward = 0
        self.threshold_reward = threshold_reward


    def step(self, action):

        #action, other = action
        obs, reward, done, info = self.env.step(action)
        self.sum_reward += reward
        #print(self.sum_reward)
        if self.sum_reward >= self.threshold_reward:
            self.sum_reward = 0
            done = True
            obs =  obs = self.env.reset()
            
            

        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.sum_reward = 0
        return self.env.reset(**kwargs)


class ReplayAll():
    def __init__(self, rho, phi, demo_dir, size_buffer, size_buffer_V):
        self.rho = rho
        self.phi = phi
        self.phi_ = phi
        self.rho_ = rho
        self.replay = False
        #self.flattener = ActionFlattener([3,3])

        self.demo_dir = demo_dir
        if len(demo_dir) > 0:
            self.files = glob.glob("{}/*".format(demo_dir))
            self.recordings = [np.load(filename) for ii, filename in enumerate(self.files) if 100 >= ii ]
        else:
            self.recordings = []
            
        self.n_original_demos = len(self.recordings)
        self.size_V_buffer = size_buffer_V
        self.size_R_buffer  = size_buffer
        self.size_buffer = size_buffer
        self.recordings_value = [] 
        self.min_value = 0
        self.value_list_index = None
        self.max_value = 0 
        self.mean_value = 0
        self.index_min = None
        self.deleted_index = []
        self.sum_reward = 0
        self.recordings_reward_sum = []



    def replay_step(self, action, value):
        
        if self.replay == True:
            if  self.num_steps > self.step:
                act = self.acts[self.step]
                obs = self.obs[self.step]
                reward = self.rews[self.step]
                self.sum_reward  += self.rews[self.step]
                if self.value_list_index:
                    if self.value_list_index not in self.deleted_index:
                        self.recordings_value[self.value_list_index]["values"][self.step] = value

                if self.step == (self.num_steps -1) or self.sum_reward >= 10:
                    done = True
                    self.sum_reward = 0
                    
                    if self.value_list_index:
                        self.new_max_value= np.max(self.recordings_value[self.value_list_index]["values"])
                        self.max_value_error = self.new_max_value - self.old_max_value
                    else:
                        self.max_value_error = 0.0
                        self.new_max_value= 0.0
                            
                else: 
                    done = False
                    if self.value_list_index:
                        if self.value_list_index in self.deleted_index:
                            done = True
                            self.deleted_index = []
                            self.max_value_error = 0.0
                            self.new_max_value= 0.0
                            
                        
                self.step +=1
                return [act, obs, reward, done]
                
            else:
                return [action]
        else:
            return [action]
    
    def reset(self):
        #print(self.max_value, "MAX VALUE")
#         print(len(self.recordings_value), "LEN VALUE BUFFER")
        print(len(self.recordings), "LEN REWARD BUFFER")
        
        rho = self.rho
        phi = self.phi

        
        #mean_trajectory_value = np.array([ np.mean(record['values']) for record in self.recordings_value])
        max_trajectory_value = np.array([ np.max(record['values']) for record in self.recordings_value])
        #self.ps_ = mean_trajectory_value*(1 - 0.9) + 0.9*max_trajectory_value
        self.ps_ = max_trajectory_value
     

        if len(self.ps_) == 0:
            ps = self.ps_
        else:
            self.max_value = np.max(self.ps_ )
            ps = np.abs(np.min(self.ps_)) + self.ps_ 
            
        P = ps*10/np.sum(ps*10)

        if len(self.recordings) != 0:
            
            if len(self.recordings_value) == 0:
                coin_toss = random.choices([0,2], weights=[rho, 1 - rho])[0]
            else:
                coin_toss = random.choices([0,1, 2], weights=[rho, phi, 1- phi -rho])[0]

            if  coin_toss == 0:
                
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
    
        
    def add_demo(self,demo, sum_rewards):
        insert_index = bisect.bisect(self.recordings_reward_sum, sum_rewards)
        
        self.recordings.insert(self.n_original_demos + insert_index,demo)
        self.recordings_reward_sum.insert(insert_index, sum_rewards)
            
#         print(sum_rewards, insert_index)
        
        if len(self.recordings) >  self.size_R_buffer:
            self.recordings.pop(self.n_original_demos)
            self.recordings_reward_sum.pop(0)

#         print('before:', sum(self.recordings[0]['rewards']),'after:' , sum(self.recordings[1]['rewards']))
#         print(self.recordings_reward_sum)
        
#         if self.size_V_buffer > 0:
#             self.rho = self.rho + self.phi_/self.size_buffer
#             self.phi = self.phi - self.phi_/self.size_buffer
#             self.size_V_buffer = self.size_V_buffer - 1
        

#         if (len(self.recordings_value) >=  self.size_V_buffer) and len(self.recordings_value) > 0:
#             max_trajectory_value = np.array([ np.max(record['values']) for record in self.recordings_value])
#             self.ps_ = max_trajectory_value
#             self.min_value = np.min(self.ps_ )
#             self.index_min = np.argmin(self.ps_)
#             self.recordings_value.pop(self.index_min)
#             self.deleted_index.append(self.index_min)
#             if self.value_list_index:
#                 if self.value_list_index > self.index_min:
#                     self.value_list_index -= 1
             
             


    def add_demo_value(self,demo):
        if self.size_V_buffer > 0:
            if len(self.recordings_value) >=  self.size_V_buffer:
                max_trajectory_value = np.array([ np.max(record['values']) for record in self.recordings_value])
                self.ps_ = max_trajectory_value
                self.min_value = np.min(self.ps_ )
                self.max_value = np.max(self.ps_ )
                self.mean_value = np.mean(self.ps_ )
                self.index_min = np.argmin(self.ps_)
                self.recordings_value[self.index_min] = demo
                self.deleted_index.append(self.index_min)

            else:
                self.recordings_value.append(demo)


                    
                    
class AtariReplayRecord(gym.Wrapper):
    def __init__(self, env, rho, phi, demo_dir, size_buffer, size_buffer_V, threshold_reward):# add a bunch of arguments here
       
        gym.Wrapper.__init__(self, env)
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.value_rollouts = []
        self.replayer = ReplayAll(rho, phi, demo_dir, size_buffer, size_buffer_V)
 
        self.demo = None
        self.demo_value = None
        self.threshold_reward = threshold_reward
        
    

    def step(self, action_):
        action, info_in = action_
        out =self.replayer.replay_step(action, info_in['value'])
        info = {}
        if len(out) == 1:
            obs, reward, done, info = self.env.step(action)
            obs[0:7,:,:] = 0
            self.obs_rollouts.append(obs)
            self.rews_rollouts.append(reward)
            self.actions_rollouts.append(action)
            self.value_rollouts.append(info_in['value'])
            # get rid of negative rewards ! 
#             if reward < 0:
#                 reward = 0

            self.env_reward_no_D += reward
            self.env_reward += reward

            
            if self.threshold_reward:
                if self.env_reward < self.threshold_reward:
                    reward_ = 0
                else:
                    reward_ = 50
            else:
                reward_ = reward
            
            
            self.len_real +=1
             
            info['true_action'] = False
            info['action'] = action
            
            self.max_value = max(self.max_value, info_in['value'][0])
            info['value'] = self.max_value 
            info['max_value_error'] = 0.0
        else:
            action, obs, reward, done = out
            obs[0:7,:,:] = 0
            
            self.env_reward += reward

            if self.threshold_reward:
                if self.env_reward < self.threshold_reward:
                    reward_ = 0    
                else:
                    reward_ = 50
            else:
                reward_ = reward

             
            #print(self.env_reward)
            info['action'] = action
            info['true_action'] = True
            info['rr'] = 0
            info['rrr'] = 0
            info['lives'] = 0
            
            if done:
                info['max_value_error'] = self.replayer.max_value_error
                info['value'] = self.replayer.new_max_value
            else:
                info['max_value_error'] = 0.0
                info['value'] = 0.0
            self.len_real  = 0
             
        self.steps += 1
        
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
                self.replayer.add_demo(demo_in_, sum(demo_in_['rewards']))

        
        if self.demo_value:
            info['demo_value_out'] = self.demo_value
            self.demo_value = None
        else:
            info['demo_value_out'] = None


        for demo_in_value_ in info_in["demo_value_in"]:

            if len(demo_in_value_)> 0:
                self.replayer.add_demo_value(demo_in_value_)
                
                
        return obs, reward_, done, info

    def reset(self, **kwargs):
        
        self.steps = 0
        self.env_reward = 0
        self.env_reward_no_D = 0
        self.len_real = 0
        self.max_value = 0

        if self.threshold_reward:
            threshold = self.threshold_reward
        else:
            threshold = 50
            
        sum_rews_rollouts = sum(self.rews_rollouts)
        if (len (self.actions_rollouts) > 0) and (sum_rews_rollouts >= 0):
            
            self.demo = {'observations':np.array(self.obs_rollouts),
                            'rewards':np.array(self.rews_rollouts),
                            'actions':np.array(self.actions_rollouts),
                            'values': np.array(self.value_rollouts)}
    
        if (len (self.actions_rollouts) > 0) and (max(self.value_rollouts) > self.replayer.min_value) and (sum(self.rews_rollouts) <= 0):
            

            
            self.demo_value = {'observations':np.array(self.obs_rollouts),
                            'rewards':np.array(self.rews_rollouts),
                            'actions':np.array(self.actions_rollouts),
                            'values': np.array(self.value_rollouts)}    
         
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.value_rollouts = []

        replay = self.replayer.reset()

        if self.demo:
            self.replayer.add_demo(self.demo, sum_rews_rollouts)
        
        if self.demo_value:
            self.replayer.add_demo_value(self.demo_value)


        return self.env.reset(**kwargs) 
    
