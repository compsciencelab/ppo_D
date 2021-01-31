import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from shutil import copy2
import random
from torch.distributions.categorical import Categorical
import csv


class LossWriter(object):
    def __init__(self, log_dir, fieldnames = ('r', 'l', 't'), header=''):
        
        assert log_dir is not None
        
        os.mkdir(log_dir + '/loss_monitor')
        filename = '{}/loss_monitor/loss_monitor.csv'.format(log_dir)
        self.f = open(filename, "wt")
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, training_info):
        if self.logger:
            self.logger.writerow(training_info)
            self.f.flush()


class PPOKL():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 actor_behaviors=None,
                 vanilla = None,
                 behaviour_cloning = None,
                 ppo_bc = None,
                 test = None,
                 log_dir = ''):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.actor_behaviors=actor_behaviors
        if self.actor_critic.gail:
            all_parameters = [para for para in actor_critic.parameters()] + [para for para in actor_critic.disc.parameters()]
            self.optimizer = optim.Adam(all_parameters , lr=lr, eps=eps)
            
           
        else:
            self.optimizer = optim.Adam(actor_critic.parameters() , lr=lr, eps=eps)
       
        self.vanilla = vanilla
        self.behaviour_cloning = behaviour_cloning
        self.ppo_bc = ppo_bc
        self.impala = False
        self.test = test
        self.loss_writer = LossWriter(log_dir, fieldnames = ('BC_loss', 'V_loss', 'action_loss', 'gail_loss', 'entropy'))



    def update(self, rollouts):

    

        if self.vanilla:
            advantages = rollouts.returns[:-1]
        else:
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        kl_div_epoch = 0
        bc_loss_epoch = 0
        gail_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator( advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator( advantages, self.num_mini_batch)
            
            # assert if self.impala then batch_size = 1

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, is_demo_batch = sample
                
#                 if self.actor_critic.gail:
#                     import ipdb; ipdb.set_trace()
#                     obs_batch=obs_batch[is_demo_batch[:,0].bool()] 
#                     recurrent_hidden_states_batch = recurrent_hidden_states_batch[is_demo_batch[:,0].bool()]
#                     actions_batch = actions_batch[is_demo_batch[:,0].bool()]
#                     value_preds_batch = value_preds_batch[is_demo_batch[:,0].bool()]
#                     return_batch = return_batch[is_demo_batch[:,0].bool()]
#                     masks_batch = old_action_log_probs_batch[is_demo_batch[:,0].bool()]
                    

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, dist_a = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,actions_batch)
            
                if self.actor_critic.continous:
                    _, action_ppo_bc, _, _, _ = self.actor_critic.act(
                    obs_batch, recurrent_hidden_states_batch, masks_batch)
                    action_ppo_bc = action_ppo_bc[is_demo_batch.squeeze(1) == 1, :]

                if self.ppo_bc:
                    
                    ratio = torch.exp(action_log_probs[is_demo_batch == 0] - old_action_log_probs_batch[is_demo_batch == 0])
                    surr1 = ratio * adv_targ[is_demo_batch == 0]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ[is_demo_batch == 0]
                    action_loss = -torch.min(surr1, surr2).mean()
                    
                    
                else:
                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ

                    if self.vanilla:
                        #action_loss = -surr1.mean()
                        surr1 = ratio*adv_targ
                        action_loss = -surr1.mean()
                    else:
                        action_loss = -torch.min(surr1, surr2).mean()
                
                if self.vanilla:
                    #action_loss = -surr1.mean()
                    surr1 = ratio*adv_targ
                    action_loss = -surr1.mean()
                else:
                    action_loss = -torch.min(surr1, surr2).mean()




                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                
                # behavioural cloning loss
                if self.behaviour_cloning or self.ppo_bc:
                    loss = nn.MSELoss()
                    if not self.actor_critic.continous:
                        loss = nn.CrossEntropyLoss()
                        
                        actions_prob = self.actor_critic.actions_prob(obs_batch, recurrent_hidden_states_batch, masks_batch,actions_batch)
                        actions_prob_demo = actions_prob[(is_demo_batch == 1).squeeze()]
                        
                    actions_batch_demo = actions_batch[is_demo_batch.squeeze(1) == 1, :]
                    #actions_batch_demo = actions_batch[is_demo_batch == 1]
                    
                    if not self.actor_critic.continous:
                        
                        if actions_batch_demo.shape[0] != 0:
                            BC_loss = loss(actions_prob_demo,actions_batch_demo.view(-1))
                        else:
                            BC_loss = 0
                    else:
                        if actions_batch_demo.shape[0] != 0:
                            BC_loss = loss(action_ppo_bc, actions_batch_demo)
                        else:
                            BC_loss = 0
              
                if self.behaviour_cloning:
                    loss = BC_loss + value_loss * self.value_loss_coef
                    
                elif self.ppo_bc:
                    loss = value_loss * self.value_loss_coef + action_loss + BC_loss
                    
                else:
                    loss = value_loss * self.value_loss_coef + action_loss
                
                kl_div = 0 * torch.as_tensor(loss)
                if self.actor_behaviors is not None:
                    for behavior in self.actor_behaviors:
                        with torch.no_grad():
                            _, _, _, _, dist_b= behavior.evaluate_actions(
                                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                            high_level_action_scaling = torch.exp( - 2 * dist_b.entropy() ).detach()  # just a way to scale. Look for better ways
                        kl_div += (high_level_action_scaling * kl_divergence(dist_b, dist_a)).mean()
                    loss += kl_div * self.entropy_coef / len(self.actor_behaviors)
                else:
                    loss +=  - dist_entropy * self.entropy_coef
                if self.test:
                    loss = 0*loss
                

                if self.actor_critic.rnd:
                    Ri = self.actor_critic.RND.get_reward(obs_batch)
                    self.actor_critic.RND.update(Ri)
                
                if self.actor_critic.gail:
                    
                    obs_batch_pi = obs_batch[~ is_demo_batch[:,0].bool()]
                    actions_batch_pi = actions_batch[~ is_demo_batch[:,0].bool()]
                    if obs_batch_pi.shape[0] > 0:
                        prob_pi, _ = self.actor_critic.disc(obs_batch_pi, actions_batch_pi)
                    else:
                        prob_pi = 0
                        
                    obs_batch_exp = obs_batch[is_demo_batch[:,0].bool()]
                    actions_batch_exp = actions_batch[is_demo_batch[:,0].bool()]
                    if obs_batch_exp.shape[0] > 0:
                        _, prob_exp = self.actor_critic.disc(obs_batch_exp, actions_batch_exp)
                    else:
                        prob_exp = 0
                    disc_loss = -torch.mean(prob_pi + prob_exp)
                    loss += disc_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                
                if self.actor_critic.gail:
                    gail_loss_epoch += disc_loss.item()
                if self.behaviour_cloning:
                    bc_loss_epoch += BC_loss.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                kl_div_epoch += kl_div.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kl_div_epoch /= num_updates
        bc_loss_epoch /= num_updates
        gail_loss_epoch /= num_updates
        
        training_info = {'BC_loss':bc_loss_epoch , 'V_loss': value_loss_epoch, 'action_loss': action_loss_epoch, 'gail_loss':gail_loss_epoch,'entropy': dist_entropy_epoch}
        self.loss_writer.write_row(training_info)
        mean_loss = value_loss_epoch * self.value_loss_coef +  action_loss_epoch 
        mean_loss += dist_entropy_epoch*self.entropy_coef if self.actor_behaviors is None else kl_div_epoch*self.entropy_coef 
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kl_div_epoch, mean_loss



def ppo_rollout_imitate(num_steps, envs, actor_critic, rollouts, infos_in):
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])
        
            if step == 0:
                action = action*0

        for i in range(len(infos_in)):
            infos_in[i]['value'] = value[i].cpu().numpy()

        action_ = (action, infos_in)
        obs, reward, done, infos = envs.step(action_)
        
        if (step +1) % 1000 == 0:
            done = np.ones(done.shape, dtype=bool) 
        else:
            done = np.zeros(done.shape, dtype=bool) 
            
        #print(done, step + 1)
        
        for info_in in infos_in:
            info_in["demo_in"] = []
            info_in["demo_value_in"] = []


        with torch.no_grad():
            
            is_demos = torch.zeros(reward.shape, dtype=torch.int32, device= action.device)
            
            
            for ii,info in enumerate(infos):
#                 print(info['true_action'])
                if info['true_action']:
                    
                    if len(info['action'].shape)> 0:
                        action[ii] =  torch.Tensor(info['action'])
                    else:
                        action[ii] =  int(info['action'])
                    action_log_prob[ii] = 0
                    is_demos[ii] = 1
                
                    
                demo_out = info['demo_out']
                if demo_out:
                    for kk in range(len(infos_in)):
                        if kk != ii:
                            infos_in[kk]['demo_in'].append(demo_out)


                demo_value_out = info['demo_value_out']
                if demo_value_out:
                    for kk in range(len(infos_in)):
                        if kk != ii:
                            infos_in[kk]['demo_value_in'].append(demo_value_out)
                            

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks, is_demos)

    return infos_in








def ppo_rollout_RND(num_steps, envs, actor_critic, rollouts,rnd_weight):
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])


        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        # Intrinsic Reward
        with torch.no_grad():
            Ri = actor_critic.RND.get_reward(obs).cpu().unsqueeze(1)
            reward += Ri*rnd_weight

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)

def ppo_rollout(num_steps, envs, actor_critic, actor_critic_expert, rollouts):
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            _, _, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])

            value, action, _, _, _ = actor_critic_expert.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)


def ppo_rollout_2(num_steps, envs, actor_critic, actor_critic_expert, rollouts, ratio=0.4):
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():

            if random.random() > ratio:
                _, _, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                    rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])

                value, action, _, _, _ = actor_critic_expert.act(
                    rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])
            else:
                value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic_expert.act(
                    rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)

def ppo_rollout_mix(num_steps, envs, actor_critic, rollouts,actor_behaviors = None):
    entropy_list= []
    for i in range(len(actor_behaviors)):
        entropy_list.append([])
        step = 0
    while True:

        if actor_behaviors is not None:

            values_array = np.zeros(2)

            with torch.no_grad():
                for i,behavior in enumerate(actor_behaviors):
                    value, action, action_log_prob, recurrent_hidden_states, _ = behavior.act(rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])
                    value, _, _, _, dist = behavior.evaluate_actions(rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step],action)
                    q = 1/(step+1)
                    values_array[i] = (q) * values_array[i] + (1-q) * dist.entropy().cpu().numpy()  

                with torch.no_grad():
                    print(np.argmin(values_array))
                    value, action, action_log_prob, recurrent_hidden_states, _  = actor_behaviors[np.argmin(values_array)].act(rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])


            """ with torch.no_grad():
                for i,behavior in enumerate(actor_behaviors):
                    
                    value, action, action_log_prob, recurrent_hidden_states, _ = behavior.act(rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])
                    value, _, _, _, dist = behavior.evaluate_actions(rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step],action)
                    
                    values_list.append(torch.exp(value))
                    probs_list.append(dist.probs)
                    entropy_list[i].append(dist.entropy().unsqueeze(0))
                    dist_list.append(dist)
            
                entropy_list = [ele / sum(values_list) for ele in values_list]
                entropy_list = [ele / sum(entropy_list)  for ele in entropy_list]
                
                weights = sum([a*b for a,b in zip(entropy_list,probs_list)])
                dist = Categorical(probs = weights)
                action = dist.sample()
                entropy_list_mean = [sum(i)/len(i) for i in entropy_list]

            
            min_tensor = torch.cat(entropy_list_mean,0) == torch.min(torch.cat(entropy_list_mean,0),0)[0]
            

            min_2 = [i .squeeze(0).unsqueeze(1) for i in torch.split(min_tensor,1)]
            min_3 = [i.repeat(1,7) for i in min_2]
            weights = sum([a*b for a,b in zip(min_3,probs_list)])
            dist = Categorical(probs = weights)
            action = dist.sample() """


        else:
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                    rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        if done:
            break
        
        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)


def ppo_update(agent, actor_critic, rollouts, use_gae, gamma, gae_lambda, use_proper_time_limits):
    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)
    value_loss, action_loss, dist_entropy, kl_div, loss = agent.update(rollouts)
    rollouts.after_update()
    return value_loss, action_loss, dist_entropy, kl_div, loss


def ppo_save_model(actor_critic, fname, iter):
    #avoid overwrite last model for safety
    torch.save(actor_critic.state_dict(), fname + ".tmp")  
    os.rename(fname + '.tmp', fname)
    copy2(fname,fname+".{}".format(iter))
