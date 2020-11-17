#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:51:35 2019

@author: gianni
"""

from  glob import glob
from time import sleep
from baselines.bench import load_results
from matplotlib import pylab as plt
import numpy as np
import argparse
import os


def get_args():

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--log-dir',default=None, help='dir save models and statistics')
    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args

num_good_traj = 0
xlim_ = 10
args = get_args()
my_dir = args.log_dir
exps = glob(my_dir+'*')
print(exps)

while True:
    for i,d in enumerate(exps):
        fig = plt.figure(i,clear=True, figsize=(15,9))
        try:
            df = load_results(d)
            #import ipdb; ipdb.set_trace()
            df['f']= df['l'].cumsum()/1000000
            df['f_real']= df['len_real'].cumsum()/1000000
            #import ipdb; ipdb.set_trace()
            df['perf']= df['ereward']/(df['max_reward'])
            df['perf'].where(df['perf']>0,0,inplace=True)
            df['goal'] = df['perf']>0.9  #guess a threadshold
            df['real_perf']= df['reward_woD']/(df['max_reward'])
            num_good_traj = df['real_perf'][df['real_perf'] > 0].count()
            roll =300
            total_time = df['t'].iloc[-1]
            total_steps = df['l'].sum()
            total_episodes = df['r'].size
             
            ax = plt.subplot(2, 2, 1)
            ax.set_title(' {} total time: {:.1f} h FPS {:.1f}'.format(d.upper(),total_time/3600, total_steps/total_time))
            df[['f','r']].rolling(roll).mean().iloc[0:-1:40].plot('f','r',  ax=ax,legend=False)
            df[['f','ereward']].rolling(roll).mean().iloc[0:-1:40].plot('f','ereward',  ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Reward')
            #plt.xlim((0, xlim_))
            ax.grid(True)
    
            ax = plt.subplot(2, 2, 2)
            df[['f','perf']].rolling(roll).mean().iloc[0:-1:40].plot('f','perf', ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Performance')
            #plt.xlim((0, xlim_))
            ax.grid(True)

            ax = plt.subplot(2, 2, 3)
            df[['f_real','real_perf']].rolling(roll).mean().iloc[0:-1:40].plot('f_real','real_perf', ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Reward without Deomnstrations')
            #plt.xlim((0, xlim_))
            ax.grid(True)

            ax = plt.subplot(2, 2, 4)
            df[['l']].rolling(roll).mean().iloc[0:-1:40].plot(y='l', ax=ax,legend=False)
            ax.set_xlabel('N. episodes')
            ax.set_ylabel('Episode lenght')
            #plt.xlim((0, xlim_))
            ax.grid(True)
              
            fig.tight_layout() 
            ax.get_figure().savefig(my_dir + '/monitor.jpg')
            plt.clf()
        except Exception as e: 
            print(e) 
    print('Total number of succesful trajectories:', num_good_traj)
    sleep(360)



quit()



