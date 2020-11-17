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
import pandas as pd


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
            df2 = pd.DataFrame() 
            
            df['f']= df['l'].cumsum()/1000000
            
            
            df['perf']= df['ereward']/(df['max_reward'])
            df['perf'].where(df['perf']>0,0,inplace=True)
            df['goal'] = df['perf']>0.9  #guess a threadshold
            
            df2['len_real'] = df['l'][~((df['ereward'] != 0) & (df['reward_woD'] == 0))] 
            df2['f_real']= df2['len_real'].cumsum()/1000000
            df2['reward_woD'] = df['reward_woD'][~((df['ereward'] != 0) & (df['reward_woD'] == 0))] 
            df2['real_perf']= df2['reward_woD']/(df['max_reward'][~((df['ereward'] != 0) & (df['reward_woD'] == 0))])



            max_index = df['value'].argmax()

            X = eval(df['X'][max_index])[0:-2]
            Y = eval(df['Y'][max_index])[0:-2]
            Z = eval(df['Z'][max_index])[0:-2]

            plt.plot(X, Z)
            plt.plot(X[-1],Z[-1],"*",markersize=15)
            plt.xlim(0,40)
            plt.ylim(0,40)
            plt.savefig(my_dir + '/best-trajectory.jpg')


            num_good_traj = df2['real_perf'][df2['real_perf'] > 0].count()
            roll =500
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
            df[['f','min_value']].rolling(roll).mean().iloc[0:-1:40].plot('f','min_value', ax=ax,legend=False)
            df[['f','max_value']].rolling(roll).mean().iloc[0:-1:40].plot('f','max_value', ax=ax,legend=False)
            df[['f','mean_value']].rolling(roll).mean().iloc[0:-1:40].plot('f','mean_value', ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Min Value')
            #plt.xlim((0, xlim_))
            ax.grid(True)

            ax = plt.subplot(2, 2, 3)
            df2[['f_real','real_perf']].rolling(roll).mean().iloc[0:-1:40].plot('f_real','real_perf', ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Performance without Deomnstrations')
            #plt.xlim((0, xlim_))
            ax.grid(True)

            ax = plt.subplot(2, 2, 4)
            df[['l']].rolling(roll).mean().iloc[0:-1:40].plot(y='l', ax=ax,legend=False)
            ax.set_xlabel('N. episodes')
            ax.set_ylabel('Episode lenght')
            #plt.xlim((0, xlim_))
            ax.grid(True)
              
            fig.tight_layout() 
            ax.get_figure().savefig(my_dir + '/monitor-2.jpg')
            plt.clf()
        except Exception as e: 
            print(e) 
    print('Total number of succesful trajectories:', num_good_traj)
    sleep(360)



quit()



