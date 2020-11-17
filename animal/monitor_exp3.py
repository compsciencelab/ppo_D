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


args = get_args()
my_dir = args.log_dir
experiments_path = glob(my_dir+'/*/')
black_list = [
    #"exp_baseline",
    "exp_baseline_rnn",
    "exp_attention_space_mid",
    "exp_attention_space_fin",
    "exp_attention_time",
    #"exp_baseline_100",
    #"exp_attention_time_100",
]

experiment_names = [path.split('/')[-2] for path in experiments_path if
                    path.split('/')[-2] not in black_list]

#experiment_names = sorted(experiment_names, key=int)  



#experiment_names = ["0.3","0.1", "0.05","0"]

df = pd.DataFrame()


fig = plt.figure(figsize=(15, 9))

for num, experiment in enumerate(experiments_path):
    df = pd.DataFrame()
    #if experiment.split("/")[-2] not in black_list:
    exps = glob(experiment+'/*/')   
    #exps = glob(experiment)
    print(exps)

    for _, name in enumerate(exps):
        df_ = load_results(name)   
        df = df.append(df_)


    df['f']= df['l'].cumsum()/1000000
    df['perf']= df['ereward']/(df['max_reward'])
    df['perf'].where(df['perf']>0,0,inplace=True)
    df['goal'] = df['perf']>0.9  #guess a threadshold

    roll = 500
    total_time = df['t'].iloc[-1]
    total_steps = df['l'].sum()
    total_episodes = df['r'].size
    experiment_names[num] += " ({:.1f} h, FPS {:.1f})".format(total_time / 3600, total_steps/total_time)

    """ ax = plt.subplot(1, 2, 1)
    df[['f','r']].rolling(roll).mean().iloc[0:-1:40].plot('f','r',  ax=ax,legend=False)
    ax.set_xlabel('N. steps (M)')
    ax.set_ylabel('Reward')
    ax.grid(True)
    plt.legend(experiment_names, loc='best') """

    """ ax = plt.subplot(1, 1, 1)
    df[['f','perf']].rolling(roll).mean().iloc[0:-1:40].plot('f','perf', ax=ax,legend=False)
    ax.set_xlabel('N. steps (M)')
    ax.set_ylabel('Performance')
    ax.grid(True)
    plt.legend(experiment_names, loc='best') """

    ax = plt.subplot(1, 1, 1)
    df[['f','reward_woD']].rolling(roll).mean().iloc[0:-1:40].plot('f','reward_woD', ax=ax,legend=False)
    ax.set_xlabel('N. steps (M)')
    ax.set_ylabel('Reward without Deomnstrations')
    ax.grid(True)

    """ ax = plt.subplot(2, 2, 3)
    df[['f','goal']].rolling(roll).mean().iloc[0:-1:40].plot('f','goal', ax=ax,legend=False)
    ax.set_xlabel('N. steps (M)')
    ax.set_ylabel('Estimated evalai score')
    ax.grid(True)
    plt.legend(experiment_names, loc='best')

    ax = plt.subplot(2, 2, 4)
    df[['l']].rolling(roll).mean().iloc[0:-1:40].plot(y='l', ax=ax,legend=False)
    ax.set_xlabel('N. episodes')
    ax.set_ylabel('Episode lenght')
    ax.grid(True) """

    plt.legend(experiment_names, loc='best')
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)


# fig.tight_layout()
ax.get_figure().savefig(my_dir+'/performance.jpg')
plt.clf()
quit()