from calendar import c
from cmath import isnan
from enum import Enum
from math import nan
import math
from re import M, search
from time import time
from unittest import result
import matplotlib.pyplot as plt
import csv
import json
import os
import numpy as np
import argparse
import re

# default model, if no model is passed as argument
model = 'dqn-rs_Acrobot-v1'

parser = argparse.ArgumentParser("python plot_results.py")
parser.add_argument("model_dir_name", help="The directory name of the model to be plotted", type=str, nargs='?', default=model)
args = parser.parse_args()

if args.model_dir_name:
    model = args.model_dir_name

# parent directory that contains all trained models
model_directory = '/Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/trained/'

# one of the algorithms dqn, ppo or a2c
algorithm = model[:3]
# one of the search methods rs, smac or pbt (short rs_, sma and pbt)
search_method = model[4:7]

# default values
config = None
best_model_mean_reward = -1000
data_to_be_plotted = []

class PlotingOptions(Enum):
    PLOT_BEST_MODEL_ONLY = True

# plot everything on the same plot
ax = None
fig, ax = plt.subplots()
fig.suptitle(model)


xlimit = 10

# set y axis depending on the model (negative / positive rewards)
if 'CartPole-v1' in model:
    ax.axis([1, xlimit, 0, 600])
elif 'Acrobot-v1' in model:
    ax.axis([1, xlimit, 0, -600])
elif 'MountainCar-v0' in model:
    ax.axis([1, xlimit, 0, -250])

# walk through all trained model directories and plot the results of the one passed as argument
for subdir, dirs, files in os.walk(model_directory + model):
    for file in files:
        pattern = re.compile(r".*eval\.json")

        if file in ['result.json'] or pattern.match(file):
            rewards = []

            file_path = os.path.join(subdir, file)
            with open(file_path) as file:
                lines = file.readlines()
                lineCount = len(lines)
                
                for line in lines:
                    row = json.loads(line)

                    if "mean_reward" in row: # PBT
                        rewards.append(row["mean_reward"])    
                    else: # SMAC and RS
                        for reward in row["rewards"]:
                            rewards.append(reward)
                        break
            
            if len(rewards) < xlimit and PlotingOptions.PLOT_BEST_MODEL_ONLY.value:
                # skip trajectories that are too short
                continue

            model_mean_reward = np.mean(rewards)

            if model_mean_reward > best_model_mean_reward:
                best_model_mean_reward = model_mean_reward
                    
            data_to_be_plotted.append(rewards)
                    

for rewards in data_to_be_plotted:
    iterations = []

    for i in range(len(rewards)):
        iterations.append(i+1)

    if PlotingOptions.PLOT_BEST_MODEL_ONLY.value:
        if best_model_mean_reward == np.mean(rewards):    
            plt.plot(iterations, rewards)
            break
    else:
        plt.plot(iterations, rewards)



plt.legend(loc='best')
plt.xlabel("training iteration")
plt.ylabel("mean reward")
plt.savefig(model+'.jpg')

# plt.show()