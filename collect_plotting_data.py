from calendar import c
from cmath import isnan
from enum import Enum
from math import nan
import math
from re import M, search
from time import time
from unittest import result
import csv
import json
import os
import numpy as np
import argparse
import re
import sys

# default model, if no model is passed as argument

parser = argparse.ArgumentParser("python collect_plotting_data.py")
parser.add_argument("model_dir_name", help="The directory name of the model to be plotted", type=str, nargs='?')
parser.add_argument("seed_dir_name", help="The directory name of the seed", type=str, nargs='?')
args = parser.parse_args()

if args.model_dir_name:
    model = args.model_dir_name

# parent directory that contains the training data to be collected
model_directory = os.path.join("/Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/trained_final/", args.seed_dir_name, args.model_dir_name)


# one of the algorithms dqn, ppo or a2c
algorithm = model[:3]
# one of the search methods rs, smac or pbt (short rs_, sma and pbt)
search_method = model[4:7]

# default values
config = None
best_model_mean_reward = -1000
data_to_be_plotted = []
smac_iterations = []

# walk through all trained model directories and plot the results of the one passed as argument
for subdir, dirs, files in os.walk(model_directory):
    for file in files:

        # PBT: Find best model
        if search_method == 'pbt':
            if file == 'result.json':
                file_path = os.path.join(subdir, file)

                with open(file_path) as file:
                    lines = file.readlines()
                    lineCount = len(lines)

                    current_line = 1

                    for line in lines:
                        if current_line == lineCount:
                            row = json.loads(line)
                            final_reward = row["mean_reward"]

                            if final_reward > best_model_mean_reward:
                                best_model_mean_reward = final_reward
                                best_pbt_model_file_path = file_path
                        current_line += 1

        # RS and SMAC:
        pattern = re.compile(r".*eval\.json")

        if isinstance(file, str) and pattern.match(file):
            final_rewards = []

            file_path = os.path.join(subdir, file)
            with open(file_path) as file:
                lines = file.readlines()
                lineCount = len(lines)

                if search_method == 'rs_':
                    for line in lines:
                        row = json.loads(line)

                        final_rewards.append(row["rewards"][-1])

                # SMAC: Collect rewards and iterations in reverse order to skip previously loaded models
                if search_method == 'sma':
                    seen_configs = []
                    for line in reversed(lines):
                        row = json.loads(line)

                        rewards = row["rewards"]
                        if len(rewards) < 1:
                            continue

                        config = str(row["gamma"])+"_"+str(row["learning_rate"])
                        
                        if config in seen_configs:
                            continue
                            
                        seen_configs.append(config)
                        final_rewards.insert(0, rewards[-1])
                        smac_iterations.insert(0, len(rewards))

            data_to_be_plotted.append(final_rewards)

# PBT: Extract final rewards from best model
if 'best_pbt_model_file_path' in locals():
    """
        # read lines reversed and only preserve data for the first time a iteration number appears (ignores iterations that were overwritten later)
        # e.g. X are ignored lines
        
        # 1
        # 2
        # 3
        # 4
        # 5
        # 6 X
        # 7 X
        # 8 X
        # 6 
        # 7 
        # 8
        # 9
        # """

    with open(best_pbt_model_file_path) as file:
        lines = file.readlines()
        final_rewards = []
        added_indeces = [0 for x in range(len(lines)+1)]

        for line in reversed(lines):
            row = json.loads(line)
            training_iteration = row["training_iteration"]
            mean_reward = row["mean_reward"]

            if added_indeces[training_iteration] == 0:
                final_rewards.insert(0, mean_reward)
                added_indeces[training_iteration] = 1

        data_to_be_plotted.append(final_rewards)


for rewards in data_to_be_plotted:
    iterations = []

    previous_reward = -10000

    smac_cur_iteration = 0

    for i, r in enumerate(rewards):
        if search_method in ['sma', 'rs_'] and r < previous_reward:
            # For plotting RS and SMAC we want monotonically increasing rewards
            rewards[i] = previous_reward
            previous_reward = previous_reward
        else:
            previous_reward = r

        if search_method in ['pbt', 'rs_']:
            iterations.append((i+1)*10)
        else:
            smac_cur_iteration += smac_iterations[i]
            iterations.append(smac_cur_iteration)

    rewards.insert(0, 0)
    iterations.insert(0, 0)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    plotting_dir = os.path.join(script_dir, "../plotting_data")

    os.makedirs(plotting_dir, exist_ok=True)

    plotting_data = {}
    iterations_and_rewards = [iterations, rewards]
    plotting_data[search_method] = iterations_and_rewards

    with open(os.path.join(plotting_dir, "%s" % (algorithm)), 'a+') as f:
        json.dump(plotting_data, f)
        f.write("\n")