from re import M, search
from time import time
from unittest import result
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import argparse

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

# plot everything on the same plot
ax = None
fig, ax = plt.subplots()
fig.suptitle(model)

# set limit of x axis depending on the time trained in seconds which is different for the algorithms
if algorithm in ['a2c', 'ppo']:
    if search_method == 'pbt':
        xlimit = 10000
    else: 
        xlimit = 600
elif algorithm == 'dqn':
    ax.set_xticks([1000, 2000, 3000])
    xlimit = 3000

# set y axis depending on the model (negative / positive rewards)
if 'CartPole-v1' in model:
    ax.axis([0, xlimit, 0, 600])
elif 'Acrobot-v1' in model:
    ax.axis([0, xlimit, 0, -600])
elif 'MountainCar-v0' in model:
    ax.axis([0, xlimit, 0, -250])

# walk through all trained model directories and plot the results of the one passed as argument
for subdir, dirs, files in os.walk(model_directory + model):
    for file in files:
        if file in ['monitor.csv', 'progress.csv']:
            mean_rewards = []
            time_trained = []
            time_elapsed = 0
            mean_reward_chunk = []

            file_path = os.path.join(subdir, file)

            # Only in random search do we have fixed parameters throughout the training of the model
            if search_method == 'rs_':
                config = file_path.split('/')[9]

            with open(file_path, newline='') as csvfile:
                resultreader = csv.reader(csvfile, delimiter=',')

                row_num = 1
                for row in resultreader:

                    # skip the heading of the csv
                    if len(row) < 3 or row[0] in ['r', 'mean_reward']:
                        continue

                    reward = float(row[0])
                    episode_length = float(row[1])
                    
                    if (file == 'monitor.csv'):    
                        time_elapsed = float(row[2])
                    elif (file == 'progress.csv'):
                        time_this_iter = float(row[1])
                        time_elapsed += time_this_iter

                    # stop reading csv when xlimit is reached and choose the best model via best mean reward across training
                    if time_elapsed > xlimit:
                        model_mean_reward_evaluation = np.mean(mean_rewards)
                        if model_mean_reward_evaluation > best_model_mean_reward:
                            best_model_mean_reward = model_mean_reward_evaluation
                        break

                    # we smoothen the plot depending on the amount of data
                    if file == 'progress.csv':
                        chunk_size = 2
                    elif file == 'monitor.csv':
                        chunk_size = 50

                    if (row_num % chunk_size == 0):
                        mean_reward = np.mean(mean_reward_chunk)
                        mean_rewards.append(mean_reward)
                        mean_reward_chunk = []
                        time_trained.append(round(float(time_elapsed), 2))
                    else:
                        mean_reward_chunk.append(float(reward))
                
                    row_num += 1

            
            # for rs we could determine the best model by looking at the graph
            best_configs = [
                ["a2c-rs_CartPole-v1", "0.002364653701144912_0.9472172939982206"],
                ["a2c-rs_Acrobot-v1", "2.389750676286248e-05_0.9244750203730072"],
                ["a2c-rs_MountainCar-v0", "0.0011058210654566563_0.8256825040860226"],
                ["dqn-rs_Acrobot-v1", "0.0061080332185138855_0.9992974101720703_0.1502436913690241"],
                ["dqn-rs_CartPole-v1", False],
                ["dqn-rs_MountainCar-v0", False],
                ["ppo-rs_CartPole-v1", "0.0011058210654566563_0.8256825040860226_0.1925974651791333"],
                ["ppo-rs_Acrobot-v1", "0.0061080332185138855_0.9992974101720703_0.1502436913690241"],
                ["ppo-rs_MountainCar-v0", "0.0061080332185138855_0.9992974101720703_0.1502436913690241"],
            ]
            
            # if search_method == 'rs_':
                # if any(config == x and model == arr[0] for arr in best_configs for x in arr):
                    # plt.plot(time_trained, mean_rewards, label=config)
                    
                    # if config is not None and config.startswith('0.0061'):
                    #    print(config)

            data_to_be_plotted.append([time_trained, mean_rewards, config])
                    
for m in data_to_be_plotted:
    # we only want to plot the best model
    # if best_model_mean_reward == np.mean(m[1]):
        plt.plot(m[0], m[1], label=m[2])

plt.legend(loc='best')
plt.xlabel("time in seconds")
plt.ylabel("mean reward")
plt.savefig(model+'.jpg')
# plt.show()
