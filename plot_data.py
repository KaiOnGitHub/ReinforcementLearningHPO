import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
plotting_dir = os.path.join(script_dir, "../plotting_data")

parser = argparse.ArgumentParser("python plot_data.py")
parser.add_argument("algorithm", help="the algorithm (dqn, ppo or a2c)", type=str, nargs='?')
args = parser.parse_args()

if args.algorithm:
    algorithm = args.algorithm
else:
    exit()

os.makedirs(plotting_dir, exist_ok=True)

# plot everything on the same plot
ax = None
fig, ax = plt.subplots()

xlimit = 100

model = ['CartPole-v1']

# set y axis depending on the model (negative / positive rewards)
if 'CartPole-v1' in model:
    ax.axis([1, xlimit, 0, 600])
elif 'Acrobot-v1' in model:
    ax.axis([1, xlimit, 0, -600])
elif 'MountainCar-v0' in model:
    ax.axis([1, xlimit, 0, -250])

fig.suptitle(algorithm.upper())
with open(os.path.join(plotting_dir, "%s" % (algorithm)), 'r') as f:
    lines = f.readlines()
    
    
    data_to_be_plotted = {
        "pbt": [],
        "sma": [],
        "rs_": [],
    }

    # first filter data
    for line in lines:
        data = json.loads(line)
        search_method = ''

        for sm in ['pbt', 'sma', 'rs_']:
            if sm in data:
                iterations_and_rewards = data[sm]
                search_method = sm

        data_to_be_plotted[search_method].append(iterations_and_rewards)

    # calculate mean and std deviation of the filtered data of the 3 different seeds
    for search_method, data in data_to_be_plotted.items():
        
        # check that we have the same amount of data points b4 computing mean
        for data_length in [len(d[1]) for d in data]:
            if data_length != len(data[0][1]):
                raise RuntimeError("It looks like for different seeds we have different lengths of iterations")
        
        reward_arrays = [d[1] for d in data]

        mean_reward_across_seeds = np.mean([reward_arrays[0], reward_arrays[1], reward_arrays[2]], axis=0)
        std_deviation_across_seeds = np.std([reward_arrays[0], reward_arrays[1], reward_arrays[2]], axis=0)
        std_err = std_deviation_across_seeds / np.sqrt(3)

        # TODO What about smac iterations? Are they always the same length as well?
        iterations = data[0][0]

        if search_method == 'pbt':
            label = 'PBT'
        elif search_method == 'sma':
            label = 'SMAC'
        elif search_method == 'rs_':
            label = 'RS'

        plt.plot(iterations, mean_reward_across_seeds, label=label)
        plt.fill_between(iterations, 
            np.add(mean_reward_across_seeds, std_err), 
            np.subtract(mean_reward_across_seeds, std_err),
            alpha=0.25)
        
    
    plt.legend(loc='best')
    plt.xlabel("training iteration")
    plt.ylabel("mean reward")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    plotting_files_dir = os.path.join(script_dir, "../plots_final")

    os.makedirs(plotting_files_dir, exist_ok=True)
    plt.savefig(os.path.join(plotting_files_dir, algorithm+'.jpg'))

    plt.show()

    