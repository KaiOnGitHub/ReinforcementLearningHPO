import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np

mean = [np.mean([x,y]) for x,y in zip(data1,data2)]
std = [np.std([x,y]) for x,y in zip(data1,data2)]

print(mean)
print(std)


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
        
    for line in lines:
        data = json.loads(line)
        search_method = ''

        for sm in ['pbt', 'sma', 'rs_']:
            if sm in data:
                iterations_and_rewards = data[sm]
                search_method = sm

        if search_method == 'pbt':
            label = 'PBT'
        elif search_method == 'sma':
            label = 'SMAC'
        elif search_method == 'rs_':
            label = 'RS'

        plt.plot(iterations_and_rewards[0], iterations_and_rewards[1], label=label)
      
        plt.legend(loc='best')
        plt.xlabel("training iteration")
        plt.ylabel("mean reward")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    plotting_files_dir = os.path.join(script_dir, "../plots_final")

    os.makedirs(plotting_files_dir, exist_ok=True)
    plt.savefig(os.path.join(plotting_files_dir, algorithm+'.jpg'))

    # plt.show()

    