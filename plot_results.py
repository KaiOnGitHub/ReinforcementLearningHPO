from time import time
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

model_directory = '/Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/trained/'
model = 'dqn-rs_Acrobot-v1'
algorithm = model[:3]
config = None

# This plots everything on the same plot
ax = None
fig, ax = plt.subplots()

fig.suptitle(model)

if algorithm == 'a2c' or 'ppo':
    ax.set_xticks([100, 200, 300, 400, 500, 600])
    xlimit = 600
elif algorithm == 'dqn':
    ax.set_xticks([1000, 2000, 3000])
    xlimit = 3000

if 'CartPole-v1' in model:
    ax.axis([0, xlimit, 0, 500])
elif 'Acrobot-v1' in model:
    ax.axis([0, xlimit, 0, -500])
elif 'MountainCar-v0' in model:
    ax.axis([0, xlimit, 0, -250])

for subdir, dirs, files in os.walk(model_directory + model):
    for file in files:
        if (file == 'monitor.csv'):
            mean_rewards = []
            time_trained = []
            mean_reward_chunk = []

            file_path = os.path.join(subdir, file)
            if 'smac' not in model:
                config = file_path.split('/')[9]

            with open(file_path, newline='') as csvfile:
                resultreader = csv.reader(csvfile, delimiter=',')

                row_num = 1
                for row in resultreader:

                    if (len(row) < 3 or row[0] == 'r'):
                        continue

                    reward = float(row[0])
                    episode_length = float(row[1])
                    time_elapsed = float(row[2])

                    if (time_elapsed > 5000):
                        break

                    if (row_num % 100 == 0):
                        mean_reward = np.mean(mean_reward_chunk)

                        mean_rewards.append(mean_reward)
                        time_trained.append(round(float(time_elapsed), 2))
                    else:
                        mean_reward_chunk.append(float(reward))

                    row_num += 1
            
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
            
            # if any(config == x and model == arr[0] for arr in best_configs for x in arr):
            
            # if config is not None and config.startswith('0.0061'):
            #    print(config)
                 
            if True:
                plt.plot(time_trained, mean_rewards, label=config)

plt.legend(loc='best')
plt.xlabel("time in seconds")
plt.ylabel("mean reward")
plt.savefig(model+'.jpg')
plt.show()


# for i, (d1, d2, d3, d4) in enumerate(zip(df1.values, df2.values, df3.values, df4.values)):
#    best_rewards.append(max(d1[0], d2[0], d3[0], d4[0]))
#    time_trained.append(sum([d1[11], d2[11], d3[11], d4[11]]))

# ax = df1.plot("training_iteration", best_rewards, ax=ax, legend=False)

# plt.plot(time_trained, best_rewards)

# plt.xlabel("time in seconds")
# plt.ylabel("mean_reward")
# plt.show()


# best_trial = analysis.best_trial  # Get best trial
# best_config = analysis.best_config  # Get best trial's hyperparameters
# best_logdir = analysis.best_logdir  # Get best trial's logdir
# best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
# best_result = analysis.best_result  # Get best trial's last results
# best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
