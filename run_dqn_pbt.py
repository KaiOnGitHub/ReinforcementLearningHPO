from typing import Dict
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import gym
import numpy as np
import math
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

def run_dqn(config: Dict, checkpoint_dir=None):
    environment = 'Acrobot-v1'
    learning_rate = config.get("lr")
    gamma = config.get("gammas")
    eps = config.get("epsilons")

    seed = 67890
    #Set numpy random seed
    np.random.seed(seed)

    # Create and wrap the environment
    env = gym.make(environment)

    model = DQN('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                exploration_fraction=1, exploration_initial_eps=eps, exploration_final_eps=eps)
   
    # TODO: Use Stopper class

    # If checkpoint_dir is not None, then we are resuming from a checkpoint.
    if checkpoint_dir:
        print("Loading from checkpoint.")
        path = os.path.join(checkpoint_dir, "checkpoint")
        model = DQN.load(path, env)


    # Train the agent
    timesteps = int(2e6/1e4)
    rewards = []
    for i in range(timesteps):
        total_timesteps = int(10e5)
        model.learn(total_timesteps)
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=env)
        rewards.append(r)

        mean_reward = rewards[-1]

        with tune.checkpoint_dir(i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            model.save(path)

        # TODO: Does it make sense to report total_timesteps and episode_num? Seems to be always None at the start of the iteration / when loading the checkpoint.
        tune.report(mean_reward=mean_reward, timesteps_total=model._total_timesteps, episodes_total=model._episode_num)


scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations={
            "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
            "gammas": tune.uniform(lower=0.8, upper=1.0),
            "epsilons": tune.uniform(0.05, 0.3)
        },
    )

# use tune for hyperparameter selection
search_space = {
    "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
    "gammas": tune.uniform(lower=0.8, upper=1.0),
    "epsilons": tune.uniform(0.05, 0.3)
}

# set `address=None` to train on laptop
ray.init(address=None)

# use local_mode to run in one process (enables debugging in IDE)
# ray.init(address=None,local_mode=True)

# TODO: For syncing on cluster see: https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
sync_config = tune.SyncConfig()

analysis = tune.run(
    run_dqn,
    name="dqn-pbt-tune",
    scheduler=scheduler,
    verbose=False,
    metric="mean_reward",
    mode="max",
    num_samples=4,

    # a directory where results are stored before being
    # sync'd to head node/cloud storage
    local_dir="tmp/pbtTune",

    # sync our checkpoints via rsync
    # you don't have to pass an empty sync config - but we
    # do it here for clarity and comparison
    sync_config=sync_config,

    # we'll keep the best five checkpoints at all times
    # checkpoints (by AUC score, reported by the trainable, descending)
    checkpoint_score_attr="mean_reward",
    keep_checkpoints_num=5,

    # a very useful trick! this will resume from the last run specified by
    # sync_config (if one exists), otherwise it will start a new tuning run
    resume="AUTO",
    
    config=search_space,
    )

print("Best hyperparameters found were: ", analysis.best_config)

# Plot by wall-clock time
dfs = analysis.fetch_trial_dataframes()

# This plots everything on the same plot
ax = None

df1, df2, df3, df4 = dfs.values()

best_rewards = []
latest_time = []

for i, (d1, d2, d3, d4) in enumerate(zip(df1.values, df2.values, df3.values, df4.values)):
    best_rewards.append(max(d1[0], d2[0], d3[0], d4[0]))
    latest_time.append(max(d1[11], d2[11], d3[11], d4[11]))

# ax = df1.plot("training_iteration", best_rewards, ax=ax, legend=False)

import matplotlib.pyplot as plt

plt.plot(latest_time, best_rewards)

plt.xlabel("time in seconds")
plt.ylabel("mean_reward")
plt.show()


# best_trial = analysis.best_trial  # Get best trial
# best_config = analysis.best_config  # Get best trial's hyperparameters
# best_logdir = analysis.best_logdir  # Get best trial's logdir
# best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
# best_result = analysis.best_result  # Get best trial's last results
# best_result_df = analysis.best_result_df  # Get best result as pandas dataframe