from typing import Dict
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import gym
import numpy as np
import json
import math
import ray
import torch
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

def run_dqn(config: Dict, checkpoint_dir=None):
    environment = 'Pong-v0'
    learning_rate = config.get("lr")
    gamma = config.get("gammas")
    eps = config.get("epsilons")
    
    seed = 67890

    # Create and wrap the environment
    env = make_atari_env(environment, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=1)

    model = DQN('CnnPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                exploration_fraction=1, exploration_initial_eps=eps, exploration_final_eps=eps, seed=seed)
    rewards = []
    std_rewards = []

    # If checkpoint_dir is not None, then we are resuming from a checkpoint.
    if checkpoint_dir:
        print("Loading from checkpoint.")
        path = os.path.join(checkpoint_dir, "checkpoint")
        model = DQN.load(path, env)
        # TODO: We should probably load the timestep from the checkpoint when training was stopped and resume there.

    # Train the agent
    timesteps = int(1e6/1e4)
    for i in range(timesteps):
        model.learn(total_timesteps=int(10e3))
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=env)
        rewards.append(r)
        std_rewards.append(std_r)

        mean_reward = np.mean(rewards[-100:])

        with tune.checkpoint_dir(i) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                model.save(path)

        # TODO: Report a mean or best result
        tune.report(mean_reward=mean_reward)


scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations={
            # TODO: Should this be same as the search space?
            "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
            "gammas": tune.uniform(lower=0.8, upper=1.0),
            "epsilons": tune.uniform(0.05, 0.3)
        },
    )

seed = 67890
#Set numpy random seed
np.random.seed(seed)

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

# r, std_r = run_dqn(config=search_space)
analysis = tune.run(
    run_dqn,
    name="dqn-pbt-tune",
    scheduler=scheduler,
    verbose=False,
    metric="mean_reward",
    mode="max",
    num_samples=10,

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


# best_trial = analysis.best_trial  # Get best trial
# best_config = analysis.best_config  # Get best trial's hyperparameters
# best_logdir = analysis.best_logdir  # Get best trial's logdir
# best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
# best_result = analysis.best_result  # Get best trial's last results
# best_result_df = analysis.best_result_df  # Get best result as pandas dataframe