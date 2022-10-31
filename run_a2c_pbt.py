from datetime import datetime
from turtle import end_fill
from typing import Dict
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import gym
import numpy as np
import math
import ray
import time
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import Stopper
import argparse


def run_a2c(config: Dict, checkpoint_dir=None):
    environment = config.get("environment")
    learning_rate = config.get("lr")
    gamma = config.get("gammas")
    optimal_env_params = config.get("optimal_env_params")

    seed = int(datetime.now().timestamp())
    #Set numpy random seed
    np.random.seed(seed)

    # Create and wrap the environment
    env = gym.make(environment)
    env = Monitor(env)

    model = A2C('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma, seed=seed, **optimal_env_params)

    # If checkpoint_dir is not None, then we are resuming from a checkpoint.
    if checkpoint_dir:
        print("Loading from checkpoint.")
        path = os.path.join(checkpoint_dir, "checkpoint")
        model = A2C.load(path, env)


    # Train the agent
    timesteps = 10
    rewards = []
    for i in range(timesteps):
        print(f"iteration {i} of {timesteps}")
        total_timesteps = int(1e4)
        model.learn(total_timesteps)
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=env)
        rewards.append(r)

        mean_reward = rewards[-1]

        with tune.checkpoint_dir(i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            # TODO save model total timesteps and episodes total here ?
            model.save(path)

        tune.report(mean_reward=mean_reward)

parser = argparse.ArgumentParser("python run_a2c_pbt.py")
parser.add_argument("environment", help="The gym environment as string", type=str)
args = parser.parse_args()

if args.environment:
    environment = args.environment
else:
    environment = 'Acrobot-v1'

print("ENV: "+ environment)

scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations={
            "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
            "gammas": tune.uniform(lower=0.8, upper=1.0),
        },
    )

# use tune for hyperparameter selection
config = {
    "environment": environment,
    "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
    "gammas": tune.uniform(lower=0.8, upper=1.0),
}

if environment == 'Acrobot-v1' or environment == 'MountainCar-v0' or environment == 'CartPole-v1':
    env_specific_config = {
        "optimal_env_params": {
            "ent_coef": 0.0,
        }
    }

config.update(env_specific_config)

print(config)

# set `address=None` to train on laptop
ray.init(address=None)

print("a2c-pbt-tune_"+environment)
# use local_mode to run in one process (enables debugging in IDE)
# ray.init(address=None,local_mode=True)

# TODO: For syncing on cluster see: https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
sync_config = tune.SyncConfig()

analysis = tune.run(
    run_a2c,
    name="a2c-pbt-tune_"+environment,
    scheduler=scheduler,
    verbose=False,
    metric="mean_reward",
    mode="max",
    num_samples=2,

    # a directory where results are stored before being
    # sync'd to head node/cloud storage
    local_dir=os.path.dirname(os.path.realpath(__file__))+'/../tmp',

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
    
    config=config,
    )

print("Best hyperparameters found were: ", analysis.best_config)