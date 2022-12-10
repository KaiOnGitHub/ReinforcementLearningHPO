import argparse
import math
import os
import pickle
from typing import Dict
from training_base import get_pretuned_hyperparameters
from training_base import create_model
from training_base import get_parsed_arguments
from training_base import create_configspace
from training_base import N_CONFIGS
from training_base import USE_PRETUNED_HYPERPARAMS

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

N_CONFIGS=N_CONFIGS

algorithmString, algorithmClass, environment, seed = get_parsed_arguments()

#Set numpy random seed
np.random.seed(seed)

print(f"Running PBT {algorithmString} {environment}")

class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            max_iter = 10
            return result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop

def run_algorithm(config: Dict, checkpoint_dir=None):
    environment = config.get("environment")
    learning_rate = config.get("learning_rates")
    gamma = config.get("gammas")
    clip = config.get("clips")
    epsilon = config.get("epsilons")
    optimal_env_params = config.get("optimal_env_params")

    # Create and wrap the environment
    env = gym.make(environment)
    env = Monitor(env)

    model = create_model('MlpPolicy', env, learning_rate, gamma, clip, epsilon, optimal_env_params, algorithmString, seed)
    
    current_iteration = 0   
    
    # If checkpoint_dir is not None, then we are resuming from a checkpoint.
    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        iteration_file_path = os.path.join(checkpoint_dir, "training_iteration_checkpoint")

        print("Loading from checkpoint.")
        model = algorithmClass.load(path, env)
        
        with open (iteration_file_path, "rb") as f:
            current_iteration = pickle.load(f)

    # Train the agent
    timesteps = 10
    rewards = []

    for i in range(current_iteration+1, timesteps+1):
        print(f"iteration {i} of {timesteps}")
        total_timesteps = int(1e4)
        model.learn(total_timesteps)
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=env)
        rewards.append(r)

        mean_reward = rewards[-1]

        with tune.checkpoint_dir(i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            iteration_file_path = os.path.join(checkpoint_dir, "training_iteration_checkpoint")

            # TODO save model total timesteps and episodes total here ?
            model.save(path)
            with open (iteration_file_path, "wb") as f:
                pickle.dump(i, f)

        tune.report(mean_reward=mean_reward, training_iteration=i)

scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=3,
        hyperparam_mutations=create_configspace('PBT', algorithmString)
    )

# the second config is used for fixed hyperparams and the environment variable
config = create_configspace('PBT', algorithmString)
config.update({"environment": environment})

if USE_PRETUNED_HYPERPARAMS:
    config.update({"optimal_env_params": get_pretuned_hyperparameters(algorithmString, environment)})

# set `address=None` to train on laptop
ray.init(address=None)

# use local_mode to run in one process (enables debugging in IDE)
# ray.init(address=None,local_mode=True)

# TODO: For syncing on cluster see: https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
sync_config = tune.SyncConfig()

working_dir = os.path.dirname(os.path.realpath(__file__))+'/../tmp/seed_%s/' % (seed)
os.makedirs(working_dir, exist_ok=True)

analysis = tune.run(
    run_algorithm,
    name="%s-pbt-tune_%s" % (str.lower(algorithmString), environment),
    scheduler=scheduler,
    verbose=False,
    stop=CustomStopper(),
    metric="mean_reward",
    mode="max",
    num_samples=10,

    # a directory where results are stored before being
    # sync'd to head node/cloud storage
    local_dir=working_dir,

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