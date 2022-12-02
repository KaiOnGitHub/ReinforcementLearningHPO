import argparse
import math
import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            max_iter = 10
            return result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop

stopper = CustomStopper()

def run_ppo(config: Dict, checkpoint_dir=None):
    environment = config.get("environment")
    learning_rate = config.get("lr")
    gamma = config.get("gammas")
    clip = config.get("clips")
    optimal_env_params = config.get("optimal_env_params")

    SEED = 51513

    #Set numpy random seed
    np.random.seed(SEED)

    # Create and wrap the environment
    env = gym.make(environment)
    env = Monitor(env)

    model = PPO('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                clip_range=clip, seed=SEED, **optimal_env_params)

    current_iteration = 0   

    # If checkpoint_dir is not None, then we are resuming from a checkpoint.
    if checkpoint_dir:
        print("Loading from checkpoint.")
        path = os.path.join(checkpoint_dir, "checkpoint")
        iteration_file_path = os.path.join(checkpoint_dir, "training_iteration_checkpoint")

        model = PPO.load(path, env)

        with open (iteration_file_path, "rb") as f:
            current_iteration = pickle.load(f)
            



    # Train the agent
    timesteps = 10
    rewards = []
    for i in range(current_iteration+1, timesteps+1):
        total_timesteps = int(1e4)
        model.learn(total_timesteps)
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=env)
        rewards.append(r)

        mean_reward = rewards[-1]

        with tune.checkpoint_dir(i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            iteration_file_path = os.path.join(checkpoint_dir, "training_iteration_checkpoint")

            model.save(path)

            with open (iteration_file_path, "wb") as f:
                
                pickle.dump(i, f)


        tune.report(mean_reward=mean_reward, training_iteration=i)

parser = argparse.ArgumentParser("python run_ppo_pbt.py")
parser.add_argument("environment", help="The gym environment as string", type=str)
args = parser.parse_args()

if args.environment:
    environment = args.environment
else:
    environment = 'Acrobot-v1'

print("Running training for environment: "+ environment)

scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=3,
        hyperparam_mutations={
            "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
            "gammas": tune.uniform(lower=0.8, upper=1.0),
            "clips": tune.uniform(0.05, 0.3),
        },
    )

# use tune for hyperparameter selection
config = {
    "environment": environment,
    "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
    "gammas": tune.uniform(lower=0.8, upper=1.0),
    "clips": tune.uniform(0.05, 0.3),
}

if environment == 'Acrobot-v1' or environment == 'MountainCar-v0':
    env_specific_config = {
        "optimal_env_params": {
            "n_steps": 256,
            "gae_lambda": 0.94, 
            "n_epochs": 4,
            "ent_coef": 0.0,
        }
    }
elif environment == 'CartPole-v1':
    env_specific_config = {
        "optimal_env_params": {
            "n_steps": 32,
            "batch_size": 256,
            "gae_lambda": 0.8, 
            "n_epochs": 20,
            "ent_coef": 0.0,
        }
    }

config.update(env_specific_config)

# set `address=None` to train on laptop
ray.init(address=None)

# use local_mode to run in one process (enables debugging in IDE)
# ray.init(address=None,local_mode=True)

# TODO: For syncing on cluster see: https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
sync_config = tune.SyncConfig()

analysis = tune.run(
    run_ppo,
    name="ppo-pbt-tune_"+environment,
    scheduler=scheduler,
    verbose=False,
    metric="mean_reward",
    mode="max",
    num_samples=10,
    stop=stopper,

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