import logging
import math
import pickle
import argparse

logging.basicConfig(level=logging.INFO)

import json
import os

import gym
import numpy as np
from ConfigSpace.hyperparameters import (UniformFloatHyperparameter)
from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

SEED = 99
MAX_ITERATIONS_PER_LOOP = 10
MAX_TOTAL_ITERATIONS = 100
# we will cut of the # of configurations after MAX_TOTAL_ITERATIONS is reached
NUMBER_OF_CONFIGURATIONS = 30
INITIAL_BUDGET = 2
ETA = 2

algorithms = {
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO
}
environments = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]

parser = argparse.ArgumentParser("python smac_mf.py")
parser.add_argument("algorithm", help="The search algorithm as string (A2C|DQN|PPO)", type=str)
parser.add_argument("environment", help="The gym environment as string", type=str)
args = parser.parse_args()

if args.environment not in environments or args.algorithm not in list(algorithms.keys()):
    print(f"passed {args.algorithm} as algorithm and {args.environment} as environment")
    print(f"allowed algorithms: {list(algorithms.keys())}")
    print(f"allowed environments: {environments}")
    raise ValueError("Did not pass the correct environment or algorithm as an argument.")

algorithmString = args.algorithm
algorithmClass = algorithms[algorithmString]
environment = args.environment

def create_configspace_for_algorithm(algorithm: str) -> ConfigurationSpace:
    cs = ConfigurationSpace()

    params = [
        UniformFloatHyperparameter(
                "learning_rate", math.pow(10, -6), math.pow(10, -2), default_value=0.001, log=True
            ),
        UniformFloatHyperparameter(
                "gamma", 0.8, 1.0, default_value=0.99, log=False
            )
    ]

    if algorithm == 'DQN':
        params.append(UniformFloatHyperparameter(
            "clip", 0.05, 0.3, default_value=0.2, log=False
        ))
    elif algorithm == 'PPO':
        params.append(UniformFloatHyperparameter(
            "clip", 0.05, 0.3, default_value=0.2, log=False
        ))

    # Add all hyperparameters at once:
    cs.add_hyperparameters(params)
    
    return cs

def create_model(policy, env, learning_rate, gamma, clip, optimal_env_params):
    if algorithmString == 'DQN':
        return DQN(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                    exploration_final_eps=clip, exploration_initial_eps=clip,
                    exploration_fraction=1, seed=SEED, **optimal_env_params)
    elif algorithmString == 'A2C':
        return A2C(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma, seed=SEED, **optimal_env_params)
    elif algorithmString == 'PPO':
        return PPO(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                clip_range=clip, seed=SEED, **optimal_env_params)


def get_pretuned_hyperparameters(algorithm: str, environment: str) -> dict:
    if algorithm == 'DQN':
        if (environment == 'CartPole-v1'):
            return dict(
                batch_size=64,
                buffer_size=100000,
                learning_starts=1000,
                target_update_interval=10,
                train_freq=256,
                gradient_steps=128,
                policy_kwargs=dict(net_arch=[256, 256]))
    elif algorithm == 'A2C':
        # nothing to improve for A2C
        return dict()
    elif algorithm == 'PPO':
        if (environment == 'MountainCar-v0'):
            return dict(
                n_steps=16,
                gae_lambda=0.98,
                n_epochs=4,
                ent_coef=0.0,
            )
        elif (environment == 'CartPole-v1'):
            return dict(
                n_steps=32,
                batch_size=256,
                gae_lambda=0.8,
                n_epochs=20,
                ent_coef=0.0,
            )
        elif (environment == 'Acrobot-v1'):
            return dict(
                n_steps=256,
                batch_size=128,
                gae_lambda=0.94,
                n_epochs=4,
                ent_coef=0.0,
            )


# Because we use parameter noise, we should use a MlpPolicy with layer normalization
def run_algorithm(config: dict, environment: str = environment, policy: str = 'MlpPolicy'
            , budget: int = 1):
    
    hyperparams = dict(
        lr = config["learning_rate"],
        gamma = config["gamma"],
        clip = config["clip"])

    # Create log dir if it doesn't exist
    log_dir = "%s-smac_%s/" % (str.lower(algorithmString), environment)
    os.makedirs(log_dir, exist_ok=True)

    hyperparamsAsString = "_".join(f'{k}_{v}' for k, v in hyperparams.items() if v is not None)

    checkpoint_dir = os.path.join(log_dir, "checkpoint_"+hyperparamsAsString)
    evaluation_dir = os.path.join(log_dir, "%s-smac_%s_seed%s_eval.json" % (str.lower(algorithmString), environment, SEED))
    model_evaluation_dir = os.path.join(checkpoint_dir, "%s-smac_%s_model-valuation.json" % (str.lower(algorithmString), hyperparamsAsString))
    model_dir = os.path.join(checkpoint_dir, "checkpoint_"+hyperparamsAsString)
    
    if os.path.exists(checkpoint_dir):
        # Create and wrap the environment
        env = gym.make(environment)
        env = Monitor(env, checkpoint_dir)

        # TODO: Check if this is working as expected
        model = algorithmClass.load(path=model_dir, env=env)

        with open(model_evaluation_dir, 'r') as f:
            data = json.load(f)
            
        rewards = data["rewards"]
        std_rewards = data["std_rewards"]
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Create and wrap the environment
        env = gym.make(environment)
        env = Monitor(env, checkpoint_dir)

        optimal_env_params = get_pretuned_hyperparameters(algorithm=algorithmString, environment=environment)

        model = create_model(policy, env, hyperparams["lr"], hyperparams["gamma"], hyperparams["clip"], optimal_env_params)

        rewards = []
        std_rewards = []

    # Train the agent
    timesteps = budget - len(rewards)
    if len(rewards) < budget:
        for i in range(int(timesteps)):
            print (f"training loop {i} of {timesteps}")

            total_iteration_count = 0
            
            total_iteration_count_path = os.path.join(log_dir, "total_iteration_count.json")
            if os.path.exists(total_iteration_count_path):
                with open (total_iteration_count_path, "rb") as f:
                    total_iteration_count = pickle.load(f)
                    print("total iteration count: ", total_iteration_count)     

            if total_iteration_count >= MAX_TOTAL_ITERATIONS:
                print("reached total iteration limit of "+str(total_iteration_count)+" !")
                break

            model.learn(total_timesteps=int(1e4), callback=None)
            # Returns average and standard deviation of the return from the evaluation
            r, std_r = evaluate_policy(model=model, env=env)
            rewards.append(r)
            std_rewards.append(std_r)

            with open(total_iteration_count_path, 'wb') as f:
                pickle.dump(total_iteration_count+1, f)

    if len(rewards) > 0:
        data = {"gamma": hyperparams["gamma"], "learning_rate": hyperparams["lr"], "epsilon": hyperparams["clip"],
                            "rewards": rewards, "std_rewards": std_rewards}
        with open(model_evaluation_dir, 'w+') as f:
            json.dump(data, f)

        with open(evaluation_dir, 'a+') as f:
            json.dump(data, f)
            f.write("\n")

        # Save state to checkpoint file.
        # No need to save optimizer for SGD.
        model.save(path=model_dir)
        return -1 * np.amax(rewards)


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = create_configspace_for_algorithm(algorithmString)

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "runcount-limit": NUMBER_OF_CONFIGURATIONS, # number of configurations
            "cs": cs,  # configuration space
            "deterministic": True,
            # Uses pynisher to limit memory and runtime
            # Alternatively, you can also disable this.
            # Then you should handle runtime and memory yourself in the TA
            "limit_resources": False,
            # "cutoff": 30,  # runtime limit for target algorithm
            # "memory_limit": 3072,  # adapt this to reasonable value for your hardware
        }
    )

    # Max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    intensifier_kwargs = {"initial_budget": INITIAL_BUDGET, "max_budget": MAX_ITERATIONS_PER_LOOP, "eta": ETA}
    
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(SEED),
        tae_runner=run_algorithm,
        intensifier_kwargs=intensifier_kwargs,
    )

    tae = smac.get_tae_runner()

    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent