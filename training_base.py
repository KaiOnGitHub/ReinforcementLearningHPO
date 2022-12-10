from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import argparse
import numpy as np
import math
import ray
from ray import tune

N_CONFIGS=10
USE_PRETUNED_HYPERPARAMS=True

ALGORITHMS={
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO
}

ENVIRONMENTS=[
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCar-v0"
]


def get_parsed_arguments() -> list:
    """ Returns the arguments passed to the script in the needed format
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", help="The search algorithm as string (A2C|DQN|PPO)", type=str)
    parser.add_argument("environment", help="The gym environment as string", type=str)
    parser.add_argument("seed", help="The random seed", type=int)
    args = parser.parse_args()

    if args.environment not in ENVIRONMENTS or args.algorithm not in list(ALGORITHMS.keys()):
        print(f"passed {args.algorithm} as algorithm and {args.environment} as environment")
        print(f"allowed algorithms: {list(ALGORITHMS.keys())}")
        print(f"allowed environments: {ENVIRONMENTS}")
        raise ValueError("Did not pass the correct environment or algorithm as an argument.")

    algorithmString = args.algorithm
    algorhtmClass = ALGORITHMS[algorithmString]

    return [algorithmString, algorhtmClass, args.environment, args.seed]

LR_LOG_UPPER=10e-2
LR_LOG_LOWER=10e-6
LR_UPPER=-2
LR_LOWER=-6
GAMMA_UPPER=1
GAMMA_LOWER=0.8
CLIP_UPPER=0.3
CLIP_LOWER=0.05
EPSILON_UPPER=0.3
EPSILON_LOWER=0.05


def create_configspace(search_method: str, algorithm: str) -> dict:
    # TODO: Check if config is working as expected
    config = dict()

    if search_method == 'RS':
        config["learning_rates"] = np.power(10, np.random.uniform(low=LR_LOWER, high=LR_UPPER, size=N_CONFIGS))
        config["gammas"] = np.random.uniform(low=GAMMA_LOWER, high=GAMMA_UPPER, size=N_CONFIGS)
        if algorithm == 'PPO':
            config["clips"] = np.random.uniform(low=CLIP_LOWER, high=CLIP_UPPER, size=N_CONFIGS)
        elif algorithm == 'DQN':
            config["epsilons"] = np.random.uniform(low=EPSILON_LOWER, high=EPSILON_UPPER, size=N_CONFIGS)
    elif search_method == 'PBT':
        config["learning_rates"] = tune.loguniform(lower=LR_LOG_LOWER, upper=LR_LOG_UPPER, base=10)
        config["gammas"] = tune.uniform(lower=GAMMA_LOWER, upper=GAMMA_UPPER)
        if algorithm == 'PPO':
            config["clips"] = tune.uniform(lower=CLIP_LOWER, upper=CLIP_UPPER)
        elif algorithm == 'DQN':
            config["epsilons"] = tune.uniform(lower=EPSILON_LOWER, upper=EPSILON_UPPER)
    
    return config

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

def create_model(policy, env, learning_rate, gamma, clip, epsilon, optimal_env_params, algorithmString, seed):
    # TODO: Correct epsilon / seed
    if algorithmString == 'DQN':
        return DQN(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                    exploration_final_eps=epsilon, seed=seed, **optimal_env_params)
    elif algorithmString == 'A2C':
        return A2C(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma, seed=seed, **optimal_env_params)
    elif algorithmString == 'PPO':
        return PPO(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                clip_range=clip, seed=seed, **optimal_env_params)