import logging
import math

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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

SEED = 42

# Because we use parameter noise, we should use a MlpPolicy with layer normalization
def run_a2c(config: dict, environment: str = 'CartPole-v1', policy: str = 'MlpPolicy'
            , budget: int = 1):
    
    learning_rate = config["learning_rate"]
    gamma = config["gamma"]
    # Create log dir if it doesn't exist
    log_dir = "a2c-smac_%s/" % (environment)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoint_lr_%s_gamma_%s" % (learning_rate, gamma))


    if os.path.exists(checkpoint_dir):
        # Create and wrap the environment
        env = gym.make(environment)
        env = Monitor(env, checkpoint_dir)
        path = os.path.join(checkpoint_dir, "checkpoint_lr_%s_gamma_%s" % (learning_rate, gamma))

        model = A2C.load(path=path, env=env)

        with open(os.path.join(checkpoint_dir, "a2c-smac_%s_lr_%s_gamma_%s_seed%s_model-valuation.json" % (environment, learning_rate, gamma, SEED)), 'r') as f:
            data = json.load(f)
            
        rewards = data["rewards"]
        std_rewards = data["std_rewards"]
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Create and wrap the environment
        env = gym.make(environment)
        env = Monitor(env, checkpoint_dir)
        optimal_env_params = dict()


        model = A2C(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma, seed=SEED, **optimal_env_params)
        rewards = []
        std_rewards = []
    # Create the callback: check every 1000 steps
    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, seed=seed)
    # Train the agent
    timesteps = budget - len(rewards)
    if len(rewards) < budget:
        for i in range(int(timesteps)):
            print (f"training loop {i} of {timesteps}")
            model.learn(total_timesteps=int(1e4), callback=None)
            # Returns average and standard deviation of the return from the evaluation
            r, std_r = evaluate_policy(model=model, env=env)
            rewards.append(r)
            std_rewards.append(std_r)
    data = {"gamma": gamma, "learning_rate": learning_rate,
                         "rewards": rewards, "std_rewards": std_rewards}
    with open(os.path.join(checkpoint_dir, "a2c-smac_%s_lr_%s_gamma_%s_seed%s_model-valuation.json" % (environment, learning_rate, gamma, SEED)), 'w+') as f:
        json.dump(data, f)
    
    with open(os.path.join(log_dir, "a2c-smac_%s_seed%s_eval.json" % (environment, SEED)), 'a+') as f:
        json.dump(data, f)
        f.write("\n")

    path = os.path.join(checkpoint_dir, "checkpoint_lr_%s_gamma_%s" % (learning_rate, gamma))

    # Save state to checkpoint file.
    # No need to save optimizer for SGD.
    model.save(path=path)
    return -1 * np.amax(rewards)


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter(
        "learning_rate", math.pow(10, -6), math.pow(10, -2), default_value=0.001, log=True
    )
    gamma = UniformFloatHyperparameter(
        "gamma", 0.8, 1.0, default_value=0.99, log=False
    )

    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [
            learning_rate,
            gamma,
        ]
    )

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "runcount-limit": 30, # number of configurations
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
    max_iterations = 10

    # Intensifier parameters
    intensifier_kwargs = {"initial_budget": 2, "max_budget": max_iterations, "eta": 2}
    # seed = int(sys.argv[1])
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(SEED),
        tae_runner=run_a2c,
        intensifier_kwargs=intensifier_kwargs,
    )

    tae = smac.get_tae_runner()

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    # def_value = tae.run(
        # config=cs.get_default_configuration(),
        # budget=max_iterations, seed=42
    #    )[1]

    #print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    # inc_value = tae.run(config=incumbent, budget=max_iterations, seed=SEED)[1]
    # print("Value for inc configuration: %.4f" % inc_value)

