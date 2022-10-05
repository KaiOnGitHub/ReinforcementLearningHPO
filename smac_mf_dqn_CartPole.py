import logging
import math

logging.basicConfig(level=logging.INFO)

import warnings
import numpy as np

import ConfigSpace as CS
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import json
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

SEED = int(sys.argv[1])

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, file: str = None,
                 gamma: float = 0.99, lr: float = 1e-4, eps: float = 0.1, seed: int = 12345):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.file = file
        self.gamma = gamma
        self.lr = lr
        self.eps = eps
        self.seed = seed

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              # if self.verbose > 0:
              #   print(f"Num timesteps: {self.num_timesteps}")
              #   print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
              #     # Example for saving best model
              #     if self.verbose > 0:
              #       print(f"Saving new best model to {self.save_path}")
              #     self.model.save(self.save_path)

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Retrieve training reward
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
              # Mean training reward over the last 100 episodes
             mean_reward = np.mean(y[-100:])
             data = {"gamma": self.gamma, "learning_rate": self.lr, "clip": self.eps,
                     "rewards": list(y)}
             with open("%s_seed%s.json" % (self.file, self.seed), 'w+') as f:
                 json.dump(data, f)
             for idx, re in enumerate(y):
                 print("Episode %s: %s" % (idx+1, re))
             if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

        return True




def run_dqn(config: dict, environment: str = 'CartPole-v1', policy: str = 'MlpPolicy'
            , budget: int = 1, seed: int = 1):
    learning_rate = config["learning_rate"]
    gamma = config["gamma"]
    clip = config["clip"]
    # Create log dir if it doesn't exist
    log_dir = "SMAC4MF_tmp_DQN_%s_%s_%s_%s_%s/" % (environment, learning_rate, gamma, clip, seed)

    if os.path.exists(log_dir):
        # Create and wrap the environment
        env = gym.make(environment)
        env = Monitor(env, log_dir)
        model = DQN.load(path=os.path.join(log_dir, "checkpoint"), env=env)
        # model.gamma = gamma
        # model.learning_rate = learning_rate
        # model.clip_range = clip
        with open(os.path.join(log_dir, "%s_DQN_random_lr_%s_gamma_%s_eps_%s_seed%s_eval.json" % (environment, learning_rate, gamma, clip, seed)), 'r') as f:
            data = json.load(f)
        rewards = data["rewards"]
        std_rewards = data["std_rewards"]
    else:
        os.makedirs(log_dir, exist_ok=True)
        # Create and wrap the environment
        env = gym.make(environment)
        env = Monitor(env, log_dir)
        # Because we use parameter noise, we should use a MlpPolicy with layer normalization

        if (environment == 'CartPole-v1'):
            optimal_env_params = dict(
                batch_size=64,
                buffer_size=100000,
                learning_starts=1000,
                target_update_interval=10,
                train_freq=256,
                gradient_steps=128,
                policy_kwargs=dict(net_arch=[256, 256]))

        model = DQN(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                    exploration_final_eps=clip, exploration_initial_eps=clip,
                    exploration_fraction=1, seed=seed, **optimal_env_params)
        rewards = []
        std_rewards = []
    # Create the callback: check every 1000 steps
    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, seed=seed)
    # Train the agent
    timesteps = budget - len(rewards)
    if len(rewards) < budget:
        for i in range(int(timesteps)):
            model.learn(total_timesteps=int(1e4), callback=None)
            # Returns average and standard deviation of the return from the evaluation
            r, std_r = evaluate_policy(model=model, env=env)
            rewards.append(r)
            std_rewards.append(std_r)
    data = {"gamma": gamma, "learning_rate": learning_rate, "epsilon": clip,
                         "rewards": rewards, "std_rewards": std_rewards}
    with open(os.path.join(log_dir,  "%s_DQN_random_lr_%s_gamma_%s_eps_%s_seed%s_eval.json" % (environment, learning_rate, gamma, clip, seed)),
              'w+') as f:
        json.dump(data, f)
    path = os.path.join(log_dir, "checkpoint")
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
    clipping = UniformFloatHyperparameter(
        "clip", 0.05, 0.3, default_value=0.2, log=False
    )

    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [
            learning_rate,
            gamma,
            clipping
        ]
    )

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "runcount-limit": 100,  # max duration to run the optimization (in seconds)
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
    max_epochs = 2e6

    # Intensifier parameters
    intensifier_kwargs = {"initial_budget": 1e4, "max_budget": max_epochs, "eta": 5}
    # seed = int(sys.argv[1])
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(SEED),
        tae_runner=run_dqn,
        intensifier_kwargs=intensifier_kwargs,
    )

    tae = smac.get_tae_runner()

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = tae.run(
        config=cs.get_default_configuration(),
        budget=max_epochs, seed=12345
    )[1]

    print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = tae.run(config=incumbent, budget=max_epochs, seed=12345)[1]
    print("Value for inc configuration: %.4f" % inc_value)

