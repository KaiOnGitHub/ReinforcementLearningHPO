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
from ray import tune

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, seed: int = 12345):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
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
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward

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
             for idx, re in enumerate(y):
                 print("Episode %s: %s" % (idx+1, re))
             if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

        return True


def run_dqn(config: Dict, environment: str = 'Pong-v0'):
    learning_rate = config.get("lr").sample()
    gamma = config.get("gammas").sample()
    eps = config.get("epsilons").sample()
    
    seed = 67890
    # Create log dir
    log_dir = "tmp_DQN_%s_%s_%s_%s/" % (environment, learning_rate, gamma, eps)
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = make_atari_env(environment, n_envs=1, seed=seed, monitor_dir=log_dir)
    env = VecFrameStack(env, n_stack=1)

    model = DQN('CnnPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                exploration_fraction=1, exploration_initial_eps=eps, exploration_final_eps=eps, seed=seed)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, seed=seed)
    rewards = []
    std_rewards = []
    # Train the agent
    timesteps = int(2e6/1e4)
    for i in range(timesteps):
        model.learn(total_timesteps=int(1e4), callback=callback)
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=env)
        rewards.append(r)
        std_rewards.append(std_r)
    data = {"gamma": gamma, "learning_rate": learning_rate, "epsilon": eps,
                         "rewards": rewards, "std_rewards": std_rewards}
    with open("%s_DQN_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json" % (environment, learning_rate, gamma, eps, seed),
              'w+') as f:
        json.dump(data, f)
    return rewards, std_rewards

n_configs = 100
seed = 67890
#Set numpy random seed
np.random.seed(seed)

learning_rates = np.power(10, np.random.uniform(low=-6, high=-2, size=n_configs))
gammas = np.random.uniform(low=0.8, high=1, size=n_configs)
epsilons = np.random.uniform(low=0.05, high=0.3, size=n_configs)

# use tune for hyperparameter selection
search_space = {
    "lr": tune.uniform(lower=math.pow(10, -6), upper=math.pow(10, -2)),
    "gammas": tune.uniform(lower=0.8, upper=1.0),
    "epsilons": tune.uniform(0.05, 0.3)
}

r, std_r = run_dqn(config=search_space)
