from datetime import datetime
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import os
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
import argparse


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


def run_ppo(learning_rate: float, gamma: float, clip: float, environment: str):
    seed = 67890
    # Create log dir
    working_dir = os.path.dirname(os.path.realpath(__file__))+'/../tmp/ppo-rs_%s/%s_%s_%s/' % (environment, learning_rate, gamma, clip)
    monitor_dir = working_dir+"monitor"
    eval_dir = working_dir+"evaluation"

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Create and wrap the environment
    env = make_vec_env(environment, n_envs=8)
    env = gym.make(environment)
    env = Monitor(env, monitor_dir)

    if (environment == 'MountainCar-v0'):
        optimal_env_params = dict(
            n_steps=16,
            gae_lambda=0.98,
            n_epochs=4,
            ent_coef=0.0,
        )
    elif (environment == 'CartPole-v1'):
        optimal_env_params = dict(n_steps=32,
        batch_size=256,
        gae_lambda=0.8,
        n_epochs=20,
        ent_coef=0.0,
        )
    elif (environment == 'Acrobot-v1'):
        optimal_env_params = dict(
            n_steps=256,
            gae_lambda=0.94,
            n_epochs=4,
            ent_coef=0.0,
        )

    # Because we use parameter noise, we should use a MlpPolicy with layer normalization
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                clip_range=clip, seed=seed, **optimal_env_params)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=monitor_dir, seed=seed)
    rewards = []
    std_rewards = []
    # Train the agent
    timesteps = 10
    for i in range(timesteps):
        model.learn(total_timesteps=int(1e4), callback=callback)
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=env)
        rewards.append(r)
        std_rewards.append(std_r)
    data = {"gamma": gamma, "learning_rate": learning_rate, "clip": clip,
                         "rewards": rewards, "std_rewards": std_rewards}
    with open("%s/%s_PPO_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json" % (eval_dir, environment, learning_rate, gamma, clip, seed),
              'w+') as f:
        json.dump(data, f)
    return rewards, std_rewards

parser = argparse.ArgumentParser("python run_ppo_rs.py")
parser.add_argument("environment", help="The gym environment as string", type=str)
args = parser.parse_args()

if args.environment:
    environment = args.environment
else:
    environment = 'Acrobot-v1'

print("ENV: "+ environment)

n_configs = 10
#Set numpy random seed
np.random.seed(int(datetime.now().timestamp()))
learning_rates = np.power(10, np.random.uniform(low=-6, high=-2, size=n_configs))
gammas = np.random.uniform(low=0.8, high=1, size=n_configs)
clips = np.random.uniform(low=0.05, high=0.3, size=n_configs)

for lr, gamma, clip in zip(learning_rates, gammas, clips):
    r, std_r = run_ppo(learning_rate=lr, gamma=gamma, clip=clip, environment=environment)
