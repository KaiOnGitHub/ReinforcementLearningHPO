from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os
import gym
import numpy as np
import json
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import argparse

SEED = 42

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

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


def run_ppo(learning_rate: float, gamma: float, eps: float, environment: str):
    # Create directories
    working_dir = os.path.dirname(os.path.realpath(__file__))+'/../tmp/dqn-rs_%s/' % (environment)
    monitor_dir = working_dir+"monitor_"+str(learning_rate)+"_"+str(gamma)
    eval_dir = working_dir+"evaluation"

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(environment)
    env = Monitor(env, monitor_dir)

    if (environment == 'MountainCar-v0'):
        optimal_env_params = dict(
            batch_size=128,
            buffer_size=10000,
            learning_starts=1000,
            target_update_interval=600,
            train_freq=16,
            gradient_steps=8,
            policy_kwargs=dict(net_arch=[256, 256])
        )
    elif (environment == 'CartPole-v1'):
        optimal_env_params = dict(
            batch_size=64,
            buffer_size=100000,
            learning_starts=1000,
            target_update_interval=10,
            train_freq=256,
            gradient_steps=128,
            policy_kwargs=dict(net_arch=[256, 256]))
    elif (environment == 'Acrobot-v1'):
        optimal_env_params = dict(
            batch_size=128,
            buffer_size=50000,
            learning_starts=0,
            target_update_interval=250,
            train_freq=4,
            gradient_steps=-1,
            policy_kwargs=dict(net_arch=[256, 256])
        )

    # Because we use parameter noise, we should use a MlpPolicy with layer normalization
    model = DQN('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                exploration_fraction=1, exploration_initial_eps=eps, exploration_final_eps=eps, seed=SEED, **optimal_env_params)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=monitor_dir)
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
    data = {"gamma": gamma, "learning_rate": learning_rate, "epsilon": eps,
                         "rewards": rewards, "std_rewards": std_rewards}
    with open("%s/dqn-rs_%s_seed%s_eval.json" % (eval_dir, environment, SEED),
              'a+') as f:
        json.dump(data, f)
        f.write("\n")
    return rewards, std_rewards

parser = argparse.ArgumentParser("python run_dqn_rs.py")
parser.add_argument("environment", help="The gym environment as string", type=str)
args = parser.parse_args()

if args.environment:
    environment = args.environment
else:
    environment = 'Acrobot-v1'

print("ENV: "+ environment)

n_configs = 10
#Set numpy random seed
np.random.seed(SEED)
learning_rates = np.power(10, np.random.uniform(low=-6, high=-2, size=n_configs))
gammas = np.random.uniform(low=0.8, high=1, size=n_configs)
epsilons = np.random.uniform(low=0.05, high=0.3, size=n_configs)

for lr, gamma, eps in zip(learning_rates, gammas, epsilons):
    r, std_r = run_ppo(learning_rate=lr, gamma=gamma, eps=eps, environment=environment)