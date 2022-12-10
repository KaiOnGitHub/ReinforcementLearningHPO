import argparse
import json
import os
from training_base import get_pretuned_hyperparameters
from training_base import create_model
from training_base import create_configspace

import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

N_CONFIGS=10

algorithms = {
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO
}
environments = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]

parser = argparse.ArgumentParser("python run_rs.py")
parser.add_argument("algorithm", help="The search algorithm as string (A2C|DQN|PPO)", type=str)
parser.add_argument("environment", help="The gym environment as string", type=str)
parser.add_argument("seed", help="The random seed", type=int)
args = parser.parse_args()

if args.environment not in environments or args.algorithm not in list(algorithms.keys()):
    print(f"passed {args.algorithm} as algorithm and {args.environment} as environment")
    print(f"allowed algorithms: {list(algorithms.keys())}")
    print(f"allowed environments: {environments}")
    raise ValueError("Did not pass the correct environment or algorithm as an argument.")

algorithmString = args.algorithm
algorithmClass = algorithms[algorithmString]
environment = args.environment
seed = args.seed

print(f"Running Random Search {algorithmString} {environment}")

#Set numpy random seed
np.random.seed(seed)

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

def run_algorithm(environment: str, learning_rate, gamma, clip, epsilon):
    # Create directories
    working_dir = os.path.dirname(os.path.realpath(__file__))+'/../tmp/seed_%s/%s-rs_%s/' % (seed, str.lower(algorithmString), environment)

    hyperparamsAsString = "_".join(str(v) for v in [learning_rate, gamma, clip, epsilon] if v is not None)

    monitor_dir = os.path.join(working_dir, "monitor_"+hyperparamsAsString)
    eval_dir = os.path.join(working_dir,"evaluation")

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(environment)
    env = Monitor(env, monitor_dir)

    # For Acrobot, Mountaincar and CartPole the same
    optimal_env_params = get_pretuned_hyperparameters(algorithmString, environment)   

    # Because we use parameter noise, we should use a MlpPolicy with layer normalization
    model = create_model('MlpPolicy', env, learning_rate, gamma, clip, epsilon, optimal_env_params, algorithmString, seed)

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
    data = {"gamma": gamma, "learning_rate": learning_rate, "clip": clip, "epsilon": epsilon, "rewards": rewards, "std_rewards": std_rewards}
    with open("%s/a2c-rs_%s_seed%s_eval.json" % (eval_dir, environment, seed),
              'a+') as f:
        json.dump(data, f)
        f.write("\n")
    return rewards, std_rewards

config = create_configspace('RS', algorithmString)

for i in range(0, N_CONFIGS):
    learning_rate = config.get("learning_rates")[i] if "learning_rates" in config.keys() else None
    gamma = config.get("gammas")[i] if "gammas" in config.keys() else None
    clip = config.get("clips")[i] if "clips" in config.keys() else None
    epsilon = config.get("epsilons")[i] if "epsilons" in config.keys() else None

    r, std_r = run_algorithm(environment, learning_rate, gamma, clip, epsilon)