import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt
from env import DroneEnv
import pandas as pd


class custom_log_callback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        # Separate logs to avoid length mismatch errors in pandas
        self.step_logs = {
            "timesteps": [],
            "rewards": [],
            "total_variance": [],
            "value_func": []
        }
        self.episode_logs = {
            "ep_timesteps": [], # To know WHEN the episode ended
            "ep_length": [],
            "ep_return": []
        }
        self.log_dir = "TRAINING_LOGS"
        os.makedirs(self.log_dir, exist_ok=True)
        # Paths
        self.csv_path = os.path.join(self.log_dir, "train_steps.csv")
        self.ep_csv_path = os.path.join(self.log_dir, "train_episodes.csv")
        # Episode trackers
        self.ep_len = 0
        self.ep_ret = 0
        self.gamma = 0.99 # Standard discount factor

    def _on_step(self) -> bool:
        # 1. Access SB3 internal values
        # 'values' is a tensor, 'rewards' and 'dones' are numpy arrays
        val = self.locals["values"][0].item()
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]
        current_step = self.num_timesteps
        # 2. Log step-level data
        self.step_logs["timesteps"].append(current_step)
        self.step_logs["rewards"].append(reward)
        self.step_logs["value_func"].append(val)
        self.step_logs["total_variance"].append(info.get("total_var", 0))
        # 3. Calculate Discounted Episode Return: $G_t = \sum \gamma^t R_t$
        self.ep_len += 1
        self.ep_ret += reward * (self.gamma ** (self.ep_len - 1))
        # 4. Handle Episode End
        if done:
            self.episode_logs["ep_timesteps"].append(current_step)
            self.episode_logs["ep_length"].append(self.ep_len)
            self.episode_logs["ep_return"].append(self.ep_ret)
            # Reset trackers
            self.ep_len = 0
            self.ep_ret = 0
        return True

    def _on_rollout_end(self):
        # Save both dataframes to CSV
        pd.DataFrame(self.step_logs).to_csv(self.csv_path, index=False)
        pd.DataFrame(self.episode_logs).to_csv(self.ep_csv_path, index=False)

    def _on_training_end(self):
        self._on_rollout_end() # Final save
        print("Training complete. Generating plots...")
        # Plotting Helper to avoid code repetition and cleanup figures
        def save_plot(x, y, xlabel, ylabel, title, filename):
            plt.figure(figsize=(10, 5))
            plt.scatter(x, y, s=1)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.log_dir, filename))
            plt.close()
        # Generate all 5 plots
        save_plot(self.step_logs["timesteps"], self.step_logs["rewards"], 'Timesteps', 'Rewards', 'Step Reward vs Time', 'rewards_vs_time.png')    
        save_plot(self.step_logs["timesteps"], self.step_logs["total_variance"], 'Timesteps', 'Total Variance', 'Variance vs Time', 'total_var_vs_time.png')
        save_plot(self.step_logs["timesteps"], self.step_logs["value_func"], 'Timesteps', 'Value Estimate', 'Value Function vs Time', 'val_func_vs_time.png')
        save_plot(self.episode_logs["ep_timesteps"], self.episode_logs["ep_length"], 'Timesteps', 'Length', 'Episode Length vs Time', 'ep_len_vs_time.png')
        save_plot(self.episode_logs["ep_timesteps"], self.episode_logs["ep_return"], 'Timesteps', 'Discounted Return', 'Episode Return vs Time', 'ep_ret_vs_time.png')

env = DroneEnv() 
policy_kwargs = dict(activation_fn=torch.nn.SiLU, net_arch=[128, 64, 32])
model = PPO("MlpPolicy", env, n_steps=1024, n_epochs=10, batch_size=64, policy_kwargs=policy_kwargs, verbose=1)
my_callback = custom_log_callback()
model.learn(total_timesteps=10000000, callback=my_callback)
model.save("TRAINING_LOGS/ppo_policy.pth")