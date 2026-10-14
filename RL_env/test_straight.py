import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import DroneEnv
import torch

print("Initializing Environment...")
env = DroneEnv()

# Update this path to exactly what your PPO weights file is named
ppo_pth_path = "TRAINING_LOGS/ppo_policy.pth" 

if not os.path.exists(ppo_pth_path):
    raise FileNotFoundError(f"Could not find the PPO weights at {ppo_pth_path}")

print("Initializing Empty Model Architecture...")
print("Loading Policy...")
# Let SB3 handle the extraction natively, even with the .pth extension
model = PPO.load(ppo_pth_path, env=env)

print("Model loaded successfully!")

# ... (Continue to Step 2: Run the Test Flight) ...

# 2. Run the Test Flight and Collect Telemetry
obs, info = env.reset()
done = False

# Telemetry arrays
xs, ys, zs = [], [], []
variances = []
distances = []

# Get the fixed target position for this episode
# Assuming env.tar_pos is a tensor, we convert it to numpy
target_pos = env.tar_pos.cpu().numpy() 

print("Simulating Flight...")
while not done:
    # Use deterministic=True for actual flight testing (no random exploration noise)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Store telemetry from the info dictionary
    xs.append(info["pos_x"])
    ys.append(info["pos_y"])
    zs.append(info["pos_z"])
    variances.append(info["total_var"])
    distances.append(info["distance_to_target"])
    
    done = terminated or truncated

flight_length = len(xs)
print(f"Flight ended after {flight_length} steps. Terminated: {terminated}, Truncated: {truncated}")

# 3. Create the Visualization
fig = plt.figure(figsize=(15, 6))

# ---- Panel 1: 3D Trajectory Color-Mapped by Variance ----
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# Scatter plot the trajectory, coloring by variance
# 'viridis' is great: purple = low variance (stable), yellow = high variance (unstable)
sc = ax1.scatter(xs, ys, zs, c=variances, cmap='viridis', s=15, alpha=0.8)

# Mark Start and Target positions
ax1.scatter(xs[0], ys[0], zs[0], color='black', s=100, marker='X', label='Start')
ax1.scatter(target_pos[0], target_pos[1], target_pos[2], color='red', s=100, marker='*', label='Target')

# Formatting the 3D plot
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')
ax1.set_title('3D Flight Trajectory (Color = Total Variance)')
ax1.legend()
fig.colorbar(sc, ax=ax1, label='Total Variance (Aleatoric + Epistemic)')

# ---- Panel 2: Telemetry Timeline ----
ax2 = fig.add_subplot(1, 2, 2)
timesteps = np.arange(flight_length)

# Plot Variance and Distance on the same timeline using twin axes
color1 = 'tab:red'
ax2.set_xlabel('Timesteps')
ax2.set_ylabel('Total Variance', color=color1)
ax2.plot(timesteps, variances, color=color1, linewidth=2, label='Variance')
ax2.tick_params(axis='y', labelcolor=color1)
ax2.axhline(y=env.var_thresh, color='r', linestyle='--', alpha=0.5, label='Var Threshold')

# Create a secondary y-axis for Distance
ax3 = ax2.twinx()
color2 = 'tab:blue'
ax3.set_ylabel('Distance to Target', color=color2)
ax3.plot(timesteps, distances, color=color2, linewidth=2, label='Distance')
ax3.tick_params(axis='y', labelcolor=color2)

ax2.set_title('Flight Telemetry over Time')
fig.tight_layout()

# Save and show
os.makedirs("TEST_LOGS", exist_ok=True)
save_path = "TEST_LOGS/flight_trajectory.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved successfully to {save_path}")
plt.show()