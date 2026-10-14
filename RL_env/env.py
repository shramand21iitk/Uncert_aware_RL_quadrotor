import numpy as np
import torch
from torch.func import stack_module_state, functional_call, vmap
import copy
from PNN import PNN
import gymnasium as gym
from gymnasium import spaces
import os

device = torch.device("cpu")
print(f"Using device: {device}")
model_folder = "env_models"
num_models = 6
models = []
for i in range(num_models):
    model = PNN().to(device)
    weight_path = os.path.join(model_folder, f"pnn_{i}.pth") 
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval() 
        print(f"Loaded weights for model {i} from {weight_path}")
    else:
        print(f"Warning: {weight_path} not found! Using random weights for model {i}.")
    models.append(model)
params, buffers = stack_module_state(models)
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')
def pnn_model(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.observation_space = spaces.Box(low=-1.2, high=1.2, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.2, high=1.2, shape=(4,), dtype=np.float32)
        #Initialise model and parameters
        self.reset_func()
        self.vmap_pnn = vmap(pnn_model, in_dims=(0, 0, None))
        self.dt = 1/400
        self.var_thresh = 5
    def reset_func(self):
        #Initialise starting state and position
        self.pos = -np.random.rand(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.orient = np.array([0,0,0,1])
        self.angvel = np.zeros(3)
        self.angacc = np.zeros(3)
        self.motors = (1125/3000)*np.ones(4)
        self.state = np.concatenate([
            self.angacc,
            self.angvel,
            self.orient,
            self.acc,
            self.vel,
            self.pos])
        self.state = torch.as_tensor(self.state, dtype=torch.float32, device=device)
        #Set the target position
        self.tar_pos = np.random.rand(3)
        self.tar_pos = torch.as_tensor(self.tar_pos, dtype=torch.float32, device=device)
    def get_info(self, action):
        m1, m2, m3, m4 = [a.item() if isinstance(a, torch.Tensor) else float(a) for a in action]
        info = {
            "motor1": m1, "motor2": m2, "motor3": m3, "motor4": m4,
            "angacc_x": self.state[0].item(), "angacc_y": self.state[1].item(), "angacc_z": self.state[2].item(),
            "angvel_x": self.state[3].item(), "angvel_y": self.state[4].item(), "angvel_z": self.state[5].item(),
            "quat1": self.state[6].item(), "quat2": self.state[7].item(), "quat3": self.state[8].item(), "quat4": self.state[9].item(),
            "acc_x": self.state[10].item(), "acc_y": self.state[11].item(), "acc_z": self.state[12].item(),
            "vel_x": self.state[13].item(), "vel_y": self.state[14].item(), "vel_z": self.state[15].item(),
            "pos_x": self.state[16].item(), "pos_y": self.state[17].item(), "pos_z": self.state[18].item()}
        return info
    def get_obs(self):
        obs_tensor = torch.cat((self.state, self.tar_pos), dim=-1)
        return obs_tensor.detach().cpu().numpy().astype(np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_func()
        info = self.get_info(self.motors)
        obs = self.get_obs()
        return obs, info
    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        input = torch.cat((self.state, action), dim=-1)
        with torch.no_grad():
            means, vars = self.vmap_pnn(params, buffers, input)
        ens_mean = torch.mean(means, dim=0)
        epistemic_var = torch.var(means, dim=0)
        aleatoric_var = torch.mean(vars, dim=0)
        total_var = (epistemic_var + aleatoric_var).mean().item()
        self.state += ens_mean * self.dt
        obs = self.get_obs()
        info = self.get_info(action)
        dist = torch.norm(self.state[16:19] - self.tar_pos).item()
        info["total_var"] = total_var
        info["distance_to_target"] = dist
        if (dist < 1e-2):
            reward = 1000
        else:
            reward = -dist-total_var
        terminated = bool(dist < 1e-2)
        truncated = bool(total_var > self.var_thresh)
        return obs, reward, terminated, truncated, info