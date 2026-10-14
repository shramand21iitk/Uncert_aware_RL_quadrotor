import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import stack_module_state, functional_call, vmap
import copy
from PNN import PNN
import os
import gc
import pandas as pd
import seaborn as sns
'''
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
vmap_pnn = vmap(pnn_model, in_dims=(0, 0, None))

df_train = pd.read_csv('data/train_data.csv')
df_test = pd.read_csv('data/test_data.csv')
input_cols = ['ang acc x', 'ang acc y', 'ang acc z',
               'ang vel x', 'ang vel y', 'ang vel z',
               'quat x', 'quat y', 'quat z', 'quat w',
               'acc x', 'acc y', 'acc z',
               'vel x', 'vel y', 'vel z',
               'pos x', 'pos y', 'pos z',
               'mot 1', 'mot 2', 'mot 3', 'mot 4']
output_cols = ['d ang acc x', 'd ang acc y', 'd ang acc z',
               'd ang vel x', 'd ang vel y', 'd ang vel z',
               'd quat x', 'd quat y', 'd quat z', 'd quat w',
               'd acc x', 'd acc y', 'd acc z',
               'd vel x', 'd vel y', 'd vel z',
               'd pos x', 'd pos y', 'd pos z']

x_train = df_train[input_cols]
y_train = df_train[output_cols]
x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)

x_test = df_test[input_cols]
y_test = df_test[output_cols]
x_test = torch.tensor(x_test.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

batch_size = 8192
csv_filename = 'metrics.csv'

# --- FIX 4: Initialize the CSV with proper headers ---
pd.DataFrame(columns=['Error', 'Aleatoric', 'Epistemic', 'Dataset']).to_csv(csv_filename, index=False)
print("Initialized metrics.csv")

# ==========================================
# TRAINING DATA LOOP
# ==========================================
num_samples_train = len(x_train)
total_batches_train = (num_samples_train + batch_size - 1) // batch_size
print(f"Processing Train Data in {total_batches_train} batches...")

for i in range(0, num_samples_train, batch_size):
    x_batch = x_train[i : i + batch_size]
    y_batch = y_train[i : i + batch_size]
    with torch.no_grad():
        means_batch, vars_batch = vmap_pnn(params, buffers, x_batch)
        ens_mean_batch = torch.mean(means_batch, dim=0)
        epistemic_var_batch = torch.var(means_batch, dim=0, unbiased=True)
        aleatoric_var_batch = torch.mean(vars_batch, dim=0)
        
        batch_error = torch.mean(torch.abs(ens_mean_batch - y_batch), dim=1).cpu().numpy()
        batch_epistemic = torch.mean(epistemic_var_batch, dim=1).cpu().numpy()
        batch_aleatoric = torch.mean(aleatoric_var_batch, dim=1).cpu().numpy()
        
        # --- FIX 1 & 2: Standardized column names, removed redundant .append() ---
        df_batch = pd.DataFrame({
            'Error': batch_error,
            'Aleatoric': batch_aleatoric,
            'Epistemic': batch_epistemic,
            'Dataset': 'Train'
        })
        
        df_batch.to_csv(csv_filename, mode='a', header=False, index=False)
        
        del x_batch, y_batch, means_batch, vars_batch
        del ens_mean_batch, epistemic_var_batch, aleatoric_var_batch
        del batch_error, batch_epistemic, batch_aleatoric, df_batch
        gc.collect()

# ==========================================
# TESTING DATA LOOP
# ==========================================
# --- FIX 3: Corrected len(x_train) to len(x_test) ---
num_samples_test = len(x_test)
total_batches_test = (num_samples_test + batch_size - 1) // batch_size
print(f"Processing Test Data in {total_batches_test} batches...")

for i in range(0, num_samples_test, batch_size):
    x_batch = x_test[i : i + batch_size]
    y_batch = y_test[i : i + batch_size]
    with torch.no_grad():
        means_batch, vars_batch = vmap_pnn(params, buffers, x_batch)
        ens_mean_batch = torch.mean(means_batch, dim=0)
        epistemic_var_batch = torch.var(means_batch, dim=0, unbiased=True)
        aleatoric_var_batch = torch.mean(vars_batch, dim=0)
        
        batch_error = torch.mean(torch.abs(ens_mean_batch - y_batch), dim=1).cpu().numpy()
        batch_epistemic = torch.mean(epistemic_var_batch, dim=1).cpu().numpy()
        batch_aleatoric = torch.mean(aleatoric_var_batch, dim=1).cpu().numpy()
        
        # --- FIX 1 & 2: Standardized column names, removed redundant .append() ---
        df_batch = pd.DataFrame({
            'Error': batch_error,
            'Aleatoric': batch_aleatoric,
            'Epistemic': batch_epistemic,
            'Dataset': 'Test'
        })
        
        df_batch.to_csv(csv_filename, mode='a', header=False, index=False)
        
        del x_batch, y_batch, means_batch, vars_batch
        del ens_mean_batch, epistemic_var_batch, aleatoric_var_batch
        del batch_error, batch_epistemic, batch_aleatoric, df_batch
        gc.collect()

# ==========================================
# PLOTTING
# ==========================================
csv_filename = 'metrics.csv'
print("Loading data for plotting...")
df_plot = pd.read_csv(csv_filename)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)     
plt.rc('legend', title_fontsize=16) 
plt.rc('axes', titlesize=18)      
plt.rc('figure', titlesize=20)
fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
# Plot 1: Epistemic Distribution (Train vs Test)
# --- FIX 5: Standardized column references for all plots ---
sns.kdeplot(data=df_plot, x='Epistemic', hue='Dataset', fill=True, common_norm=False, alpha=0.4, palette=['blue', 'red'], ax=axs[0, 0])
axs[0, 0].set_title("(a) Epistemic Uncertainty Distribution")
# Plot 2: Aleatoric Distribution (Train vs Test)
sns.kdeplot(data=df_plot, x='Aleatoric', hue='Dataset', fill=True, common_norm=False, alpha=0.4, palette=['blue', 'red'], ax=axs[0, 1])
axs[0, 1].set_title("(b) Aleatoric Uncertainty Distribution")
# Filter just the test data for the scatter plots
df_test_only = df_plot[df_plot['Dataset'] == 'Test']
# Plot 3: Error vs Epistemic (Test)
sns.scatterplot(data=df_test_only, x='Epistemic', y='Error', alpha=0.3, color='purple', s=10, ax=axs[1, 0])
axs[1, 0].set_title("(c) Test Data: Error vs. Epistemic Uncertainty")
# Plot 4: Error vs Aleatoric (Test)
sns.scatterplot(data=df_test_only, x='Aleatoric', y='Error', alpha=0.3, color='orange', s=10, ax=axs[1, 1])
axs[1, 1].set_title("(d) Test Data: Error vs. Aleatoric Uncertainty")
plt.savefig('plots.png')
'''

csv_filename = 'metrics.csv'
print("Loading data for plotting...")
df_plot = pd.read_csv(csv_filename)

# Increased Font Sizes
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=20)     
plt.rc('legend', title_fontsize=20) 
plt.rc('axes', titlesize=26)      
plt.rc('figure', titlesize=28)

fig, axs = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)

# Plot 1: Epistemic Distribution (Train vs Test)
sns.kdeplot(data=df_plot, x='Epistemic', hue='Dataset', fill=True, common_norm=False, alpha=0.4, palette=['blue', 'red'], ax=axs[0, 0])
axs[0, 0].set_title("(a) Epistemic Uncertainty Distribution")

# Plot 2: Aleatoric Distribution (Train vs Test)
sns.kdeplot(data=df_plot, x='Aleatoric', hue='Dataset', fill=True, common_norm=False, alpha=0.4, palette=['blue', 'red'], ax=axs[0, 1])
axs[0, 1].set_title("(b) Aleatoric Uncertainty Distribution")

# Filter just the test data for the scatter plots
df_test_only = df_plot[df_plot['Dataset'] == 'Test']

# Plot 3: Error vs Epistemic (Test)
sns.scatterplot(data=df_test_only, x='Epistemic', y='Error', alpha=0.3, color='purple', s=30, ax=axs[1, 0])
axs[1, 0].set_title("(c) Test Data: Error vs. Epistemic Uncertainty")

# Plot 4: Error vs Aleatoric (Test)
sns.scatterplot(data=df_test_only, x='Aleatoric', y='Error', alpha=0.3, color='orange', s=30, ax=axs[1, 1])
axs[1, 1].set_title("(d) Test Data: Error vs. Aleatoric Uncertainty")

plt.savefig('uncert_error.png')
print("Plots saved successfully.")