import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PNN import PNN
from torch.utils.data import TensorDataset, DataLoader

# Perturbation function
def fgsm_attack(data, eps, grad, std_dev):
    sign_data_grad = grad.sign()
    scaled_perturbation = eps * std_dev * sign_data_grad
    perturbed_data = data + scaled_perturbation
    with torch.no_grad():
        quats = perturbed_data[:, 6:10] 
        norm = torch.linalg.norm(quats, dim=1, keepdim=True)
        perturbed_data[:, 6:10] = quats / (norm + 1e-8)
    return perturbed_data
# Testing FGSM attack
def test(model, device, test_loader, epsilon, b, std):
    mse_loss = 0.0
    nll_loss = 0.0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        mean, var = model(data)
        loss = model.gauss_beta_log_likelihood_loss(target, mean, var, beta=b)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad, std)
        mean_pert, var_pert = model(perturbed_data)
        mse = F.mse_loss(mean_pert, target)
        nll = model.gauss_beta_log_likelihood_loss(target, mean_pert, var_pert, beta=b)
        mse_loss += mse.item()
        nll_loss += nll.item()
        avg_mse = mse_loss / len(test_loader)
        avg_nll = nll_loss / len(test_loader)
    return avg_mse, avg_nll
# Add device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
# Instantiate the model
model = PNN().to(device=device)
pretrained_model = "beta_nll/beta_1.0/pnn_dynamics_weights.pth"
model.load_state_dict(torch.load(pretrained_model, map_location=device))
model.eval()
# Get the test dataset from stored CSV files
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
# Prepare testing data
x_test = df_test[input_cols]
y_test = df_test[output_cols]
x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(dataset=test_dataset)
feature_std = torch.std(x_test, dim=0) + 1e-6 
feature_std = feature_std.to(device)
# Hyperparameters
epsilons = np.arange(0.07, 0.1, 0.01).tolist()
beta_value = 1.0
MSE = []
NLL = []
torch.manual_seed(42)
# Loop through test dataset
for e in epsilons:
    mseloss, nllloss = test(model, device, test_dataloader, e, beta_value, feature_std)
    MSE.append(mseloss)
    NLL.append(nllloss)
    print(f"Epsilon: {e}\tTest MSE: {mseloss:.4f}\tTest NLL: {nllloss:.4f}")
# MSE LOSS plot
plt.figure(figsize=(5,5))
plt.plot(epsilons, MSE, "*-")
plt.title("MSE vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("mse_loss_vs_eps.png")
# NLL Loss plot
plt.figure(figsize=(5,5))
plt.plot(epsilons, NLL, "*-")
plt.title("NLL vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("NLL")
plt.tight_layout()
plt.savefig("nll_loss_vs_eps.png")
# Save to CSV
results_df = pd.DataFrame({
    'Epsilon': epsilons,
    'MSE': MSE,
    'NLL': NLL})
csv_filename = "fgsm_attack_results.csv"
results_df.to_csv(csv_filename, index=False)
print(f"Successfully saved numerical results to {csv_filename}")