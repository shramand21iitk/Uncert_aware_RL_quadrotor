import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from PNN import PNN

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
#Add device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
#instantiate the model
model = PNN().to(device=device)
#define hyperparameters
batch_size = 1024
num_epochs = 2500
beta = 0.5
adv_ratio = 0.5
eps = 0.06
history = {
    'train_rmse': [], 'train_nll': [], 'train_cov': [], 'train_loss': [], 'train_LR': [],
    'test_rmse':  [], 'test_nll':  [], 'test_cov':  [], 'test_loss': []
}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
#get the dataset from stored CSV files
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
#prepare training data
x_train = df_train[input_cols]
y_train = df_train[output_cols]
x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
feature_std = torch.std(x_train, dim=0) + 1e-6 
feature_std = feature_std.to(device)
#prepare testing data
x_test = df_test[input_cols]
y_test = df_test[output_cols]
x_test = torch.tensor(x_test.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
#training loop
for epoch in range(num_epochs):
    model.train()
    running_train_mse = 0.0
    running_train_nll = 0.0
    running_train_cov = 0.0
    running_train_loss = 0.0
    total_train_samples = 0.0
    for batch_x, batch_y in train_dataloader:
        #Generate adverserial examples
        #batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        current_batch_size = batch_x.size(0)
        num_adv_samples = int(current_batch_size * adv_ratio)
        adv_x = batch_x[:num_adv_samples].clone().detach()
        adv_y = batch_y[:num_adv_samples]
        adv_x.requires_grad = True
        mean, var = model(adv_x)
        loss = model.gauss_beta_log_likelihood_loss(adv_y, mean, var, beta)
        optimizer.zero_grad()
        loss.backward()
        gradients = adv_x.grad.data
        adv_x = fgsm_attack(adv_x, eps, gradients, feature_std)
        adv_x = adv_x.detach()
        adv_x.requires_grad = False
        #Create a combined dataset of clean and adverserial data
        mixed_x = torch.cat([adv_x, batch_x], dim=0)
        mixed_y = torch.cat([adv_y, batch_y], dim=0)
        mixed_batch_size = mixed_x.size(0)
        total_train_samples += mixed_batch_size
        #train the model on combined data
        optimizer.zero_grad()
        mean, var = model.forward(mixed_x)
        loss = model.gauss_beta_log_likelihood_loss(target=mixed_y, pred_mean=mean, pred_var=var, beta=beta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # MSE
            mse = F.mse_loss(mean, mixed_y)
            running_train_mse += mse.item() * mixed_batch_size
            # Pure NLL (beta=0.0)
            nll = model.gauss_beta_log_likelihood_loss(mixed_y, mean, var, beta=0.0)
            running_train_nll += nll.item() * mixed_batch_size
            # Coverage
            std_dev = torch.sqrt(var)
            in_bounds = (mixed_y >= mean - 1.96 * std_dev) & (mixed_y <= mean + 1.96 * std_dev)
            running_train_cov += in_bounds.float().mean().item() * 100 * mixed_batch_size
            # Training loss
            running_train_loss += loss.item() * mixed_batch_size
    epoch_train_rmse = (running_train_mse / total_train_samples) ** 0.5
    epoch_train_nll = running_train_nll / total_train_samples
    epoch_train_cov = running_train_cov / total_train_samples
    epoch_train_loss = running_train_loss / total_train_samples
    current_lr = optimizer.param_groups[0]['lr']
    history['train_LR'].append(current_lr)
    history['train_rmse'].append(epoch_train_rmse)
    history['train_nll'].append(epoch_train_nll)
    history['train_cov'].append(epoch_train_cov)
    history['train_loss'].append(epoch_train_loss)
    scheduler.step(epoch_train_loss)
    #Model evaluation
    model.eval()
    running_test_mse = 0.0
    running_test_nll = 0.0
    running_test_cov = 0.0
    running_test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            current_batch_size = batch_x.size(0)
            mean, var = model(batch_x)
            loss = model.gauss_beta_log_likelihood_loss(target=batch_y, pred_mean=mean, pred_var=var, beta=1.0)
            # MSE
            mse = F.mse_loss(mean, batch_y)
            running_test_mse += mse.item() * current_batch_size
            # Pure NLL
            nll = model.gauss_beta_log_likelihood_loss(batch_y, mean, var, beta=0.0)
            running_test_nll += nll.item() * current_batch_size
            # Coverage
            std_dev = torch.sqrt(var)
            in_bounds = (batch_y >= mean - 1.96 * std_dev) & (batch_y <= mean + 1.96 * std_dev)
            running_test_cov += in_bounds.float().mean().item() * 100 * current_batch_size
            # Test loss
            running_test_loss += loss.item() * current_batch_size
    epoch_test_rmse = (running_test_mse / len(test_dataset)) ** 0.5
    epoch_test_nll = running_test_nll / len(test_dataset)
    epoch_test_cov = running_test_cov / len(test_dataset)
    epoch_test_loss = running_test_loss / len(test_dataloader.dataset)
    history['test_rmse'].append(epoch_test_rmse)
    history['test_nll'].append(epoch_test_nll)
    history['test_cov'].append(epoch_test_cov)
    history['test_loss'].append(epoch_test_loss)
    # LOGGING
    print(f"Epoch [{epoch+1:04d}/{num_epochs}]", flush=True)
    print(f"  Train -> RMSE: {epoch_train_rmse:.4f} | NLL: {epoch_train_nll:.4f} | Cov: {epoch_train_cov:.2f}% | Loss: {epoch_train_loss:.4f}", flush=True)
    print(f"  Test  -> RMSE: {epoch_test_rmse:.4f} | NLL: {epoch_test_nll:.4f} | Cov: {epoch_test_cov:.2f}% | Loss: {epoch_test_loss:.4f}\n", flush=True)
    # SAVE MODEL CHECKPOINT EVERY 5 EPOCHS
    if (epoch + 1) % 25 == 0:
        #Save history to csv file
        history_df = pd.DataFrame(history)
        history_df.to_csv("training_history.csv", index=False)
        #Save model checkpoint
        checkpoint_path = f"pnn_weights_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"*** Saved checkpoint: {checkpoint_path} ***", flush=True)
#save final trained model weights
print("Saving model weights...")
save_path = "pnn_dynamics_weights.pth"
torch.save(model.state_dict(), save_path)
print(f"Model successfully saved to {save_path}")
#Generate plots
print("Generating 4-Panel Metrics Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten() 
epochs_range = range(1, num_epochs + 1)
# Plot 1: RMSE
axes[0].plot(epochs_range, history['train_rmse'], label='Train RMSE', color='blue')
axes[0].plot(epochs_range, history['test_rmse'], label='Test RMSE', color='orange', linestyle='--')
axes[0].set_title('Root Mean Squared Error (Deterministic)')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('RMSE')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.7)
# Plot 2: Pure NLL
axes[1].plot(epochs_range, history['train_nll'], label='Train NLL', color='green')
axes[1].plot(epochs_range, history['test_nll'], label='Test NLL', color='red', linestyle='--')
axes[1].set_title('Negative Log-Likelihood (Pure Probabilistic)')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('NLL')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.7)
# Plot 3: 95% CI Coverage
axes[2].plot(epochs_range, history['train_cov'], label='Train Coverage', color='purple')
axes[2].plot(epochs_range, history['test_cov'], label='Test Coverage', color='brown', linestyle='--')
axes[2].axhline(y=95.0, color='black', linestyle='-.', alpha=0.5, label='95% Target') 
axes[2].set_title('95% Confidence Interval Coverage')
axes[2].set_xlabel('Epochs')
axes[2].set_ylabel('Coverage %')
axes[2].legend()
axes[2].grid(True, linestyle=':', alpha=0.7)
# Plot 4: Gaussian Beta NLL-Loss
axes[3].plot(epochs_range, history['train_loss'], label='Train Beta-NLL Loss', color='teal')
axes[3].plot(epochs_range, history['test_loss'], label='Test Beta-NLL Loss', color='magenta', linestyle='--')
axes[3].set_title('Beta-NLL Optimization Loss')
axes[3].set_xlabel('Epochs')
axes[3].set_ylabel('Beta-NLL Loss')
axes[3].legend()
axes[3].grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("training_dashboard.png", dpi=300, bbox_inches='tight')