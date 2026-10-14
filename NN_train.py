import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from NN import NN

# Add device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
# Instantiate the model
model = NN().to(device=device)
# Define hyperparameters
batch_size = 64
num_epochs = 2000
history = {
    'train_rmse': [], 'test_rmse':  []
}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
# Get the dataset from stored CSV files
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
# Prepare training data
x_train = torch.tensor(df_train[input_cols].values, dtype=torch.float32)
y_train = torch.tensor(df_train[output_cols].values, dtype=torch.float32)
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# Prepare testing data
x_test = torch.tensor(df_test[input_cols].values, dtype=torch.float32)
y_test = torch.tensor(df_test[output_cols].values, dtype=torch.float32)
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_train_mse = 0.0
    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        current_batch_size = batch_x.size(0)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = F.mse_loss(out, batch_y)
        loss.backward()
        optimizer.step()
        running_train_mse += loss.item() * current_batch_size
    epoch_train_rmse = (running_train_mse / len(train_dataset)) ** 0.5
    history['train_rmse'].append(epoch_train_rmse)
    # Model evaluation
    model.eval()
    running_test_mse = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            current_batch_size = batch_x.size(0)
            out = model(batch_x)
            loss = F.mse_loss(out, batch_y)
            running_test_mse += loss.item() * current_batch_size
    epoch_test_rmse = (running_test_mse / len(test_dataset)) ** 0.5
    history['test_rmse'].append(epoch_test_rmse)
    scheduler.step(epoch_train_rmse)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:04d}/{num_epochs}] | LR: {current_lr:.6f}")
        print(f"  Train -> RMSE: {epoch_train_rmse:.4f}")
        print(f"  Test  -> RMSE: {epoch_test_rmse:.4f}\n")
        history_df = pd.DataFrame(history)
        history_df.to_csv("training_history.csv", index=False)
        checkpoint_path = "nn_latest_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        if epoch == 0 or epoch_test_rmse < min(history['test_rmse'][:-1]):
             torch.save(model.state_dict(), "nn_best_weights.pth")
             print(f"*** New best model saved at epoch {epoch+1} ***", flush=True)
# Save final trained model weights
print("Saving model weights...")
save_path = "nn_baseline_weights.pth"
torch.save(model.state_dict(), save_path)
print(f"Model successfully saved to {save_path}")
# Generate plots
print("Generating Metrics Dashboard...")
plt.figure(figsize=(10, 6))
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, history['train_rmse'], label='Train RMSE', color='blue')
plt.plot(epochs_range, history['test_rmse'], label='Test RMSE', color='orange', linestyle='--')
plt.title('Baseline NN: Root Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("nn_training_dashboard.png", dpi=300, bbox_inches='tight')
plt.show()