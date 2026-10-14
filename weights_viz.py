import torch
import matplotlib.pyplot as plt

# Load the file
weights = torch.load('beta_nll/beta_0.5/pnn_dynamics_weights.pth', map_location=torch.device('cpu'))

# Extract the weight matrix for the first layer and convert it to a NumPy array
layer_0_weights = weights['neural_net.0.weight'].numpy()

# 1. Plot a Histogram (shows the distribution/spread of weight values)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(layer_0_weights.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Weight Distribution (Layer 0)")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")

# 2. Plot a Heatmap (shows the actual matrix as colors)
plt.subplot(1, 2, 2)
plt.imshow(layer_0_weights, aspect='auto', cmap='viridis')
plt.colorbar(label='Weight Value')
plt.title("Weight Heatmap (Layer 0)")
plt.xlabel("Input Features")
plt.ylabel("Output Neurons")

plt.tight_layout()
plt.show()