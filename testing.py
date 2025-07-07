import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setting Random Seed 
np.random.seed(42)
torch.manual_seed(42)

# Generate training data: short windows of pure 1s or 0s
def generate_small_windows(n_samples=10000, window_size=3):
    data = []
    for _ in range(n_samples):
        if np.random.rand() < 0.5:
            data.append(np.ones(window_size))
        else:
            data.append(np.zeros(window_size))
    return np.array(data)

# Define a small autoencoder for 3-timestep windows
class SmallAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SmallAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Generate training data
window_size = 3
train_data = generate_small_windows(n_samples=10000, window_size=window_size)
train_tensor = torch.tensor(train_data, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

# Initialize model
model = SmallAutoencoder(input_dim=window_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

# Train the model
for epoch in range(7):
    for batch in train_loader:
        x = batch[0]
        output = model(x)
        loss = criterion(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
# Generate test sequence with a transition from 1 to 0
full_seq = np.concatenate([np.ones(50), np.zeros(50)])
errors = []

# Slide a 3-timestep window over the sequence
with torch.no_grad():
    for i in range(len(full_seq) - window_size + 1):
        window = full_seq[i:i + window_size]
        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        recon = model(input_tensor).squeeze(0).numpy()
        error = np.mean((window - recon) ** 2)
        errors.append(error)

error_series = [0] * (window_size // 2) + errors + [0] * (window_size - 1 - window_size // 2)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(full_seq)
plt.grid()
plt.title("Test Sequence (1s to 0s)")

plt.subplot(1, 2, 2)
plt.plot(error_series)
plt.title("Reconstruction Error (3-timestep windows)")
plt.tight_layout()
plt.grid()
plt.show()
