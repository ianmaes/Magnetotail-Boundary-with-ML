import torch
import numpy as np
import pandas as pd
from datetime import datetime

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_dim=155, hidden_layer_1=128, hidden_layer_2=64, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, input_dim)
        )



    def forward(self, x):
        # Flatten input: [batch_size, timestamps, 31] -> [batch_size, timestamps * 31]
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        
        # Reshape back to original dimensions
        decoded = decoded.view(batch_size, x.shape[1], x.shape[2])
        
        return encoded, decoded

    def fit(self, data, test_data, epochs=100, batch_size=32, lr=0.001):
        """
        Train the autoencoder.
        
        Args:
            data: torch.Tensor of shape [num_samples, timestamps, 31]
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = data.to(device)
        test_data = test_data.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        num_samples = data.shape[0]
        num_batches = num_samples // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            total_test_loss = 0
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            shuffled_data = data[indices]
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_data = shuffled_data[start_idx:end_idx]
                
                # Forward pass
                _, decoded = self(batch_data)
                
                # Compute loss
                loss = criterion(decoded, batch_data)
                total_loss += loss.item()
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute test loss
            with torch.no_grad():
                _, test_decoded = self(test_data)
                test_loss = criterion(test_decoded, test_data)
                total_test_loss += test_loss.item()
            
            avg_loss = total_loss / num_batches
            avg_test_loss = total_test_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        return self

class ConvAutoencoder(nn.Module):
    def __init__(self, hidden_layer_1=16, hidden_layer_2=32, hidden_layer_3=32, latent_dim=32, timestamps=5):
        super().__init__()

        self._timestamps = timestamps
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_layer_1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Preserves frequency relationships
            nn.ReLU(),
            nn.Conv2d(hidden_layer_1, hidden_layer_2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_layer_2, hidden_layer_3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_layer_3 * self._timestamps * 31, latent_dim)  # Adjust dimensions based on your specific dimensions
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_layer_3 * self._timestamps * 31),  # Match the flattened dimensions
            nn.Unflatten(1, (hidden_layer_3, self._timestamps, 31)),  # Reshape back to feature maps
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_layer_3, hidden_layer_2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_layer_2, hidden_layer_1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_layer_1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
    
    
    def forward(self, x):
        # x shape: [batch_size, 4, 31]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, 4, 31]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.squeeze(1)  # Remove channel dimension: [batch_size, 4, 31]
        return encoded, decoded
    
    def fit(self, data, test_data, epochs=100, batch_size=32, lr=0.001):
        """
        Train the autoencoder.
        
        Args:
            data: torch.Tensor of shape [num_samples, timestamps, 31]
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = data.to(device)
        test_data = test_data.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        num_samples = data.shape[0]
        num_batches = num_samples // batch_size

        num_test_samples = test_data.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0
            total_test_loss = 0
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            shuffled_data = data[indices]
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_data = shuffled_data[start_idx:end_idx]
                
                # Forward pass
                _, decoded = self(batch_data)
                
                # Compute loss
                loss = criterion(decoded, batch_data)
                total_loss += loss.item()
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute test loss
            with torch.no_grad():
                _, test_decoded = self(test_data)
                test_loss = criterion(test_decoded, test_data)
                total_test_loss += test_loss.item()
            
            avg_loss =  total_loss / num_batches
            avg_test_loss = total_test_loss
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        return self
    
    def evaluate_reconstruction_error(self, sessions_data, data_type='B_ion_eflux', times=None):
        """
        Evaluate reconstruction error for sessions data and plot it over timestamps.
        
        Args:
            sessions_data: Dictionary where each key is a session and value contains the data
            data_type: Key to access the data in each session (default: 'spectrogram')
            times: Times array with nanosecond timestamps
        
        Returns:
            dict: Dictionary with session names as keys and reconstruction errors as values
        """
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        session_errors = {}
        
        with torch.no_grad():
            for session_name, session_data in sessions_data.items():
                # Get the spectrogram data for this session
                data = session_data[data_type]
                
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                
                data = data.to(device)
                  
                # Get reconstruction
                _, reconstructed = self(data)
                
                # Calculate MSE for this window
                mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2))       
                
                session_errors[session_name] = mse
        
        # Plot reconstruction errors for all sessions
        plt.figure(figsize=(12, 6))
        
        for session_name, errors in session_errors.items():
            if times is not None:
                # Create timestamp indices (centered on the middle of each window)
                timestamp_indices = np.arange(self._timestamps//2, len(errors) + self._timestamps//2)
                
                # Convert nanosecond timestamps to datetime
                datetime_stamps = pd.to_datetime(times[timestamp_indices], unit='ns')
                
                plt.plot(datetime_stamps, errors, label=f'Session {session_name}', alpha=0.7)
                plt.xlabel('Date and Time')
                plt.xticks(rotation=45)
            else:
                # Use simple indices when times are not provided
                plt.plot(errors, label=f'Session {session_name}', alpha=0.7)
                plt.xlabel('Sample Index')
        
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Reconstruction Error vs Time for All Sessions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return session_errors
    
    def calculate_mean_std(self, data, data_type='B_ion_eflux'):
        """
        Calculate mean and standard deviation for normalization.
        """
        all_data = []
        
        for key in data.keys():
            if not isinstance(data[key][data_type], torch.Tensor):
                data[key][data_type] = torch.tensor(data[key][data_type], dtype=torch.float32)
            curr_data = data[key][data_type]

            curr_data = torch.log(curr_data + 1e-9)  # Add small epsilon to avoid log(0)
            all_data.append(curr_data)
        
        combined_data = torch.cat(all_data, dim=0)
        mean = torch.mean(combined_data)
        std = torch.std(combined_data)
        
        self.register_buffer('normalization_mean', mean)
        self.register_buffer('normalization_std', std)
        
        return mean, std
    
    def prepare_data(self, data, data_type='B_ion_eflux', test_fraction=0.15, norm_mode='new', storage_mode='combined', rand_perm=True, averaging=True):
        """
        Prepare data for MAE transformer training.
        """
        
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        if norm_mode == 'new':
            all_data = []
            
            for key in data.keys():
                if not isinstance(data[key][data_type], torch.Tensor):
                    data[key][data_type] = torch.tensor(data[key][data_type], dtype=torch.float32, device=device)
                else:
                    data[key][data_type] = data[key][data_type].to(device)
                
                curr_data = data[key][data_type]
            
                
                log_data = torch.log(curr_data + 1e-9)  # Add small epsilon
                all_data.append(log_data)
            
            combined_data = torch.cat(all_data, dim=0)
            mean = torch.mean(combined_data)
            std = torch.std(combined_data)
            
            self.register_buffer('normalization_mean', mean)
            self.register_buffer('normalization_std', std)
            
        elif norm_mode == 'old':
            if not hasattr(self, 'normalization_mean') or not hasattr(self, 'normalization_std'):
                raise ValueError("Normalization parameters not found.")
            mean = self.normalization_mean
            std = self.normalization_std
        else:
            raise ValueError("Mode must be either 'new' or 'old'")
        
        normalized_data_all = []
        
        for key in data.keys():
            curr_data = data[key][data_type]
            if not isinstance(curr_data, torch.Tensor):
                curr_data = torch.tensor(curr_data, dtype=torch.float32, device=device)
            else:
                curr_data = curr_data.to(device)
            
            if averaging:
                curr_data = window_average_spectrogram(curr_data, window_size=3)
            
            log_data = torch.log(curr_data + 1e-9)
            normalized_data = (log_data - mean) / (std + 1e-9)
            normalized_data_all.append(normalized_data)
        
        all_samples = []
        
        if storage_mode == 'seperated':
            test_data_dict = {f'section_{i:03d}': {} for i in range(len(normalized_data_all))}
            normalized_data_dict = {f'section_{i:03d}': {} for i in range(len(normalized_data_all))}
            
            for session_idx in range(len(normalized_data_all)):
                session_data = normalized_data_all[session_idx]
                num_samples = session_data.shape[0] - self._timestamps + 1
                
                session_samples = []
                for i in range(num_samples):
                    sample = session_data[i:i+self._timestamps]
                    session_samples.append(sample)
                
                section_key = f'section_{session_idx:03d}'
                test_data_dict[section_key][data_type] = torch.stack(session_samples).to(device) if session_samples else torch.empty(0, self._timestamps, session_data.shape[1], device=device)
                normalized_data_dict[section_key][data_type] = normalized_data_all[session_idx]
                
                # Copy time data if it exists
                original_key = list(data.keys())[session_idx]
                if 'times' in data[original_key]:
                    normalized_data_dict[section_key]['times'] = data[original_key]['times']
            
            return test_data_dict, normalized_data_dict
        else:
            for session_idx in range(len(normalized_data_all)):
                session_data = normalized_data_all[session_idx]
                num_samples = session_data.shape[0] - self._timestamps + 1
                
                for i in range(num_samples):
                    sample = session_data[i:i+self._timestamps]
                    all_samples.append(sample)
            
            samples = torch.stack(all_samples).to(device)
            
            # Split data
            num_samples = len(all_samples)
            test_size = int(num_samples * test_fraction)
            train_size = num_samples - test_size
            
            if rand_perm:
                indices = torch.randperm(num_samples, device=device)
                train_indices = indices[:train_size]
                test_indices = indices[train_size:]
            else:
                train_indices = torch.arange(train_size, device=device)
                test_indices = torch.arange(train_size, num_samples, device=device)
            
            train_samples = samples[train_indices]
            test_samples = samples[test_indices]
            
            return train_samples, test_samples


def window_average_spectrogram(spectrogram, window_size=3):
    """
    Apply a sliding window average to spectrogram data.
    
    Parameters:
        spectrogram (tensor): Size of the sliding window for averaging 
        data_type (str): Type of spectrogram data to process (e.g., 'B_ion_eflux')
        
    Returns:
        dict: Processed data with averaged spectrograms for each session
    """
     
    n_timestamps, n_energy_bins = spectrogram.shape
    half_window = window_size // 2
    
    # Initialize the averaged spectrogram with the same shape
    averaged_spectrogram = torch.zeros_like(spectrogram)
    
    for i in range(n_timestamps):
        # Define window boundaries with padding
        start_idx = max(0, i - half_window)
        end_idx = min(n_timestamps, i + half_window + 1)
        
        # Average over the available window for each energy bin
        window_data = spectrogram[start_idx:end_idx, :]
        averaged_spectrogram[i, :] = torch.mean(window_data, dim=0)
        
    return averaged_spectrogram


if __name__ == "__main__":
    # Example usage
    model = ConvAutoencoder()
    print(model)
    
    # Synthetic data example
    # Replace this with your actual data loading
    synthetic_data = np.random.randn(1000, 16)  # 1000 timestamps, 16 features per timestamp
    data = prepare_data(synthetic_data)
    print(f"Data shape: {data.shape}")
    
    # Train the model
    train_autoencoder(model, data, epochs=100)
    
    # Example of encoding data to the latent space
    with torch.no_grad():
        encoded, _ = model(data[:10])
        print(f"Encoded shape: {encoded.shape}")
        print("Encoded values (first 3 samples):")
        print(encoded[:3])