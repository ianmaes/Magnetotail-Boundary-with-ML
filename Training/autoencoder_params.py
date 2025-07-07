import torch.nn as nn
import torch
import pandas as pd
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, n_features, latent_dim=8, time_steps=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features*time_steps, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_features*time_steps)
        )

        self._time_steps = time_steps

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def fit(self, data, test_data, epochs=100, batch_size=32, lr=0.001):
        """
        Train the autoencoder.
        
        Args:
            data: torch.Tensor of shape [num_samples, n_features * time_steps]
            test_data: torch.Tensor of shape [num_samples, n_features * time_steps]
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        """
        import torch.optim as optim
        
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
                decoded = self(batch_data)
                
                # Compute loss
                loss = criterion(decoded, batch_data)
                total_loss += loss.item()
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute test loss
            with torch.no_grad():
                test_decoded = self(test_data)
                test_loss = criterion(test_decoded, test_data)
                total_test_loss += test_loss.item()
            
            avg_loss = total_loss / num_batches
            avg_test_loss = total_test_loss
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        return self
    
    def evaluate_reconstruction_error(self, sessions_data, data_type='all_features', times=None):
        """
        Evaluate reconstruction error for sessions data and plot it over timestamps.
        
        Args:
            sessions_data: Dictionary where each key is a session and value contains the data
            data_type: Key to access the data in each session (default: 'all_features')
            times: Times array with nanosecond timestamps
        
        Returns:
            dict: Dictionary with session names as keys and reconstruction errors as values
        """
        import matplotlib.pyplot as plt
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        session_errors = {}
        
        with torch.no_grad():
            for session_name, session_data in sessions_data.items():
                # Get the data for this session
                data = session_data[data_type]
                
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                
                data = data.to(device)
                  
                # Get reconstruction
                reconstructed = self(data)
                
                # Calculate MSE for each sample
                mse = torch.mean((data - reconstructed) ** 2, dim=1)       
                
                session_errors[session_name] = mse.cpu().numpy()
        
        # Plot reconstruction errors for all sessions
        plt.figure(figsize=(12, 6))
        
        for session_name, errors in session_errors.items():
            if times is not None:
                # Create timestamp indices (centered on the middle of each window)
                timestamp_indices = np.arange(self._time_steps//2, len(errors) + self._time_steps//2)
                
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
    
    def prepare_data(self, data_trainable_samples, test_ratio=0.2):
        """
        Prepares the data for training by normalizing and creating sliding windows.
        Args:
            data_trainable_samples (dict): Dictionary containing training data samples.
                The structure is expected to be {session_key: {data_type_key: tensor}}.
            test_ratio (float): Ratio of data to use for testing (default: 0.2).
        Returns:
            tuple: (train_data, test_data) where each is a dictionary containing 
                   normalized and windowed data for each data type.
        """
        data = data_trainable_samples
        
        # Get the keys for data types and sessions
        data_type_keys = data[list(data.keys())[0]].keys()
        session_keys = list(data.keys())

        # Remove 'time', anything containing 'eflux', and anyting containing 'session' from data_type_keys
        data_type_keys = [key for key in data_type_keys if 'time' not in key and 'eflux' not in key and 'session' not in key]

        # Normalize each data type across all sessions
        for data_type_key in data_type_keys:
            # Create a tensor to store data for this data type of all sessions
            data_type_list = []

            for session_key in data.keys():
                # Retrieve the session tensor for the current session and data type
                session_tensor = data[session_key][data_type_key]
                data_type_list.append(session_tensor)

            # Concatenate all session tensors for this data type
            data_type_tensor = torch.cat(data_type_list, dim=0)

            # Calculate the number of timesteps
            if data_type_key == 'times':
                n_timesteps = data_type_tensor.shape[0] 

            else:
                # Calculate the mean and standard deviation for normalization
                mean = data_type_tensor.mean(dim=0)
                std = data_type_tensor.std(dim=0)

                for session_key in data.keys():
                    # Retrieve the session tensor for the current session and data type
                    session_tensor = data[session_key][data_type_key]

                    # Remove outliers from the session tensor
                    cleaned_tensor = remove_local_outliers(session_tensor)
                    
                    # Normalize the cleaned tensor
                    normalized_tensor = (cleaned_tensor - mean) / std

                    # Update the session data with the normalized tensor
                    data[session_key][data_type_key] = normalized_tensor

        # Create sliding windows for each data type
        windowed_data = {}
        for data_type_key in data_type_keys:
            windowed_tensors = []
            
            for session_key in session_keys:
                session_tensor = data[session_key][data_type_key]
                n_timesteps = session_tensor.shape[0]
                
                # Create sliding windows
                for i in range(n_timesteps - self._time_steps + 1):
                    if len(session_tensor.shape) == 1:
                        # Extract window of size self._time_steps
                        window = session_tensor[i:i+self._time_steps]

                    else:
                        # Extract window of size self._time_steps for multi-dimensional data
                        window = session_tensor[i:i+self._time_steps, :]

                    # Flatten to shape (n_features * self._time_steps,)
                    flattened_window = window.flatten()
                    windowed_tensors.append(flattened_window)
            
            # Stack all windows into tensor of shape (num_windows, n_features * self._time_steps)
            if windowed_tensors:
                windowed_data[data_type_key] = torch.stack(windowed_tensors)

        # Additionally, create a tensor where  all data types are concatenated
        # Create concatenated tensor with all data types
        if windowed_data:
            # Get all windowed tensors and concatenate along feature dimension
            all_data_tensors = list(windowed_data.values())
            concatenated_data = torch.cat(all_data_tensors, dim=1)
            windowed_data['all_features'] = concatenated_data

        # Split data into train and test sets
        train_data = {}
        test_data = {}
        
        for data_type_key, data_tensor in windowed_data.items():
            n_samples = data_tensor.shape[0]
            n_test = int(n_samples * test_ratio)
            
            # Split data
            test_data[data_type_key] = data_tensor[:n_test]
            train_data[data_type_key] = data_tensor[n_test:]

        return train_data, test_data

            
            

class AnomalyTransformer(nn.Module):
    def __init__(self, n_features, time_steps=3, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.time_steps = time_steps
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(time_steps, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, n_features)
        
        self.dropout = nn.Dropout(dropout)
        
    # Better approach - modify the forward method:
    def forward(self, x):
        # x should be (batch_size, time_steps, n_features) instead of flattened
        batch_size, time_steps, n_features = x.shape
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch_size, time_steps, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.dropout(x)
        
        # Apply transformer
        encoded = self.transformer(x)
        
        # Project back to original feature space
        reconstructed = self.output_projection(encoded)  # (batch_size, time_steps, n_features)
        
        return reconstructed  # Keep 3D structure

    def fit(self, data, test_data, epochs=100, batch_size=32, lr=0.001):
        """
        Train the anomaly transformer.
        
        Args:
            data: torch.Tensor of shape [num_samples, n_features * time_steps]
            test_data: torch.Tensor of shape [num_samples, n_features * time_steps]
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        """
        import torch.optim as optim
        
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
            
            self.train()
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_data = shuffled_data[start_idx:end_idx]
                
                # Forward pass
                reconstructed = self(batch_data)
                
                # Compute loss
                loss = criterion(reconstructed, batch_data)
                total_loss += loss.item()
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute test loss
            self.eval()
            with torch.no_grad():
                test_reconstructed = self(test_data)
                test_loss = criterion(test_reconstructed, test_data)
                total_test_loss += test_loss.item()
            
            avg_loss = total_loss / num_batches
            avg_test_loss = total_test_loss
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        return self

    def evaluate_reconstruction_error(self, sessions_data, data_type='all_features', times=None):
        """
        Evaluate reconstruction error for sessions data and plot it over timestamps.
        
        Args:
            sessions_data: Dictionary where each key is a session and value contains the data
            data_type: Key to access the data in each session (default: 'all_features')
            times: Times array with nanosecond timestamps
        
        Returns:
            dict: Dictionary with session names as keys and reconstruction errors as values
        """
        import matplotlib.pyplot as plt
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        session_errors = {}
        
        with torch.no_grad():
            for session_name, session_data in sessions_data.items():
                # Get the data for this session
                data = session_data[data_type]
                
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                
                data = data.to(device)
                    
                # Get reconstruction
                reconstructed = self(data)
                
                # Calculate MSE for each sample
                mse = torch.mean((data - reconstructed) ** 2, dim=1)       
                
                session_errors[session_name] = mse.cpu().numpy()
        
        # Plot reconstruction errors for all sessions
        plt.figure(figsize=(12, 6))
        
        for session_name, errors in session_errors.items():
            if times is not None:
                # Create timestamp indices (centered on the middle of each window)
                timestamp_indices = np.arange(self.time_steps//2, len(errors) + self.time_steps//2)
                
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

    def prepare_data(self, data_trainable_samples, test_ratio=0.2):
        """
        Prepares the data for training by normalizing and creating sliding windows.
        Args:
            data_trainable_samples (dict): Dictionary containing training data samples.
                The structure is expected to be {session_key: {data_type_key: tensor}}.
            test_ratio (float): Ratio of data to use for testing (default: 0.2).
        Returns:
            tuple: (train_data, test_data) where each is a dictionary containing 
                    normalized and windowed data for each data type.
        """
        data = data_trainable_samples
        
        # Get the keys for data types and sessions
        data_type_keys = data[list(data.keys())[0]].keys()
        session_keys = list(data.keys())

        # Remove 'time', anything containing 'eflux', and anyting containing 'session' from data_type_keys
        data_type_keys = [key for key in data_type_keys if 'time' not in key and 'eflux' not in key and 'session' not in key]

        # Normalize each data type across all sessions
        for data_type_key in data_type_keys:
            # Create a tensor to store data for this data type of all sessions
            data_type_list = []

            for session_key in data.keys():
                # Retrieve the session tensor for the current session and data type
                session_tensor = data[session_key][data_type_key]
                data_type_list.append(session_tensor)

            # Concatenate all session tensors for this data type
            data_type_tensor = torch.cat(data_type_list, dim=0)

            # Calculate the number of timesteps
            if data_type_key == 'times':
                n_timesteps = data_type_tensor.shape[0] 

            else:
                # Calculate the mean and standard deviation for normalization
                mean = data_type_tensor.mean(dim=0)
                std = data_type_tensor.std(dim=0)

                for session_key in data.keys():
                    # Retrieve the session tensor for the current session and data type
                    session_tensor = data[session_key][data_type_key]

                    # Remove outliers from the session tensor
                    cleaned_tensor = remove_local_outliers(session_tensor)

                    # Normalize the session tensor
                    normalized_tensor = (cleaned_tensor - mean) / std

                    # Update the session data with the normalized tensor
                    data[session_key][data_type_key] = normalized_tensor

        # Create sliding windows for each data type - DON'T flatten
        windowed_data = {}
        for data_type_key in data_type_keys:
            windowed_tensors = []
            
            for session_key in session_keys:
                session_tensor = data[session_key][data_type_key]
                n_timesteps = session_tensor.shape[0]
                
                # Create sliding windows
                for i in range(n_timesteps - self.time_steps + 1):
                    if len(session_tensor.shape) == 1:
                        # Keep as (time_steps, 1) instead of flattening
                        window = session_tensor[i:i+self.time_steps].unsqueeze(-1)
                    else:
                        # Keep as (time_steps, n_features) instead of flattening
                        window = session_tensor[i:i+self.time_steps, :]
                    
                    windowed_tensors.append(window)
            
            # Stack to (num_windows, time_steps, n_features)
            if windowed_tensors:
                windowed_data[data_type_key] = torch.stack(windowed_tensors)
        
        # For 'all_features', concatenate along feature dimension
        if windowed_data:
            all_data_tensors = list(windowed_data.values())
            concatenated_data = torch.cat(all_data_tensors, dim=2)  # Concatenate features, not flatten
            windowed_data['all_features'] = concatenated_data

        # Split data into train and test sets
        train_data = {}
        test_data = {}
        
        for data_type_key, data_tensor in windowed_data.items():
            n_samples = data_tensor.shape[0]
            n_test = int(n_samples * test_ratio)
            
            # Split data
            test_data[data_type_key] = data_tensor[:n_test]
            train_data[data_type_key] = data_tensor[n_test:]

        return train_data, test_data
    

    # Update the session data with the normalized tensor
# Remove outliers before normalization
def remove_local_outliers(data, window=10, threshold=3):
    clean_data = data.clone()
    if len(data.shape) == 1:
        # 1D data
        for i in range(window, len(data) - window):
            local = data[i - window : i + window + 1]
            local_mean = torch.mean(local)
            local_std = torch.std(local)
            if torch.abs(data[i] - local_mean) > threshold * local_std:
                clean_data[i] = (data[i - 1] + data[i + 1]) / 2
    else:
        # Multi-dimensional data
        for feature_idx in range(data.shape[1]):
            for i in range(window, data.shape[0] - window):
                local = data[i - window : i + window + 1, feature_idx]
                local_mean = torch.mean(local)
                local_std = torch.std(local)
                if torch.abs(data[i, feature_idx] - local_mean) > threshold * local_std:
                    clean_data[i, feature_idx] = (data[i - 1, feature_idx] + data[i + 1, feature_idx]) / 2
    return clean_data