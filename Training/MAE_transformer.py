import torch

import math

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MAETransformer(nn.Module):
    def __init__(self, input_dim=31, timestamps=4, d_model=64, nhead=4, num_encoder_layers=4, 
             num_decoder_layers=4, dim_feedforward=512, mask_ratio=0.50, patches_per_timestamp=1):
        super().__init__()
        
        self._timestamps = timestamps
        self.original_input_dim = input_dim  # Store original dimension
        self.patches_per_timestamp = patches_per_timestamp
        
        # Calculate padded input dimension to make it divisible
        if input_dim % patches_per_timestamp != 0:
            self.padded_input_dim = ((input_dim // patches_per_timestamp) + 1) * patches_per_timestamp
            self.padding_size = self.padded_input_dim - input_dim
            print(f"Padding input_dim from {input_dim} to {self.padded_input_dim} to make it divisible by {patches_per_timestamp}")
        else:
            self.padded_input_dim = input_dim
            self.padding_size = 0
        
        self.input_dim = self.padded_input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        
        # Calculate patch size using padded dimension
        self.patch_size = self.padded_input_dim // patches_per_timestamp
        
        # Rest of initialization remains the same...
        self.total_patches = timestamps * patches_per_timestamp
        self.patch_embedding = nn.Linear(self.patch_size, d_model)

        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.total_patches)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.patch_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def patchify(self, x):
        """
        Convert input to patches with automatic padding if needed.
        x: [B, T, original_input_dim] -> [B, T*patches_per_timestamp, patch_size]
        """
        B, T, D = x.shape
        
        # Apply padding if needed
        if self.padding_size > 0:
            padding = torch.zeros(B, T, self.padding_size, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)  # [B, T, padded_input_dim]
        
        # Reshape to create patches
        x = x.view(B, T, self.patches_per_timestamp, self.patch_size)
        # Reshape to sequence of patches
        x = x.view(B, T * self.patches_per_timestamp, self.patch_size)
        return x

    def unpatchify(self, x):
        """
        Convert patches back to original format, removing padding.
        x: [B, T*patches_per_timestamp, patch_size] -> [B, T, original_input_dim]
        """
        B, total_patches, patch_size = x.shape
        # Reshape to separate timestamps and patches
        x = x.view(B, self._timestamps, self.patches_per_timestamp, patch_size)
        # Reshape to padded format
        x = x.view(B, self._timestamps, self.padded_input_dim)
        
        # Remove padding to get back to original dimension
        if self.padding_size > 0:
            x = x[:, :, :self.original_input_dim]
        
        return x
        
    def random_masking(self, x):
        """
        Perform per-sample random masking on patches
        """
        N, L, D = x.shape  # batch, length (total_patches), dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio=None):
        # Convert to patches
        x = self.patchify(x)  # [B, total_patches, patch_size]
        
        # Embed patches
        x = self.patch_embedding(x)  # [B, total_patches, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Masking: length -> length * mask_ratio
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
            
        x, mask, ids_restore = self.random_masking(x)
        
        # Apply Transformer encoder
        x = self.encoder(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.norm(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # Add positional encoding
        x_ = self.pos_encoding(x_)
        
        # Apply Transformer decoder
        # For MAE, we can use the encoder output as both memory and target
        memory = x_.clone()
        x_ = self.decoder(x_, memory)
        
        # Predictor projection
        x_ = self.output_proj(x_)  # [B, total_patches, patch_size]
        
        # Convert back to original format
        x_ = self.unpatchify(x_)  # [B, timestamps, input_dim]
        
        return x_
    
    def forward(self, x):
        # x shape: [batch_size, timestamps, input_dim]
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)  # [N, timestamps, input_dim]
        
        return latent, pred, mask
    
    def fit(self, data, test_data, epochs=100, batch_size=32, lr=0.001):
        """
        Train the MAE transformer.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = data.to(device)
        test_data = test_data.to(device)
        
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.05)
        
        num_samples = data.shape[0]
        num_batches = num_samples // batch_size

        loss_list = []
        test_loss_list = []

        for epoch in range(epochs):
            total_loss = 0
            total_test_loss = 0

            
            # Shuffle data
            indices = torch.randperm(num_samples)
            shuffled_data = data[indices]

            criterion = nn.MSELoss(reduction='none')
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_data = shuffled_data[start_idx:end_idx]
                
                # Forward pass
                _, pred, mask = self(batch_data)
                
                # Convert mask back to original format for loss calculation
                # mask is for patches, need to map back to original dimensions
                batch_size_curr = batch_data.shape[0]
                mask_reshaped = mask.view(batch_size_curr, self._timestamps, self.patches_per_timestamp)
                
                # Compute loss
                loss = criterion(pred, batch_data)  # [B, T, input_dim]
                
                # Apply mask - if any patch in a timestamp is masked, mask the whole timestamp
                timestamp_mask = mask_reshaped.any(dim=2).float()  # [B, T]
                timestamp_mask = timestamp_mask.unsqueeze(-1).expand_as(batch_data)  # [B, T, input_dim]
                
                masked_loss = (loss * timestamp_mask).sum() / (timestamp_mask.sum() + 1e-8)
                
                total_loss += masked_loss.item()

                # Add batch loss to loss list
                
                
                # Backward pass
                optimizer.zero_grad()
                masked_loss.backward()
                optimizer.step()
            
            # Compute test loss
            with torch.no_grad():
                _, test_pred, test_mask = self(test_data)
                test_loss = criterion(test_pred, test_data)
                test_batch_size = test_data.shape[0]
                test_mask_reshaped = test_mask.view(test_batch_size, self._timestamps, self.patches_per_timestamp)
                test_timestamp_mask = test_mask_reshaped.any(dim=2).float()
                test_timestamp_mask = test_timestamp_mask.unsqueeze(-1).expand_as(test_data)
                masked_test_loss = (test_loss * test_timestamp_mask).sum() / (test_timestamp_mask.sum() + 1e-8)
                total_test_loss = masked_test_loss.item()

            loss_list.append(total_loss / num_batches)
            test_loss_list.append(total_test_loss)
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {total_test_loss:.4f}")
        
        return self, loss_list, test_loss_list
    
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
        # Import the window_average_spectrogram function (assuming it exists)
        # from your_utils_module import window_average_spectrogram
        
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        if norm_mode == 'new':
            all_data = []
            
            for key in data.keys():
                if not isinstance(data[key][data_type], torch.Tensor):
                    data[key][data_type] = torch.tensor(data[key][data_type], dtype=torch.float32, device=device)
                else:
                    data[key][data_type] = data[key][data_type].to(device)
                
                curr_data = data[key][data_type]
                
                if averaging:
                    curr_data = window_average_spectrogram(curr_data, window_size=self._timestamps)
                
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

def huber_loss(pred, target, delta=0.2):  # delta is key parameter
    error = torch.abs(pred - target)
    return torch.where(error < delta, 
                      0.5 * error ** 2,      # MSE for small errors
                      delta * (error - 0.5 * delta))  # Linear for large errors