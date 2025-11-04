import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from Helper.preprocess_functions import average_window, remove_outliers_with_interpolation, remove_outliers_with_local_interpolation
from itertools import combinations

class MAETransformerMulti(nn.Module):
    """
    MAE Transformer for multi-modal data, including spectrograms, vectors, and scalars.
    """

    # Constructor
    def __init__(self, input_dim=31, timestamps=4, d_model=32, nhead=4, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=256, mask_ratio=0.30, patches_per_timestamp_spect=4,
                 n_vectors=0, n_scalars=0, dropout=0.1, vector_dim=3, noise=None, device='cuda', mask_scalars=False):
        """
        Initialize the MAE Transformer for multi-modal data processing.
        This constructor sets up a Masked Autoencoder (MAE) Transformer that can handle
        multiple data modalities including spectrograms, vector data, and scalar data.
        The model uses patch-based processing and attention mechanisms for reconstruction.

        Args:
            input_dim (int, optional):          Number of energy bins in the input spectrogram. Defaults to 31.
            timestamps (int, optional):         Number of timestamps in the input data. Defaults to 4.
            d_model (int, optional):            Dimension of the model embeddings. Defaults to 32.
            nhead (int, optional):              Number of attention heads in transformer layers. Defaults to 4.
            num_encoder_layers (int, optional): Number of encoder transformer layers. Defaults to 4.
            num_decoder_layers (int, optional): Number of decoder transformer layers. Defaults to 4.
            dim_feedforward (int, optional):    Dimension of the feedforward network in transformer layers. Defaults to 256.
            mask_ratio (float, optional):       Ratio of patches to mask during training (0.0-1.0). Defaults to 0.30.
            patches_per_timestamp_spect (int, optional): Number of patches per timestamp for spectrogram data. Defaults to 4.
            n_vectors (int, optional):          Number of vector modalities in the input data. Defaults to 0.
            n_scalars (int, optional):          Number of scalar modalities in the input data. Defaults to 0.
            dropout (float, optional):          Dropout rate for regularization. Defaults to 0.1.
            vector_dim (int, optional):         Dimension of each vector modality. Defaults to 3.
            device (str, optional):             Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

        Note:
            - Input dimensions are automatically padded if not divisible by patches_per_timestamp_spect
            - Total patches are calculated as timestamps × (patches_per_timestamp_spect + n_vectors + min(1, n_scalars))
            - Separate embedding layers are created for each modality type
        """
        super().__init__()

        # Set device on which model performs operations
        self.device = device

        ## --- INPUT/MODEL PARAMETER DEFINITION --- ##
        # Defining the input dimensions of the spectrogram (usually 31 [bins]), and how that will be split up into patches.
        # Defining the model dimension and its mask ratio (how many patches are masked when using the model).

        # Input dimension of the vector and scalar data
        self.n_vectors = n_vectors
        self.n_scalars = n_scalars
        self.mask_scalars = mask_scalars

        # Get vector dimension
        if n_vectors > 0:
            self.vector_dim = vector_dim 

        # Number of timestamps per sample
        self.timestamps = timestamps

        # How many patches the Spectrogram is divided into, also determines patch size for vectors and scalars
        self.patches_per_timestamp = patches_per_timestamp_spect + n_vectors + min(1, n_scalars)
        self.patches_per_timestamp_spect = patches_per_timestamp_spect

        # Calculate the padded input dimension (if padding is necessary due to patches not lining up with input dimension)
        if input_dim % patches_per_timestamp_spect != 0:
            self.input_dim = (input_dim // patches_per_timestamp_spect + 1) * patches_per_timestamp_spect
            self.padding_size = self.input_dim - input_dim
            print(f"Padding input_dim from {input_dim} to {self.input_dim} to make it divisible by {patches_per_timestamp_spect}")

        # If no padding is necessary, set padding size to 0
        else:
            self.input_dim = input_dim
            self.padding_size = 0

        # Calculate the patch size by dividing the input dimension (of spectrogram) by the number of patches per timestamp for spectrogram
        self.patch_size = self.input_dim // patches_per_timestamp_spect

        # Define the masking ratio of the MAE transformer, and define the model dimension
        self.mask_ratio = mask_ratio
        self.d_model = d_model

        # Calculate ratio of scalar patches to total patches
        self.scalar_patch_ratio = (1 / self.patches_per_timestamp) if self.n_scalars > 0 else 0

        # Calculate ratio of vector patches to total patches
        self.vector_patch_ratio = (self.n_vectors / self.patches_per_timestamp) if self.n_vectors > 0 else 0

        # Calculate ratio of spectrogram patches to total patches
        self.spect_patch_ratio = (self.patches_per_timestamp_spect / self.patches_per_timestamp)

        
        ## --- INPUT PATCH EMBEDDINGS --- ##
        # Creation of the linear layers that translate the patches containing raw data into the model dimension.
        # This is seperate per data shape (Spectrogram, vectors, scalars). For vectors, it is split up per data type as well.

        # Calculate total number of patches
        self.total_patches = timestamps * (self.patches_per_timestamp)

        # Create patch embedding for spectrograms
        self.patch_embedding = nn.Linear(self.patch_size, self.d_model)
        
        # Create patch embeddings for vectors if they exist
        if n_vectors > 0:
            # Create separate projections for each vector type
            self.vector_embedding = nn.ModuleList([
                nn.Linear(vector_dim, self.d_model) for _ in range(n_vectors)
            ])

        # Create patch embedding for scalars if they exist 
        if n_scalars > 0:
            self.scalar_embedding = nn.Linear(n_scalars, self.d_model)  # Project to patch_size first

      
        ## --- POSITIONAL ENCODING --- ##
        # Creation of the positional encoding to add positional information to the patches. 
        # Done using sine and cosine waves.

        # Initialize the positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.total_patches)


        ## --- TRANSFORMER ENCODER & DECODER --- ##
        # Creation encoder and decoder transformer layers.
        # Creation of the mask token. The standard values masked patches (in model dimension) will get to build upon. 

        # Mask token for the MAE transformer
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )

        # Decoder
        decoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layers, 
            num_layers=num_decoder_layers
        )


        ## --- OUTPUT PROJECTIONS --- ##
        # Linear layers that will translate the output of the decoder into "raw" data patches to compare to initial masked data.
        # Again, seperate for each data type. Different data types communicate within the transformers.

        # Final linear layer to project the output back to the input dimension for spectrogram
        self.output_proj_spect = nn.Linear(d_model, self.patch_size)

        # Additional output projections for vectors and scalars
        if n_vectors > 0:
            self.output_proj_vectors = nn.ModuleList([
                nn.Linear(d_model, vector_dim) for _ in range(n_vectors)
            ])
        if n_scalars > 0:
            self.output_proj_scalars = nn.Linear(d_model, n_scalars)

        # Normalization layer
        self.norm = nn.LayerNorm(d_model)


        ## --- RECONSTRUCTION ERROR STORAGE -- ##
        # These parameters store the avg and std reconstruction error of the last epoch in training, using the constant mask.
        # This is done such that when comparing tail and sheath reconstruction error, errors are normalized to usual model performance.

        # Instantiate average reconstruction error
        self.avg_recon_error = None
        self.std_recon_error = None


        ## --- MASK CREATION --- ##
        # Creation of a constant noise mask for the MAE transformer based on the patches per timestamp.
        # This is used for evaluation purposes (not training) to have a consistent mask when using the model.
        # This makes performance consistent. 100 noise patterns are created such that model performance is (largely)
        # independent of the random mask initialization.

        # Set constant mask to false
        self.const_mask = False

        # Create a constant noise mask for the MAE transformer based on the patches per timestamp
        if noise is not None:
            self.noises = noise.to(device)
        else:
            if self.timestamps < 2:
                # For single timestamp, create all possible mask patterns for consistency
                total_patches = self.patches_per_timestamp
                patches_to_mask = int(round(total_patches * self.mask_ratio))
                
                if self.n_scalars > 0 and not self.mask_scalars:
                    # Generate all possible combinations of which patches to mask
                    all_combinations = list(combinations(range(total_patches - 1), patches_to_mask))
                    noises = torch.zeros(len(all_combinations), total_patches, device=device)
                else:
                    # If no scalars, ensure scalar patches are never masked
                    all_combinations = list(combinations(range(total_patches), patches_to_mask))
                    noises = torch.zeros(len(all_combinations), total_patches, device=device)
                
                for i, mask_indices in enumerate(all_combinations):
                    # Set low values for patches to keep (high priority)
                    noises[i] = torch.zeros(total_patches, device=device)
                    # Set high values for patches to mask (low priority)
                    noises[i, list(mask_indices)] = 1.0
                    
       
                self.noises = noises
                print(f"Created {len(all_combinations)} deterministic noise patterns for single timestamp")
            else:

                # Create 500 noise patterns and cycle through them during forward passes for evaluation
                noises = torch.zeros(200, self.patches_per_timestamp * self.timestamps, device=device)
                for i in range(noises.shape[0]):
                    noises[i] = torch.rand(self.patches_per_timestamp * self.timestamps, device=device)

                self.noises = noises
    
    def get_noise(self):
        return self.noises
    
    def save_model(self, path):
        """Save the model to the specified file path."""
        torch.save(self.state_dict(), path)
        # Save model mask
        torch.save(self.noises, path.replace('.pth', '_mask.pth'))

        # Save mean and std of training data
        torch.save(self.means, path.replace('.pth', '_means.pth'))
        torch.save(self.stds, path.replace('.pth', '_stds.pth'))

        # Save mean and std of reconstruction error
        torch.save(self.avg_recon_error, path.replace('.pth', '_avg_recon_error.pth'))
        torch.save(self.std_recon_error, path.replace('.pth', '_std_recon_error.pth'))

        # Save mask ratio
        torch.save(self.mask_ratio, path.replace('.pth', '_mask_ratio.pth'))

        # Save amount of timestamps used
        torch.save(self.timestamps, path.replace('.pth', '_timestamps.pth'))

    def load_model(self, path):
        """Load the model from the specified file path."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)

        # Load mean and std of training data if available
        if os.path.exists(path.replace('.pth', '_means.pth')):
            self.means = torch.load(path.replace('.pth', '_means.pth'), map_location=self.device)
        if os.path.exists(path.replace('.pth', '_stds.pth')):
            self.stds = torch.load(path.replace('.pth', '_stds.pth'), map_location=self.device)

        # Load mean and std of reconstruction error if available
        if os.path.exists(path.replace('.pth', '_avg_recon_error.pth')):
            self.avg_recon_error = torch.load(path.replace('.pth', '_avg_recon_error.pth'), map_location=self.device)
        if os.path.exists(path.replace('.pth', '_std_recon_error.pth')):
            self.std_recon_error = torch.load(path.replace('.pth', '_std_recon_error.pth'), map_location=self.device)

        # Load mask ratio if available
        if os.path.exists(path.replace('.pth', '_mask_ratio.pth')):
            self.mask_ratio = torch.load(path.replace('.pth', '_mask_ratio.pth'), map_location=self.device)


        # Load amount of timestamps used if available
        if os.path.exists(path.replace('.pth', '_timestamps.pth')):
            self.timestamps = torch.load(path.replace('.pth', '_timestamps.pth'), map_location=self.device)
        else:
            self.timestamps = 4

    # Forward pass functions
    def patchify(self, x_spectrogram, x_vectors=None, x_scalars=None):
        """
        Creates patches from all modalities' "raw" data using learnable projections.
        Args:
            x_spectrogram:  [B, T, D] - Batch size, timestamps, frequency bins
            x_vectors:      [B, T, n_vectors, vector_dim] - Batch size, timestamps, number of vectors, vector dimension
            x_scalars:      [B, T, n_scalars] - Batch size, timestamps, number of scalars

        Returns:
            patches:        [B, total_patches, patch_size] - All patches from all modalities
            patch_types:    [total_patches] - Type indicator for each patch (0=spectrogram, 1=vector, 2=scalar)
        """
        B, T, D = x_spectrogram.shape
        all_patches = []
        patch_types = []
        
        # Process each timestamp
        for t in range(T):
            # Add spectrogram patches for this timestamp
            spec_data = x_spectrogram[:, t, :].unsqueeze(1)  # [B, 1, D]
            
            # Apply padding if needed
            if self.padding_size > 0:
                padding = torch.zeros(B, 1, self.padding_size, device=spec_data.device, dtype=spec_data.dtype)
                spec_data = torch.cat([spec_data, padding], dim=-1)
            
            # Create patches for this timestamp
            spec_patches = spec_data.view(B, self.patches_per_timestamp_spect, self.patch_size)
            spec_patches = self.patch_embedding(spec_patches)  # [B, patches_per_timestamp, d_model]
            
            all_patches.append(spec_patches)
            patch_types.extend([0] * self.patches_per_timestamp_spect)
            
            # Add vector patches for this timestamp
            if x_vectors is not None:
                for v in range(x_vectors.shape[2]):  # n_vectors
                    vector_data = x_vectors[:, t, v, :].unsqueeze(1)  # [B, 1, vector_dim]
                    # Use the specific projection for this vector type
                    vector_patch = self.vector_embedding[v](vector_data)  # [B, 1, d_model]
                    
                    all_patches.append(vector_patch)
                    patch_types.append(1)  # 1 = vector
                    
            # Add scalar patches for this timestamp
            if x_scalars is not None and x_scalars.shape[2] > 0:
                scalar_data = x_scalars[:, t, :].unsqueeze(1)  # [B, 1, n_scalars]
                # Use learnable projection instead of padding
                scalar_patch = self.scalar_embedding(scalar_data)  # [B, 1, d_model]
                all_patches.append(scalar_patch)
                patch_types.append(2)  # 2 = scalar
        
        # Concatenate all patches
        patches = torch.cat(all_patches, dim=1)  # [B, total_patches, d_model]

        # Add positional encoding
        patches = self.positional_encoding(patches)

        patch_types = torch.tensor(patch_types, device=patches.device)
        
        return patches, patch_types
    
    def unpatchify(self, x_patches, patch_types, mask):
        """
        Reconstructs the original spectrogram from patches and creates separate masks for each modality. (As of right now, scalars are never masked, only the spectrogram).
        Args:
            x_patches:      [B, T * patches_per_timestamp, d_model]
            patch_types:    [T * patches_per_timestamp] - Type indicator for each patch (0=spectrogram, 1=vector, 2=scalar)
            mask:           [B, T * patches_per_timestamp] - Binary mask (0=keep, 1=mask)
        Returns:
            tuple: A tuple containing:
                - spect_data: [B, T, input_dim] - Reconstructed spectrogram
                - vector_data: [B, T, n_vectors, vector_dim] - Reconstructed vectors
                - scalar_data: [B, T, n_scalars] - Reconstructed scalars
                - spect_mask: [B, T, input_dim] - Mask for spectrogram loss
                - vector_mask: [B, T, n_vectors, vector_dim] - Mask for vector loss
                - scalar_mask: [B, T, n_scalars] - Mask for scalar loss
        """
        B, L, D = x_patches.shape
        T = L // self.patches_per_timestamp

        # Reshape patch types and x_patches
        patch_types = patch_types.view(T, self.patches_per_timestamp)
        x_patches = x_patches.view(B, T, self.patches_per_timestamp, D)
        mask = mask.view(B, T, self.patches_per_timestamp)

        # Initialize all modalities with zeros
        spect_data_patches = torch.zeros(B, T, self.patches_per_timestamp_spect, self.patch_size, device=x_patches.device, dtype=x_patches.dtype)
        vector_data = torch.zeros(B, T, self.n_vectors, self.vector_dim, device=x_patches.device, dtype=x_patches.dtype) if self.n_vectors > 0 else None
        scalar_data = torch.zeros(B, T, self.n_scalars, device=x_patches.device, dtype=x_patches.dtype) if self.n_scalars > 0 else None

        # Initialize masks for each modality
        spect_mask_patches = torch.zeros(B, T, self.patches_per_timestamp_spect, self.patch_size, device=x_patches.device, dtype=x_patches.dtype)
        vector_mask = torch.zeros(B, T, self.n_vectors, self.vector_dim, device=x_patches.device, dtype=x_patches.dtype) if self.n_vectors > 0 else None
        scalar_mask = torch.zeros(B, T, self.n_scalars, device=x_patches.device, dtype=x_patches.dtype) if self.n_scalars > 0 else None

        # Create patch indices to track location of the patches in the output
        vector_index = 0
        spect_index = 0

        for t in range(T):
            # Get the patch types for the current timestamp
            patch_types_t = patch_types[t, :]
            spect_index = 0
            vector_index = 0

            for patch_index in range(self.patches_per_timestamp):
                # Get the type of the current patch
                patch_type = patch_types_t[patch_index]

                ### Process the patch based on its type ###
                if patch_type == 0:
                    # Perform the inverse projection for spectrogram patches
                    spect_data_patches[:, t, spect_index, :] = \
                        self.output_proj_spect(x_patches[:, t, patch_index, :])
                    
                    # Set mask for spectrogram patches
                    spect_mask_patches[:, t, spect_index, :] = mask[:, t, patch_index].unsqueeze(-1).expand(-1, self.patch_size)

                    # Increment the spectrogram patch index
                    spect_index += 1



                elif patch_type == 1:
                    # Perform the inverse projection for vector patches
                    vector_data[:, t, vector_index, :] = \
                        self.output_proj_vectors[vector_index](x_patches[:, t, patch_index, :])
                    
                    # Set mask for vector patches
                    vector_mask[:, t, vector_index, :] = mask[:, t, patch_index].unsqueeze(-1).expand(-1, self.vector_dim)

                    # Increment the vector patch index
                    vector_index += 1

                elif patch_type == 2:
                    # Perform the inverse projection for scalar patches
                    scalar_data[:, t, :] = \
                        self.output_proj_scalars(x_patches[:, t, patch_index, :])
                    
                    # Set mask for scalar patches
                    scalar_mask[:, t, :] = mask[:, t, patch_index].unsqueeze(-1).expand(-1, self.n_scalars)
        
        # Concatenate the spectrogram patches to [B, T, input_dim]
        spect_data = spect_data_patches.view(B, T, -1)
        spect_mask = spect_mask_patches.view(B, T, -1)

        # Remove padding if it was added
        if self.padding_size > 0:
            spect_data = spect_data[:, :, :-self.padding_size]
            spect_mask = spect_mask[:, :, :-self.padding_size]

        return spect_data, vector_data, scalar_data, spect_mask, vector_mask, scalar_mask
    
    def random_masking(self, x, patch_types=None):
        """
        Per-sample random masking of patch embeddings.

        Args:
            x (torch.Tensor):   [B, L, D] patch-embedded sequence
            patch_types (torch.Tensor): [L] - Type indicator for each patch (0=spectrogram, 1=vector, 2=scalar)
            mask_ratio (float): float in (0, 1) – fraction of patches to mask
            mask_scalars (bool): if True, treat scalar patches as normal patches and mask them randomly
        
        Returns:
            tuple: A tuple containing:
                - x_keep (torch.Tensor):  kept (unmasked) tokens [B, L*(1-r), D]
                - mask (torch.Tensor):    binary mask in original order [B, L] (0=keep, 1=mask)
                - ids_restore (torch.Tensor):   indices that restore original order [B, L]
        """

        B, L, D = x.shape
        
        # If mask_scalars is True, treat all patches equally (original behavior)
        if self.mask_scalars or patch_types is None or self.n_scalars == 0:
            # Original behavior when no scalar constraints or when mask_scalars=True
            len_keep = int(round(L * (1 - self.mask_ratio), 0))
            
            if self.const_mask:
                # If constant mask is used, select the premade noise parameter, same for each sample
                noise = self.noise.expand(B, -1)  # Expand noise to match batch size
            else:
                # random noise, different for each sample
                noise = torch.rand(B, L, device=x.device)
            
            ids_shuffle = torch.argsort(noise, dim=1)
        
        else:
            # Configuration: number of scalar patches to mask (can be adjusted here)
            scalar_patches_to_mask = 0  # Change this value to mask more scalar patches
            
            # Count scalar patches across all timestamps
            scalar_mask = (patch_types == 2)
            num_scalar_patches = scalar_mask.sum().item()
            
            # Calculate how many patches to mask
            total_patches_to_mask = int(round(L * self.mask_ratio, 0))
            
            # Ensure we have enough scalar patches and don't exceed total patches
            if num_scalar_patches == 0:
                raise ValueError("No scalar patches found but n_scalars > 0")
            
            # Clamp scalar patches to mask to available amount and total mask budget
            scalar_patches_to_mask = min(scalar_patches_to_mask, num_scalar_patches, total_patches_to_mask)
            non_scalar_patches_to_mask = max(0, total_patches_to_mask - scalar_patches_to_mask)
            
            if self.const_mask:
                noise = self.noise.expand(B, -1)
            else:
                noise = torch.rand(B, L, device=x.device)

            # Create a modified noise that masks exactly the specified number of scalar patches
            modified_noise = noise.clone()

            # Get indices of scalar patches
            scalar_indices = torch.where(scalar_mask)[0]

            if scalar_patches_to_mask > 0:
                # Randomly select scalar patches to mask
                selected_scalar_indices = np.random.choice(
                    scalar_indices.cpu().numpy(), 
                    scalar_patches_to_mask, 
                    replace=False
                )

                # Set the noise values for all scalar patches to be very low initially
                modified_noise[:, scalar_mask] = -torch.inf
                # Set the noise values for the selected scalar patches to be very high
                modified_noise[:, selected_scalar_indices] = torch.inf
            else:
                # If 0 scalar patches to mask, ensure scalar patches are never masked
                # by setting their noise values to be very low (high priority to keep)
                modified_noise[:, scalar_mask] = -torch.inf

            # Use the TOTAL patches to mask
            len_keep = L - total_patches_to_mask
            ids_shuffle = torch.argsort(modified_noise, dim=1)
        
        # Rest of the function remains the same
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # build binary mask (0 keep, 1 mask) and restore original order
        mask = torch.ones(B, L, device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        # Create a class parameter to store the constant mask
        if self.const_mask:
            self.mask = mask

        return x_keep, mask, ids_restore, ids_keep
    
    def forward_encoder(self, x_patches, patch_types=None):
        """
        Forward pass through the encoder.
        Args:
            x_patches:      [B, L, D] - Patch embeddings
        Returns:
            tuple: A tuple containing:
                encoded:        [B, L, d_model] - Encoded representations
                mask:           [B, L] - Binary mask indicating which patches were kept
                ids_restore:    [B, L] - Indices to restore original order
        """

        # Randomly mask patches
        x_keep, mask, ids_restore, ids_keep = self.random_masking(x_patches, patch_types=patch_types)

        # encoder forward pass
        encoded = self.encoder(x_keep)
        # Normalize the output
        encoded = self.norm(encoded)

        return encoded, mask, ids_restore, ids_keep

    def forward_decoder(self, encoded, ids_restore, ids_keep):
        """
        Forward pass through the decoder.
        Args:
            encoded:        [B, L, d_model] - Encoded representations from the encoder
            ids_restore:    [B, L] - Indices to restore original order
        Returns:
            decoded:        [B, L, d_model] - Decoded representations
        """
       
       
    #    # Gather the dimensions to match the original order
    #     B, L_keep, D = encoded.shape
    #     L_total = self.total_patches
    #     L_mask = L_total - L_keep

    #     # Create a tensor to hold all tokens in correct positions
    #     x = torch.zeros(B, L_total, D, device=encoded.device, dtype=encoded.dtype)

    #     # Place encoded tokens in their correct positions
    #     # Assuming you have ids_keep from the encoding phase
    #     x.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D), encoded)

    #     # Fill remaining positions with mask tokens
    #     mask_positions = torch.ones(B, L_total, device=encoded.device, dtype=torch.bool)
    #     mask_positions.scatter_(1, ids_keep, False)
    #     x[mask_positions] = self.mask_token.expand(B, L_mask, D).reshape(-1, D)

    #     # Add positional encoding (tokens now in correct spatial/temporal positions)
    #     x = self.positional_encoding(x)
       
       
        # Gather the dimensions to match the original order
        B, L_keep, D = encoded.shape
        L_total = self.total_patches
        L_mask = L_total - L_keep

        # Create mask tokens
        mask_tokens = self.mask_token.expand(B, L_mask, D)  # [B, L_mask, d_model]

        # Concatenate encoded and mask tokens
        x = torch.cat([encoded, mask_tokens], dim=1)  # [B, L_total, d_model]

        # Reorder the concatenated tensor to match the original order
        ids_restore_exp = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x = torch.gather(x, 1, ids_restore_exp)

        # Add positional encoding 
        x = self.positional_encoding(x)

        # Decoder forward pass
        decoded = self.decoder(x) # [B, L_total, d_model]

        # Normalize the output
        decoded = self.norm(decoded) # [B, L_total, d_model]

        return decoded
    
    # Forward pass
    def forward(self, x_spectrogram, x_vectors=None, x_scalars=None, const_mask=False, evaluate=False):
        """
        Forward pass through the MAE Transformer.
        Args:
            x_spectrogram:  [B, T, input_dim] - Input spectrogram
            x_vectors:      [B, T, n_vectors, vector_dim] - Input vectors (optional)
            x_scalars:      [B, T, n_scalars] - Input scalars (optional)
        Returns:
            tuple: A tuple containing:
            pred_spect:     [B, T, input_dim] - Predicted spectrogram
            pred_vecs:      [B, T, n_vectors, vector_dim] - Predicted vectors
            pred_scalars:   [B, T, n_scalars] - Predicted scalars
            mask:           [B, L] - Binary mask indicating which patches were kept
        """
        # Check whether constant mask is used
        if const_mask:
            self.const_mask = True
            
        # 1. patchify & embed every modality
        patches, patch_types = self.patchify(x_spectrogram, x_vectors, x_scalars)

        if evaluate and const_mask:
            self.const_mask = True

            pred_spect_list = []
            pred_vecs_list = []
            pred_scalars_list = []
            mask_list = []
            mask_spect_list = []
            mask_vectors_list = []
            mask_scalars_list = []
            
            for i in range(len(self.noises)):
                self.noise = self.noises[i]

                # 2. run encoder (does random masking internally)
                encoded, mask, ids_restore, ids_keep = self.forward_encoder(patches, patch_types=patch_types)

                # 3. run decoder (re‑inject mask tokens)
                decoded = self.forward_decoder(encoded, ids_restore, ids_keep)

                # 4. unpatchify -> modality‑shaped outputs
                pred_spect, pred_vecs, pred_scalars, mask_spect, mask_vectors, mask_scalars = self.unpatchify(decoded, patch_types, mask)

                # Collect results
                pred_spect_list.append(pred_spect)
                pred_vecs_list.append(pred_vecs)
                pred_scalars_list.append(pred_scalars)
                mask_list.append(mask)
                mask_spect_list.append(mask_spect)
                mask_vectors_list.append(mask_vectors)
                mask_scalars_list.append(mask_scalars)
            
            return pred_spect_list, pred_vecs_list, pred_scalars_list, mask_list, mask_spect_list, mask_vectors_list, mask_scalars_list

        else:

            self.noise = self.noises[0]

            # 2. run encoder (does random masking internally)
            encoded, mask, ids_restore, ids_keep = self.forward_encoder(patches, patch_types=patch_types)

            # 3. run decoder (re‑inject mask tokens)
            decoded = self.forward_decoder(encoded, ids_restore, ids_keep)

            # 4. unpatchify -> modality‑shaped outputs
            pred_spect, pred_vecs, pred_scalars, mask_spect, mask_vectors, mask_scalars = self.unpatchify(decoded, patch_types, mask)

        # Reset constant mask after forward pass
        self.const_mask = False  

        # 5. Return the outputs:
        # loss = ( (pred - target)**2 * mask.unsqueeze(-1) ).sum() / mask.sum()
        return pred_spect, pred_vecs, pred_scalars, mask, mask_spect, mask_vectors, mask_scalars
    
    # Loss function
    def loss_function(self, 
                      pred_spect, 
                      target_spect, 
                      pred_vecs, 
                      target_vecs, 
                      pred_scalars, 
                      target_scalars, 
                      mask_spect, 
                      mask_vectors, 
                      mask_scalars,
                      ):
        """
        Computes the loss for the MAE Transformer.
        Args:
            pred_spect:     [B, T, input_dim] - Predicted spectrogram
            target_spect:   [B, T, input_dim] - Target spectrogram
            pred_vecs:      [B, T, n_vectors, vector_dim] - Predicted vectors
            target_vecs:    [B, T, n_vectors, vector_dim] - Target vectors
            pred_scalars:   [B, T, n_scalars] - Predicted scalars
            target_scalars: [B, T, n_scalars] - Target scalars
            mask_spect:     [B, T, input_dim] - Mask for spectrogram loss
            mask_vectors:   [B, T, n_vectors, vector_dim] - Mask for vector loss
            mask_scalars:   [B, T, n_scalars] - Mask for scalar loss
        Returns:
            loss: Computed loss value
        """        

        # Per-sample losses [B]
        B = pred_spect.shape[0]

        if mask_spect.sum() != 0:

            # Total loss 
            spect_loss = (pred_spect - target_spect) ** 2 * mask_spect
            spect_loss_total = spect_loss.sum() / mask_spect.sum()

            # Spectrogram loss per sample
            spect_loss_per_sample = (spect_loss.sum(dim=(1, 2)) / mask_spect.sum(dim=(1, 2)))  # [B]

            # Turn nan's into zeros
            spect_loss_per_sample = torch.nan_to_num(spect_loss_per_sample, nan=0)
        
        else:
            spect_loss_total = 0.0
            spect_loss_per_sample = torch.zeros(B, device=pred_spect.device)


        # Vector loss
        if self.n_vectors > 0 and mask_vectors.sum() != 0:
            vec_loss = (pred_vecs - target_vecs) ** 2 * mask_vectors
            vec_loss_total = vec_loss.sum() / mask_vectors.sum()

            # Vector loss per sample
            vec_loss_per_sample = vec_loss.sum(dim=(1, 2, 3)) / mask_vectors.sum(dim=(1, 2, 3))

             # Turn nan's into zeros
            vec_loss_per_sample = torch.nan_to_num(vec_loss_per_sample, nan=0)
        else:
            vec_loss_total = 0.0
            vec_loss_per_sample = torch.zeros(B, device=pred_spect.device)

        # Scalar loss
        if self.n_scalars > 0 and mask_scalars.sum() != 0:
            scalar_loss = (pred_scalars - target_scalars) ** 2 * mask_scalars
            scalar_loss_total = scalar_loss.sum() / mask_scalars.sum()

            # Scalar loss per sample
            scalar_loss_per_sample = (scalar_loss.sum(dim=(1, 2)) / mask_scalars.sum(dim=(1, 2)))
            # Turn nan's into zeros
            scalar_loss_per_sample = torch.nan_to_num(scalar_loss_per_sample, nan=0)
        
        else:
            scalar_loss_total = 0.0
            scalar_loss_per_sample = torch.zeros(B, device=pred_spect.device)

        # Total loss (original) - weighted by the number of patches of each type
        loss = (spect_loss_total * self.spect_patch_ratio) + \
               (vec_loss_total * self.vector_patch_ratio) + \
                (scalar_loss_total * self.scalar_patch_ratio)
                    
        # Total per-sample loss [B]
        loss_per_sample = (spect_loss_per_sample * self.spect_patch_ratio) + \
                          (vec_loss_per_sample * self.vector_patch_ratio) + \
                            (scalar_loss_per_sample * self.scalar_patch_ratio*1/2)

        return loss, loss_per_sample
    
    # Training loop
    def fit(self, train_data, test_data, epochs=100, batch_size=32, lr=0.001, device=None, train_fraction=1.0):
        """
        Training loop for the MAE Transformer.
        
        Args:
            train_data (dict): Dictionary containing training data with keys:
                - 'spectrograms': [N, T, input_dim] - Training spectrograms
                - 'vectors': [N, T, n_vectors, vector_dim] - Training vectors (optional)
                - 'scalars': [N, T, n_scalars] - Training scalars (optional)
            test_data (dict): Dictionary containing test data with same structure as train_data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            lr (float): Learning rate
            device (str, optional): Device to use ('cuda' or 'cpu'). If None, uses self.device
        
        Returns:
            dict: Dictionary containing training history with keys 'train_loss' and 'test_loss'
        """
        # Use provided device or fallback to self.device
        if device is None:
            device = self.device
        
        # Move model to device
        self.to(device)
        
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Extract data and move to device
        train_spect = train_data.get('spectrograms', None)
        train_vecs = train_data.get('vectors', None)
        train_scalars = train_data.get('scalars', None)
        
        test_spect = test_data.get('spectrograms', None)
        test_vecs = test_data.get('vectors', None)
        test_scalars = test_data.get('scalars', None)
        
        # Move data to device
        if train_spect is not None:
            train_spect = train_spect.to(device)
        if train_vecs is not None:
            train_vecs = train_vecs.to(device)
        if train_scalars is not None:
            train_scalars = train_scalars.to(device)
        if test_spect is not None:
            test_spect = test_spect.to(device)
        if test_vecs is not None:
            test_vecs = test_vecs.to(device)
        if test_scalars is not None:
            test_scalars = test_scalars.to(device)
        
        # Get dataset size
        train_size = train_spect.shape[0] if train_spect is not None else 0
        test_size = test_spect.shape[0] if test_spect is not None else 0
        
        # Training history
        train_losses = []
        test_losses = []
        
        print(f"Starting training for {epochs} epochs on device: {device}")
        print(f"Train size: {train_size}, Test size: {test_size}, Batch size: {batch_size}")
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss_epoch = 0.0
            num_train_batches = 0

            # Randomly select a fraction of the training data if specified
            if train_fraction < 1.0:
                perm = torch.randperm(train_size)
                selected_size = int(train_size * train_fraction)
                selected_indices = perm[:selected_size]
                
                train_spect_use = train_spect[selected_indices] if train_spect is not None else None
                train_vecs_use = train_vecs[selected_indices] if train_vecs is not None else None
                train_scalars_use = train_scalars[selected_indices] if train_scalars is not None else None
                
                train_size_use = selected_size
            
            else:
                train_spect_use = train_spect
                train_vecs_use = train_vecs
                train_scalars_use = train_scalars
                train_size_use = train_size

            # Create batches
            for i in range(0, train_size_use, batch_size):
                end_idx = min(i + batch_size, train_size)
                
                # Extract batch data
                batch_spect = train_spect_use[i:end_idx] if train_spect is not None else None
                batch_vecs = train_vecs_use[i:end_idx] if train_vecs is not None else None
                batch_scalars = train_scalars_use[i:end_idx] if train_scalars is not None else None
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward passs
                pred_spect, pred_vecs, pred_scalars, _, mask_spect, mask_vectors, mask_scalars = self.forward(
                    batch_spect, batch_vecs, batch_scalars, const_mask=False
                )
                
                # Compute loss
                loss, loss_per_sample = self.loss_function(
                    pred_spect, batch_spect,
                    pred_vecs, batch_vecs,
                    pred_scalars, batch_scalars, 
                    mask_spect, mask_vectors, 
                    mask_scalars
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss_epoch += loss.item()
                num_train_batches += 1

            avg_train_loss = train_loss_epoch / num_train_batches
            train_losses.append(avg_train_loss)

            # For final epoch, calculate the average reconstruction error of the model
            if epoch == epochs - 1:

                # Forward pass on the entire training set to get reconstruction errors
                self.eval()
                loss_all_samples = torch.zeros(train_size, device=device)
                loss_all_batches = 0.0
                num_train_batches = 0
                with torch.no_grad():
                    for i in range(0, train_size, batch_size):
                        end_idx = min(i + batch_size, train_size)
                        
                        # Extract batch data
                        batch_spect = train_spect[i:end_idx] if train_spect is not None else None
                        batch_vecs = train_vecs[i:end_idx] if train_vecs is not None else None
                        batch_scalars = train_scalars[i:end_idx] if train_scalars is not None else None
                        
                        # Forward pass
                        pred_spect, pred_vecs, pred_scalars, _, mask_spect, mask_vectors, mask_scalars= self.forward(
                            batch_spect, batch_vecs, batch_scalars, const_mask=True, evaluate=True
                        )
                        
                        # Compute loss for each prediction in the lists
                        total_loss = 0.0
                        total_loss_per_sample = torch.zeros(batch_spect.shape[0] if batch_spect is not None else 
                                                           batch_vecs.shape[0] if batch_vecs is not None else 
                                                           batch_scalars.shape[0], device=device)
                        
                        for j in range(len(pred_spect)):
                            loss_j, loss_per_sample_j = self.loss_function(
                                pred_spect[j], batch_spect,
                                pred_vecs[j] if pred_vecs[j] is not None else None, batch_vecs,
                                pred_scalars[j] if pred_scalars[j] is not None else None, batch_scalars,
                                mask_spect[j], mask_vectors[j], 
                                mask_scalars[j]
                            )
                            total_loss += loss_j
                            total_loss_per_sample += loss_per_sample_j
                        
                        # Average the losses
                        loss = total_loss / len(pred_spect)
                        loss_per_sample = total_loss_per_sample / len(pred_spect)
                        
                        loss_all_samples[i:end_idx] = loss_per_sample
                        loss_all_batches += loss.item()
                        num_train_batches += 1


                
                self.avg_recon_error = loss_all_samples.mean().item()
                self.std_recon_error = loss_all_samples.std().item()
                print(f"Average reconstruction error on training set: {self.avg_recon_error:.6f}")
                print(f"Standard deviation of reconstruction error on training set: {self.std_recon_error:.6f}")
                print(f"Average batch loss on training set: {loss_all_batches / num_train_batches:.6f}")
                self.const_mask = False
            
            # Evaluation phase
            self.eval()
            test_loss_epoch = 0.0
            num_test_batches = 0
            
            with torch.no_grad():
                for i in range(0, test_size, batch_size):
                    end_idx = min(i + batch_size, test_size)
                    
                    # Extract batch data
                    batch_spect = test_spect[i:end_idx] if test_spect is not None else None
                    batch_vecs = test_vecs[i:end_idx] if test_vecs is not None else None
                    batch_scalars = test_scalars[i:end_idx] if test_scalars is not None else None
                    
                    # Forward pass
                    pred_spect, pred_vecs, pred_scalars, _, mask_spect, mask_vectors, mask_scalars= self.forward(
                        batch_spect, batch_vecs, batch_scalars
                    )
                    
                    # Compute loss
                    loss, _ = self.loss_function(
                        pred_spect, batch_spect,
                        pred_vecs, batch_vecs,
                        pred_scalars, batch_scalars,
                        mask_spect, mask_vectors, 
                        mask_scalars
                    )
                    
                    test_loss_epoch += loss.item()
                    num_test_batches += 1
            
            avg_test_loss = test_loss_epoch / num_test_batches
            test_losses.append(avg_test_loss)
            
            # Print progress
            if (epoch + 1) % 1 == 0 or epoch == 0:
                
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        print("Training completed!")
        
        return train_losses, test_losses
        
    # Data Preparation 
    def calculate_mean_std(self, data, data_types=['ion_eflux', 'magnetic_field_gsm', 'ion_velocity_gsm', 'ion_avgtemp'],  window_average=1):

        self.means = {}
        self.stds = {}   

        # Loop through each data type
        for data_type in data_types:

            # Get data and shape of data, to determine whether it is spectrogram, vector, or scalar
            data_shape = data[list(data.keys())[0]][data_type].shape
            
            # Initialize full datatype list, to concatenate all sections later
            full_data = []

            # Loop through each section in the data, to gather all sections of the same data type, and window average 
            for section in data.keys():
                
                # Extract section data
                section_data = data[section][data_type].to(self.device)  # Move to device

                # If set to window average, apply window averaging to the data
                if window_average > 1:
                    section_data = average_window(section_data, window_size=window_average)

                # Append the spectrogram data to the full_data_spectrogram list
                full_data.append(section_data) if data_type in data[section] else None

            # Concatenate all spectrograms into a single tensor
            full_data = torch.cat(full_data, dim=0)

            # If the data is a spectrogram, take the logarithm of the data
            if len(data_shape) == 2 and data_shape[1] > 3:  # Spectrogram data
                full_data = torch.log(full_data + 1e-10)
                mean = full_data.mean().to(self.device)
                std = full_data.std().to(self.device)

            # Calculate the mean and standard deviation for normalization, and store in the model for later use
            elif len(data_shape) == 2 and data_shape[1] == 3:  # Vector data (per direction normalisation)
                
                mean = full_data.mean(dim=0, keepdim=True).to(self.device)
                std = full_data.std(dim=0, keepdim=True).to(self.device)

            else:  # Scalar data
                # Take the logarithm of the data if it is part of certain scalars
                if data_type in ['ion_avgtemp', 'ion_density', 'plasma_beta', 'ion_vthermal', 'magnetic_field_gsm_magnitude', 'magnetic_field_gsm_x', 'electron_avgtemp', 'electron_density', 'electron_vthermal']:
                    # Logarithm for these scalars
                    section_data = remove_outliers_with_local_interpolation(section_data, 2, window_size=20)
                
                if data_type in ['ion_avgtemp', 'ion_density', 'plasma_beta', 'ion_vthermal', 'electron_avgtemp', 'electron_density', 'electron_vthermal']:
                    section_data = torch.log(section_data + 1e-10)


                mean = full_data.mean().to(self.device)
                std = full_data.std().to(self.device)

            self.means[data_type] = mean
            self.stds[data_type] = std 

    def prepare_data(self, 
                     data, 
                     data_types=['ion_eflux', 'magnetic_field_gsm', 'ion_velocity_gsm', 'ion_avgtemp'], 
                     test_fraction=0.15, 
                     rand_perm=True, 
                     calc_mean_std=True, 
                     timestamps=3, 
                     window_average=1
                     ):
        """
        Prepares the data for training and testing.
        
        Args:
            data (dict): Dictionary containing the data for each modality, per section.

                1. Spectrograms Shape: [X, D] || X = section length , D = energy bins
                2. Vectors Shape: [X, vector_dim]
                3. Scalars Shape: [X]
            data_types (list): List of data types to include in the preparation.
            test_fraction (float): Fraction of data to use for testing.
            rand_perm (bool): Whether to randomly permute the data.
            normalize (bool): Whether to normalize the data.

        Returns:
            tuple: A tuple containing:
                - train_data (dict): Dictionary containing the training data for each modality.
                    - Spectrograms Shape: [L, D] || L = Section Length, D = energy bins
                    - Vectors Shape: [L, n_vectors, vector_dim] (None if n_vectors=0)
                    - Scalars Shape: [L, n_scalars] (None if n_scalars=0)

                - test_data (dict): Dictionary containing the testing data for each modality.
                    - Spectrograms Shape: [X, T, D] || X = Total Length, T = Timestamps, D = energy bins
                    - Vectors Shape: [X, T, n_vectors, vector_dim] (None if n_vectors=0)
                    - Scalars Shape: [X, T, n_scalars] (None if n_scalars=0)

        """
        # Extract and prepare the data
        train_data = {}
        test_data = {}

        # 1. Normalization variables storage loop, store means and stds for each data type
        if calc_mean_std:

            self.means = {}
            self.stds = {}
            self.calculate_mean_std(
                data, 
                data_types, 
                window_average=window_average, 
            )

        # Create empty dict that will hold all data
        all_data = {}

        # 2. Normalization and data preparation loop
        for data_type in data_types:


            # Initialize full datatype list, to concatenate all sections later
            full_data = []
            
            # Loop through each section in the data, to gather all sections of the same data type, and window average 
            for section in data.keys():
                
                variable_type = None

                # Get section data 
                section_data = data[section][data_type].to(self.device)  # Move to device

                # Window average the data if specified
                if window_average > 1:
                    # If set to window average, apply window averaging to the data
                    section_data = average_window(section_data, window_size=window_average)

                # Take the logarithm of the data if it is a spectrogram
                if len(section_data.shape) == 2 and section_data.shape[1] > 3:  # Spectrogram data
                    # If set to window average, apply window averaging to the data
                    section_data = torch.log(section_data + 1e-10)   

                # Take the logarithm of the data if it is part of certain scalars
                if data_type in ['ion_velocity', 'ion_avgtemp', 'ion_density',  'ion_vthermal', 'electron_avgtemp', 'electron_density', 'electron_vthermal', 'plasma_beta', 'magnetic_field_gsm_magnitude', 'magnetic_field_gsm_x']:
                    # Logarithm for these scalars
                    section_data = remove_outliers_with_local_interpolation(section_data, 2, window_size=20)

                
                if data_type in ['ion_avgtemp', 'ion_density', 'plasma_beta', 'ion_vthermal', 'electron_avgtemp', 'electron_density', 'electron_vthermal']:
                    section_data = torch.log(section_data + 1e-10)

                # Normalize the data if normalization variables are stored
                if data_type in self.means:
                    mean = self.means[data_type].to(self.device)
                    std = self.stds[data_type].to(self.device)
                    section_data = (section_data - mean) / (std + 1e-10)

                # Create samples from the section data by grouping several timestamps together
                if len(section_data.shape) == 1 and len(section_data) > timestamps:  # Scalar data [L]

                    variable_type = 'scalar'
                    
                    # Reshape to [samples, timestamps] where samples = L - timestamps + 1
                    samples = []
                    for i in range(len(section_data) - timestamps + 1):
                        sample = section_data[i:i + timestamps]  # [timestamps]
                        samples.append(sample)
                    section_data = torch.stack(samples, dim=0)  # [samples, timestamps]
                    
                elif len(section_data.shape) == 2 and len(section_data) > timestamps:  # Spectrogram [L, D] or Vector [L, vector_dim]

                    variable_type = 'spectrogram' if section_data.shape[1] > 3 else 'vector'

                    # Reshape to [samples, timestamps, D] or [samples, timestamps, vector_dim]
                    samples = []
                    for i in range(len(section_data) - timestamps + 1):
                        sample = section_data[i:i + timestamps]  # [timestamps, D] or [timestamps, vector_dim]
                        samples.append(sample)
                    section_data = torch.stack(samples, dim=0)  # [samples, timestamps, D] or [samples, timestamps, vector_dim]
                
                # Append the spectrogram data to the full_data_spectrogram list
                full_data.append(section_data) if data_type in data[section] else None

            # Remove all sets of samples that are not 3 dimensions in full_data
            if variable_type == 'spectrogram' or variable_type == 'vector':
                full_data = [fd for fd in full_data if len(fd.shape) == 3 or (len(fd.shape) == 2 and fd.shape[1] == timestamps)]

            else:
                full_data = [fd for fd in full_data if len(fd.shape) == 2 or (len(fd.shape) == 1 and fd.shape[0] == timestamps)]

            # Concatenate all sections into a single tensor
            full_data = torch.cat(full_data, dim=0)

            # Randomly permute the data if specified
            if rand_perm:   
                perm = torch.randperm(full_data.shape[0])
                full_data = full_data[perm]

            # Store the full data in the all_data dict
            all_data[data_type] = full_data

        # 3. Split the data into training and testing sets
        total_samples = all_data[data_types[0]].shape[0]  # Get the number of samples from the first data type
        test_size = int(total_samples * test_fraction)
        train_size = total_samples - test_size

        # Separate data types into categories
        spectrograms = []
        vectors = []
        scalars = []
        
        for data_type in data_types:
            data_shape = all_data[data_type].shape
            
            if len(data_shape) == 3 and data_shape[2] > 3:  # Spectrogram data [samples, timestamps, D]
                spectrograms.append(all_data[data_type])
            elif len(data_shape) == 3 and data_shape[2] <= 3:  # Vector data [samples, timestamps, vector_dim]
                vectors.append(all_data[data_type])
            elif len(data_shape) == 2:  # Scalar data [samples, timestamps]
                scalars.append(all_data[data_type])
        
        # Combine data by category
        if spectrograms:
            # Assume only one spectrogram type for now
            spectrogram_data = spectrograms[0]
        else:
            spectrogram_data = None
            
        if vectors:
            # Stack vectors along a new dimension: [samples, timestamps, n_vectors, vector_dim]
            vector_data = torch.stack(vectors, dim=2)
        else:
            vector_data = None
            
        if scalars:
            # Stack scalars along a new dimension: [samples, timestamps, n_scalars]
            scalar_data = torch.stack(scalars, dim=2)
        else:
            scalar_data = None
        
        # Split into train and test sets
        if spectrogram_data is not None:
            train_data['spectrograms'] = spectrogram_data[:train_size]
            test_data['spectrograms'] = spectrogram_data[train_size:]

        else:
            train_data['spectrograms'] = None
            test_data['spectrograms'] = None
        
        if vector_data is not None:
            train_data['vectors'] = vector_data[:train_size]
            test_data['vectors'] = vector_data[train_size:]

        else:
            train_data['vectors'] = None
            test_data['vectors'] = None

        if scalar_data is not None:
            train_data['scalars'] = scalar_data[:train_size]
            test_data['scalars'] = scalar_data[train_size:]

        else:
            train_data['scalars'] = None
            test_data['scalars'] = None

        return train_data, test_data
    
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
    
class PositionalEncoding2D(nn.Module):
    """
    Sinusoidal 2D positional encoding (y ⊕ x) added to the last dimension.
    Supports inputs shaped [B, H, W, C] or [B, C, H, W] or [B, L, C] (with H,W).
    """
    def __init__(self, d_model, max_height=100, max_width=100):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 (sin/cos for y and x)."
        self.d_model = d_model
        self.max_height = max_height
        self.max_width = max_width

        d_half = d_model // 2          # per-axis channels (y + x)
        d_quarter = d_half // 2        # sin/cos pairs per axis

        # Precompute y-axis encodings: [H, d_half]
        pe_h = torch.zeros(max_height, d_half)
        pos_h = torch.arange(max_height, dtype=torch.float).unsqueeze(1)  # [H,1]
        div_h = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        pe_h[:, 0::2] = torch.sin(pos_h * div_h)
        pe_h[:, 1::2] = torch.cos(pos_h * div_h)

        # Precompute x-axis encodings: [W, d_half]
        pe_w = torch.zeros(max_width, d_half)
        pos_w = torch.arange(max_width, dtype=torch.float).unsqueeze(1)   # [W,1]
        div_w = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        pe_w[:, 0::2] = torch.sin(pos_w * div_w)
        pe_w[:, 1::2] = torch.cos(pos_w * div_w)

        self.register_buffer("pe_h", pe_h)  # moves with .to()
        self.register_buffer("pe_w", pe_w)

    def _grid(self, H, W, dtype, device):
        # Safety: grow on-the-fly if needed
        if H > self.pe_h.size(0) or W > self.pe_w.size(0):
            # Generate just-in-time encodings for the requested size
            d_half = self.d_model // 2
            def build_axis(L):
                pe = torch.zeros(L, d_half, dtype=torch.float32, device=device)
                pos = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)
                div = torch.exp(torch.arange(0, d_half, 2, device=device).float()
                                * (-math.log(10000.0) / d_half))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                return pe
            pe_h = build_axis(H)
            pe_w = build_axis(W)
        else:
            pe_h = self.pe_h[:H].to(device=device, dtype=torch.float32)
            pe_w = self.pe_w[:W].to(device=device, dtype=torch.float32)

        # [H, W, d_model]: concat y and x encodings
        pos_h = pe_h.unsqueeze(1).expand(H, W, -1)  # [H, W, d_half]
        pos_w = pe_w.unsqueeze(0).expand(H, W, -1)  # [H, W, d_half]
        pe = torch.cat([pos_h, pos_w], dim=-1)      # [H, W, d_model]
        return pe.to(dtype=dtype, device=device)

    def forward(self, x, hw=None, channels_last=None):
        """
        x: [B, H, W, C] or [B, C, H, W] or [B, L, C]
        hw: (H, W) if x is flattened [B, L, C]
        channels_last: force interpretation if ambiguous
        """
        if x.dim() == 4:
            # Disambiguate layout
            if channels_last is None:
                # Heuristic: if last dim equals d_model -> channels_last
                channels_last = (x.size(-1) == self.d_model)

            if channels_last:  # [B, H, W, C]
                B, H, W, C = x.shape
                assert C == self.d_model, "Last dim must equal d_model."
                pe = self._grid(H, W, dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1, -1)
                return x + pe
            else:              # [B, C, H, W]
                B, C, H, W = x.shape
                assert C == self.d_model, "Channel dim must equal d_model."
                pe = self._grid(H, W, dtype=x.dtype, device=x.device)      # [H, W, C]
                pe = pe.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1) # [B, C, H, W]
                return x + pe

        elif x.dim() == 3:  # [B, L, C]
            B, L, C = x.shape
            assert C == self.d_model, "Last dim must equal d_model."
            if hw is None:
                # Try perfect square; otherwise raise a helpful error
                r = int(math.isqrt(L))
                if r * r != L:
                    raise ValueError("For flattened input [B, L, C], provide hw=(H, W) so H*W=L.")
                H, W = r, r
            else:
                H, W = hw
                assert H * W == L, f"Provided hw={hw} does not match L={L}."

            pe = self._grid(H, W, dtype=x.dtype, device=x.device).reshape(1, L, C).expand(B, -1, -1)
            return x + pe

        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")
        