import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from autoencoder import ConvAutoencoder


class DoubleAutoencoder(nn.Module):
    def __init__(self, hidden_layer_1=16, hidden_layer_2=32, hidden_layer_3=32, latent_dim=32, timestamps=5):
        super(DoubleAutoencoder, self).__init__()
        
        self.autoencoder_magnetotail = ConvAutoencoder(
            hidden_layer_1=hidden_layer_1,
            hidden_layer_2=hidden_layer_2,
            hidden_layer_3=hidden_layer_3,
            latent_dim=latent_dim,
            timestamps=timestamps
        )

        self.autoencoder_magnetosheath = ConvAutoencoder(
            hidden_layer_1=hidden_layer_1,
            hidden_layer_2=hidden_layer_2,
            hidden_layer_3=hidden_layer_3,
            latent_dim=latent_dim,
            timestamps=timestamps
        )

    def prepare_data(self, data, data_type='B_ion_eflux', test_fraction=0.15):
        magnetotail_data = dict()
        magnetosheath_data = dict()

        for session in data.keys():
            if data[session]['region'] == 'magnetotail':
                magnetotail_data[session] = data[session]
            elif data[session]['region'] == 'magnetosheath':
                magnetosheath_data[session] = data[session]

        self.autoencoder_magnetotail.calculate_mean_std(data)
        self.autoencoder_magnetosheath.calculate_mean_std(data)

        train_data_tail, test_data_tail = \
            self.autoencoder_magnetotail.prepare_data(
                magnetotail_data, 
                data_type=data_type,
                test_fraction=test_fraction,
                norm_mode='old', 
                storage_mode='combined'
            )
        
        train_data_sheath, test_data_sheath = \
            self.autoencoder_magnetosheath.prepare_data(
                magnetosheath_data, 
                data_type=data_type,
                test_fraction=test_fraction,
                norm_mode='old', 
                storage_mode='combined'
            )

        return train_data_tail, test_data_tail, train_data_sheath, test_data_sheath

    def fit(self, data, epochs=10, batch_size=32, learning_rate=0.001):
        train_data_tail, test_data_tail, train_data_sheath, test_data_sheath = self.prepare_data(data)

        self.autoencoder_magnetotail.fit(train_data_tail, 
                                        test_data_tail,
                                        epochs=epochs, 
                                        batch_size=batch_size, 
                                        lr=learning_rate)
        
        self.autoencoder_magnetosheath.fit(train_data_sheath, 
                                          test_data_sheath, 
                                          epochs=epochs, 
                                          batch_size=batch_size, 
                                          lr=learning_rate)

    def predict(self, data, data_type='B_ion_eflux'):
        prepared_data, _ = self.autoencoder_magnetosheath.prepare_data(data, 
                                                                      data_type=data_type,
                                                                      test_fraction=0,
                                                                      norm_mode='old', 
                                                                      storage_mode='combined',
                                                                      rand_perm=False)
        
        self.autoencoder_magnetotail.eval()
        self.autoencoder_magnetosheath.eval()
        
        with torch.no_grad():
            _, results_tail_model = self.autoencoder_magnetotail(prepared_data)
            _, results_sheath_model = self.autoencoder_magnetosheath(prepared_data)
        
        return results_tail_model, results_sheath_model, prepared_data
    
    def plot_results(self, data):
        results_tail_model, results_sheath_model, prepared_data = self.predict(data)

        # Calculate reconstruction error (mean squared error)
        reconstruction_error_tail_model = torch.mean((prepared_data - results_tail_model) ** 2, dim=(1, 2))
        reconstruction_error_sheath_model = torch.mean((prepared_data - results_sheath_model) ** 2, dim=(1, 2))

        max_error_tail = torch.max(reconstruction_error_tail_model)
        max_error_sheath = torch.max(reconstruction_error_sheath_model)

        # Make a windowing average of the reconstruction errors with a window size of 10
        window_size = 5
        
        # Apply windowing average with edge padding for tail model
        padded_tail = torch.nn.functional.pad(reconstruction_error_tail_model.unsqueeze(0), 
                              (window_size//2, window_size//2), 
                              mode='replicate').squeeze(0)
        windowed_tail = torch.nn.functional.avg_pool1d(padded_tail.unsqueeze(0).unsqueeze(0), 
                                kernel_size=window_size, 
                                stride=1).squeeze()
        
        # Apply windowing average with edge padding for sheath model
        padded_sheath = torch.nn.functional.pad(reconstruction_error_sheath_model.unsqueeze(0), 
                            (window_size//2, window_size//2), 
                            mode='replicate').squeeze(0)
        windowed_sheath = torch.nn.functional.avg_pool1d(padded_sheath.unsqueeze(0).unsqueeze(0), 
                                  kernel_size=window_size, 
                                  stride=1).squeeze()
        
        # Update the reconstruction errors to use windowed versions
        reconstruction_error_tail_model = windowed_tail
        reconstruction_error_sheath_model = windowed_sheath

                
        mean_reconstruction_error_tail = torch.mean(reconstruction_error_tail_model)
        mean_reconstruction_error_sheath = torch.mean(reconstruction_error_sheath_model)

        # Calculate the average of combined reconstruction errors
        avg_reconstruction_error = (reconstruction_error_tail_model/mean_reconstruction_error_tail +    \
                                    reconstruction_error_sheath_model/mean_reconstruction_error_sheath) \
                                    / 2

        # Find minimum of the reconstruction errors at each sample index
        min_reconstruction_error = torch.min(torch.stack((reconstruction_error_tail_model/mean_reconstruction_error_tail, 
                                                          reconstruction_error_sheath_model/mean_reconstruction_error_sheath)),
                                                          dim=0)

        # Calculate the delta of the reconstruction error values 
        reconstruction_error_tail_model_grad = torch.diff(reconstruction_error_tail_model, prepend=torch.tensor([0.0]).to(device=reconstruction_error_tail_model.device))
        
        # Calculate the difference between the two models' reconstruction errors
        reconstruction_error_diff = reconstruction_error_tail_model/max_error_tail - reconstruction_error_sheath_model/max_error_sheath

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].plot(reconstruction_error_tail_model.cpu().numpy())
        axs[0, 0].set_title('Reconstruction Error - Magnetotail Model')
        axs[0, 0].set_xlabel('Sample Index')
        axs[0, 0].set_ylabel('Reconstruction Error')

        axs[0, 1].plot(reconstruction_error_sheath_model.cpu().numpy())
        axs[0, 1].set_title('Reconstruction Error - Magnetosheath Model')
        axs[0, 1].set_xlabel('Sample Index')
        axs[0, 1].set_ylabel('Reconstruction Error')
         
        axs[1, 0].plot(avg_reconstruction_error.cpu().numpy())
        axs[1, 0].set_title('Average Reconstruction Error of Both Models')
        axs[1, 0].set_xlabel('Sample Index')
        axs[1, 0].set_ylabel('Average Reconstruction Error')

        axs[1, 1].plot(min_reconstruction_error.values.cpu().numpy())
        axs[1, 1].set_title('Minimum Reconstruction Error at Each Sample Index')
        axs[1, 1].set_xlabel('Sample Index')
        axs[1, 1].set_ylabel('Minimum Reconstruction Error')

        axs[0, 2].plot(reconstruction_error_tail_model_grad.cpu().numpy())
        axs[0, 2].set_title('Gradient of Reconstruction Error - Magnetotail Model')
        axs[0, 2].set_xlabel('Sample Index')
        axs[0, 2].set_ylabel('Gradient of Reconstruction Error')

        axs[1, 2].plot(reconstruction_error_diff.cpu().numpy())
        axs[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axs[1, 2].set_title('Difference between Models Reconstruction Errors')
        axs[1, 2].set_xlabel('Sample Index')
        axs[1, 2].set_ylabel('Reconstruction Error Difference')

        plt.tight_layout()
        plt.show()

        return reconstruction_error_tail_model, reconstruction_error_sheath_model, avg_reconstruction_error, min_reconstruction_error