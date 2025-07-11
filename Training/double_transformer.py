import torch
import torch.nn as nn
import torch.optim as optim
import math
from datetime import datetime, timezone
from MAE_transformer import MAETransformer, PositionalEncoding
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class DoubleTransformer(nn.Module):
    def __init__(self, input_dim=31, timestamps=4, d_model=64, nhead=4, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=512, mask_ratio=0.50, patches_per_timestamp=1):
        
        super().__init__()
        
        self.transformer_magnetotail = MAETransformer(
            input_dim=input_dim, 
            timestamps=timestamps, 
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            mask_ratio=mask_ratio,
            patches_per_timestamp=patches_per_timestamp
        )

        self.transformer_magnetosheath = MAETransformer(
            input_dim=input_dim, 
            timestamps=timestamps, 
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            mask_ratio=mask_ratio,
            patches_per_timestamp=patches_per_timestamp
        )

    def prepare_data(self, data, data_type='B_ion_eflux', test_fraction=0.15):
         # Split data into magnetotail and magnetosheath regions
        magnetotail_data = dict()
        magnetosheath_data = dict()

        # Loop through the data and separate it based on region
        for session in data.keys():
            if data[session]['region'] == 'magnetotail':
                magnetotail_data[session] = data[session]
            elif data[session]['region'] == 'magnetosheath':
                magnetosheath_data[session] = data[session]

        # Both models calculate the mean and std of the full dataset, to get constistent normalization
        self.transformer_magnetotail.calculate_mean_std(data)
        self.transformer_magnetosheath.calculate_mean_std(data)

        # Prepare data for training
        train_data_tail, test_data_tail = \
            self.transformer_magnetotail.prepare_data(
                magnetotail_data, 
                data_type=data_type,
                test_fraction=test_fraction,
                norm_mode='old', 
                storage_mode='combined'
                )
        
        train_data_sheath, test_data_sheath = \
            self.transformer_magnetosheath.prepare_data(
                magnetosheath_data, 
                data_type=data_type,
                test_fraction=test_fraction,
                norm_mode='old', 
                storage_mode='combined'
                )

        return train_data_tail, test_data_tail, train_data_sheath, test_data_sheath

    def fit(self, data, epochs=10, batch_size=32, learning_rate=0.001):
        
        # Prepare data for both models
        train_data_tail, test_data_tail, train_data_sheath, test_data_sheath = self.prepare_data(data)

        if isinstance(epochs, tuple):
            # If epochs is a tuple, unpack it
            epochs_tail, epochs_sheath = epochs
        else:
            # If epochs is a single value, use it for both models
            epochs_tail = epochs_sheath = epochs
            

        # Fit both transformers
        _, train_loss_tail, test_loss_tail = self.transformer_magnetotail.fit(train_data_tail, 
                                         test_data_tail,
                                         epochs=epochs_tail, 
                                         batch_size=batch_size, 
                                         lr=learning_rate)
        
        _, train_loss_sheath, test_loss_sheath  = self.transformer_magnetosheath.fit(train_data_sheath, 
                                           test_data_sheath, 
                                           epochs=epochs_sheath, 
                                           batch_size=batch_size, 
                                           lr=learning_rate)
        
        return train_loss_tail, test_loss_tail, train_loss_sheath, test_loss_sheath

    def predict(self, data, data_type='ion_eflux'):
        # Use one of the models to prepare the data
        # This ensures consistent data preparation (both models have same normalization parameters)
        prepared_data, _ = self.transformer_magnetosheath.prepare_data(data, 
                                                                        data_type=data_type,
                                                                        test_fraction=0,
                                                                        norm_mode='old', 
                                                                        storage_mode='combined',
                                                                        rand_perm=False)
        

        
        # Set models to evaluation mode for inference
        self.transformer_magnetotail.eval()
        self.transformer_magnetosheath.eval()
        
        # Predict using both models without gradient computation
        with torch.no_grad():
            _, results_tail_model, _ = self.transformer_magnetotail(prepared_data)
            _, results_sheath_model, _ = self.transformer_magnetosheath(prepared_data)
        
        return results_tail_model, results_sheath_model, prepared_data
    
    def plot_results(self, data, times=None, figsize=(15, 10)):
        if times is not None:
            # Ensure times is a tensor
            if isinstance(times, torch.Tensor):
                # Time data
                
                time_ns = times.cpu().numpy().astype('int64')
                time_s = time_ns / 1e9
                times_datetime = [datetime.fromtimestamp(ts, timezone.utc) for ts in time_s]


        results_tail_model, results_sheath_model, prepared_data = self.predict(data)

        # Calculate reconstruction error (mean squared error)
        reconstruction_error_tail_model = torch.mean((prepared_data - results_tail_model) ** 2, dim=(1, 2))
        reconstruction_error_sheath_model = torch.mean((prepared_data - results_sheath_model) ** 2, dim=(1, 2))

        max_error_tail = torch.max(reconstruction_error_tail_model)
        max_error_sheath = torch.max(reconstruction_error_sheath_model)

        min_error_tail = torch.min(reconstruction_error_tail_model)
        min_error_sheath = torch.min(reconstruction_error_sheath_model)

        # Make a windowing average of the reconstruction errors with a window size of 10
        window_size = 1
        
        # Apply windowing average for tail model
        windowed_tail = torch.nn.functional.avg_pool1d(reconstruction_error_tail_model.unsqueeze(0).unsqueeze(0), 
                    kernel_size=window_size, 
                    stride=1).squeeze()
        
        # Pad to obtain times shape
        pad_size = (len(times_datetime) - windowed_tail.shape[0]) // 2
        if len(times_datetime) - windowed_tail.shape[0] % 2 != 0:
            # If the difference is odd, we need to pad one less element
            times_datetime = times_datetime[pad_size:len(times_datetime) - pad_size - 1]
        else:
            times_datetime = times_datetime[pad_size:len(times_datetime) - pad_size]

        

        
        # Apply windowing average for sheath model
        windowed_sheath = torch.nn.functional.avg_pool1d(reconstruction_error_sheath_model.unsqueeze(0).unsqueeze(0), 
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

        # Calculate the difference between the two models' reconstruction errors
        reconstruction_error_diff_relative = reconstruction_error_tail_model/(max_error_tail - min_error_tail) - reconstruction_error_sheath_model/(max_error_sheath - min_error_sheath)
        reconstruction_error_diff = reconstruction_error_tail_model - reconstruction_error_sheath_model

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        

        axs[0].plot(mdates.date2num(times_datetime), reconstruction_error_tail_model.cpu().numpy())
        axs[0].set_title('Reconstruction Error - Magnetotail Model')
        axs[0].set_xlabel('Sample Index')
        axs[0].set_ylabel('Reconstruction Error')
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))
        axs[0].grid()


        axs[1].plot(mdates.date2num(times_datetime), reconstruction_error_sheath_model.cpu().numpy())
        axs[1].set_title('Reconstruction Error - Magnetosheath Model')
        axs[1].set_xlabel('Sample Index')
        axs[1].set_ylabel('Reconstruction Error')
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))
        axs[1].grid()
        fig.autofmt_xdate()
        # axs[1, 0].plot(avg_reconstruction_error.cpu().numpy())
        # axs[1, 0].set_title('Average Reconstruction Error of Both Models')
        # axs[1, 0].set_xlabel('Sample Index')
        # axs[1, 0].set_ylabel('Average Reconstruction Error')

        # axs[1, 1].plot(min_reconstruction_error.values.cpu().numpy())
        # axs[1, 1].set_title('Minimum Reconstruction Error at Each Sample Index')
        # axs[1, 1].set_xlabel('Sample Index')
        # axs[1, 1].set_ylabel('Minimum Reconstruction Error')

        # axs[2].plot(reconstruction_error_diff_relative.cpu().numpy())
        # axs[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        # axs[2].set_title('Relative Difference between Models Reconstruction Errors')
        # axs[2].set_xlabel('Sample Index')
        # axs[2].set_ylabel('Relative Reconstruction Error Difference')

        # axs[1, 2].plot(reconstruction_error_diff.cpu().numpy())
        # axs[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        # axs[1, 2].set_title('Difference between Models Reconstruction Errors')
        # axs[1, 2].set_xlabel('Sample Index')
        # axs[1, 2].set_ylabel('Reconstruction Error Difference')

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if times is not None:
            ax2.plot(mdates.date2num(times_datetime), reconstruction_error_diff_relative.cpu().numpy(), label='Relative Difference')    
        else:
            ax2.plot(reconstruction_error_diff_relative.cpu().numpy(), label='Relative Difference')

        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))
        fig2.autofmt_xdate()
        plt.title('Relative Difference Between Model Reconstruction Errors')
        plt.xlabel('Sample Index')
        plt.ylabel('Relative Reconstruction Error Difference')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.grid()

        plt.tight_layout()
        plt.show()

        return reconstruction_error_tail_model, reconstruction_error_sheath_model, avg_reconstruction_error, min_reconstruction_error