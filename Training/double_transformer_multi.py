import os
import torch
import torch.nn as nn
from datetime import datetime, timezone
from MAE_transformer_multi import MAETransformerMulti
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import accuracy_score, classification_report


from Helper.crossoptim import match_crossings_hungarian, match_crossings_greedy
from Helper.preprocess_functions import average_window, remove_outliers_with_interpolation, remove_outliers_with_local_interpolation

class DoubleTransformerMulti(nn.Module):
    def __init__(self, input_dim=31, timestamps=4, d_model=64, nhead=4, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=512, mask_ratio=0.50, patches_per_timestamp_spect=1,
                 n_vectors=0, n_scalars=0, data_types=['ion_eflux'], device=None, window_average_data=3, data_sampling_rate=10, mask_scalars=False):
        
        """
        Initialize the DoubleTransformerMulti model with two MAETransformerMulti instances.
        One instance is for the magnetotail region and the other for the magnetosheath region.
        Args:
            input_dim (int): Number of input features of the spectrogram data. Normally 31.
            timestamps (int): Number of timestamps to consider in the input sequence. (Group n timestamps together)
            d_model (int or tuple): Dimension of the model. If tuple, first value is for magnetotail, second for magnetosheath.
            nhead (int or tuple): Number of attention heads. If tuple, first value is for magnetotail, second for magnetosheath.
            num_encoder_layers (int or tuple): Number of encoder layers. If tuple, first value is for magnetotail, second for magnetosheath.
            num_decoder_layers (int or tuple): Number of decoder layers. If tuple, first value is for magnetotail, second for magnetosheath.
            dim_feedforward (int or tuple): Dimension of the feedforward network. If tuple, first value is for magnetotail, second for magnetosheath.
            mask_ratio (float): Ratio of the input to mask during training.
            patches_per_timestamp_spect (int): Number of patches per timestamp in spectrogram data.
            n_vectors (int): Number of vector features in the input data.
            n_scalars (int): Number of scalar features in the input data.
            data_types (list): List of data types to be used
            device (torch.device or None): Device to run the model on. If None, uses CUDA if available.
            window_average_data (int): Number of data points to average over for smoothing.
            data_sampling_rate (int): Sampling rate of the data in minutes.
        """

        
        super().__init__()

        self.timestamps = timestamps


        ### ---INPUT PARSING--- ###
        if isinstance(d_model, tuple):
            d_model_tail = d_model[0]
            d_model_sheath = d_model[1]
    
        else:
            d_model_tail = d_model_sheath = d_model

        if isinstance(dim_feedforward, tuple):
            dim_feedforward_tail = dim_feedforward[0]
            dim_feedforward_sheath = dim_feedforward[1]
        
        else:
            dim_feedforward_tail = dim_feedforward_sheath = dim_feedforward

        if isinstance(nhead, tuple):
            nhead_tail = nhead[0]
            nhead_sheath = nhead[1]
        else:
            nhead_tail = nhead_sheath = nhead
            
        if isinstance(num_encoder_layers, tuple):
            num_encoder_layers_tail = num_encoder_layers[0]
            num_encoder_layers_sheath = num_encoder_layers[1]
        else:
            num_encoder_layers_tail = num_encoder_layers_sheath = num_encoder_layers

        if isinstance(num_decoder_layers, tuple):
            num_decoder_layers_tail = num_decoder_layers[0]
            num_decoder_layers_sheath = num_decoder_layers[1]   
        else:
            num_decoder_layers_tail = num_decoder_layers_sheath = num_decoder_layers
            

        # Store all parameters as instance variables
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.mask_ratio = mask_ratio
        self.patches_per_timestamp_spect = patches_per_timestamp_spect
        self.n_vectors = n_vectors
        self.n_scalars = n_scalars
        self.timestamps = timestamps
        
        self.transformer_magnetotail = MAETransformerMulti(
            input_dim=input_dim, 
            timestamps=timestamps, 
            d_model=d_model_tail, 
            nhead=nhead_tail, 
            num_encoder_layers=num_encoder_layers_tail, 
            num_decoder_layers=num_decoder_layers_tail, 
            dim_feedforward=dim_feedforward_tail, 
            mask_ratio=mask_ratio,
            patches_per_timestamp_spect=patches_per_timestamp_spect,
            n_vectors=n_vectors,
            n_scalars=n_scalars,
            device=device,  # Set device for the model
            mask_scalars=mask_scalars
        )

        self.noises = self.transformer_magnetotail.get_noise()

        self.transformer_magnetosheath = MAETransformerMulti(
            input_dim=input_dim, 
            timestamps=timestamps, 
            d_model=d_model_sheath, 
            nhead=nhead_sheath, 
            num_encoder_layers=num_encoder_layers_sheath, 
            num_decoder_layers=num_decoder_layers_sheath, 
            dim_feedforward=dim_feedforward_sheath, 
            mask_ratio=mask_ratio,
            patches_per_timestamp_spect=patches_per_timestamp_spect,
            n_vectors=n_vectors,
            n_scalars=n_scalars,
            noise=self.noises, # Use the same noise for both models to ensure consistency for the constant mask
            device=device,  # Set device for the model  
            mask_scalars=mask_scalars
        )

        self.data_types = data_types
        self.exclude_sections = None  # Initialize exclude_sections attribute
        self.window_average_data = window_average_data
        self.data_sampling_rate = data_sampling_rate  # in minutes
    
    def prepare_data(self, data, test_fraction=0.15):
         # Split data into magnetotail and magnetosheath regions
        magnetotail_data = {}
        magnetosheath_data = {}

        # Loop through the data and separate it based on region <=== STUPID NEED TO FIX (both models need to have same normalization parameters)
        for session in data.keys():

            # Skip excluded sections if specified
            if self.exclude_sections and session in self.exclude_sections:
                print(f"Excluding section {session} from training and testing data.")
                continue

            if data[session]['region'] == 'magnetotail':
                magnetotail_data[session] = data[session]
            elif data[session]['region'] == 'magnetosheath':
                magnetosheath_data[session] = data[session]

        # Also remove excluded sections from the main data for mean/std calculation
        if self.exclude_sections:
            data = {k: v for k, v in data.items() if k not in self.exclude_sections}


        # Calculate the mean and standard deviation for normalization on magnetotail data
        self.transformer_magnetotail.calculate_mean_std(magnetotail_data,
            data_types      = self.data_types, 
            window_average  = self.window_average_data, 
        )
        
        # Calculate the mean and standard deviation for normalization on magnetosheath data
        self.transformer_magnetosheath.calculate_mean_std(magnetosheath_data,
            data_types      = self.data_types, 
            window_average  = self.window_average_data, 
        )

        # Use models for data preparation 
        train_data_tail, test_data_tail = \
            self.transformer_magnetotail.prepare_data(
                magnetotail_data, 
                data_types=self.data_types,
                test_fraction=test_fraction,
                calc_mean_std=False,  # Use pre-calculated means and stds
                timestamps=self.timestamps,
                window_average=self.window_average_data
            )
        
        train_data_sheath, test_data_sheath = \
            self.transformer_magnetosheath.prepare_data(
                magnetosheath_data, 
                data_types=self.data_types,
                test_fraction=test_fraction,
                calc_mean_std=False,  # Use pre-calculated means and stds
                timestamps=self.timestamps,
                window_average=self.window_average_data
            )

        return train_data_tail, test_data_tail, train_data_sheath, test_data_sheath

    def fit(self, data, epochs=10, batch_size=32, learning_rate=0.001, test_fraction=0.15, exclude_sections=None, train_fraction=1.0):
        
        self.exclude_sections = exclude_sections

        # Prepare data for both models
        train_data_tail, test_data_tail, train_data_sheath, test_data_sheath = self.prepare_data(data, test_fraction=0.15)

        if isinstance(epochs, tuple):
            # If epochs is a tuple, unpack it
            epochs_tail, epochs_sheath = epochs
        else:
            # If epochs is a single value, use it for both models
            epochs_tail = epochs_sheath = epochs
            
        if isinstance(batch_size, tuple):
            # If batch_size is a tuple, unpack it
            batch_size_tail, batch_size_sheath = batch_size
        else:
            # If batch_size is a single value, use it for both models
            batch_size_tail = batch_size_sheath = batch_size

        if isinstance(learning_rate, tuple):
            # If learning_rate is a tuple, unpack it
            learning_rate_tail, learning_rate_sheath = learning_rate
        else:
            # If learning_rate is a single value, use it for both models
            learning_rate_tail = learning_rate_sheath = learning_rate

        if isinstance(train_fraction, tuple):
            # If train_fraction is a tuple, unpack it
            train_fraction_tail, train_fraction_sheath = train_fraction

        else:
            # If train_fraction is a single value, use it for both models
            train_fraction_tail = train_fraction_sheath = train_fraction

        # Fit both transformers
        train_loss_tail, test_loss_tail = self.transformer_magnetotail.fit(
            train_data_tail, 
            test_data_tail,
            epochs=epochs_tail, 
            batch_size=batch_size_tail, 
            lr=learning_rate_tail,
            train_fraction=train_fraction_tail
        )
        
        train_loss_sheath, test_loss_sheath  = self.transformer_magnetosheath.fit(
            train_data_sheath, 
            test_data_sheath, 
            epochs=epochs_sheath, 
            batch_size=batch_size_sheath, 
            lr=learning_rate_sheath,
            train_fraction=train_fraction_sheath
        )
        
        return train_loss_tail, test_loss_tail, train_loss_sheath, test_loss_sheath

    def fit_one_model(self, model_name, data, epochs=10, batch_size=32, learning_rate=0.001, test_fraction=0.15, exclude_sections=None):
        self.exclude_sections = exclude_sections

        # Prepare data for both models
        train_data_tail, test_data_tail, train_data_sheath, test_data_sheath = self.prepare_data(data, test_fraction=0.15)


        if model_name == 'magnetotail':
             # Create a new version of the model with the same parameters
            self.transformer_magnetotail = MAETransformerMulti(
                input_dim=self.input_dim, 
                timestamps=self.timestamps, 
                d_model=self.d_model, 
                nhead=self.nhead, 
                num_encoder_layers=self.num_encoder_layers, 
                num_decoder_layers=self.num_decoder_layers, 
                dim_feedforward=self.dim_feedforward, 
                mask_ratio=self.mask_ratio,
                patches_per_timestamp_spect=self.patches_per_timestamp_spect,
                n_vectors=self.n_vectors,
                n_scalars=self.n_scalars,
                device=self.transformer_magnetotail.device  # Set device for the model
            )


            return self.transformer_magnetotail.fit(train_data_tail, 
                                         test_data_tail,
                                         epochs=epochs, 
                                         batch_size=batch_size, 
                                         lr=learning_rate)
        elif model_name == 'magnetosheath':



            # Create a new version of the model with the same parameters
            self.transformer_magnetosheath = MAETransformerMulti(
                input_dim=self.input_dim, 
                timestamps=self.timestamps, 
                d_model=self.d_model[1] if isinstance(self.d_model, tuple) else self.d_model, 
                nhead=self.nhead[1] if isinstance(self.nhead, tuple) else self.nhead, 
                num_encoder_layers=self.num_encoder_layers[1] if isinstance(self.num_encoder_layers, tuple) else self.num_encoder_layers, 
                num_decoder_layers=self.num_decoder_layers[1] if isinstance(self.num_decoder_layers, tuple) else self.num_decoder_layers, 
                dim_feedforward=self.dim_feedforward[1] if isinstance(self.dim_feedforward, tuple) else self.dim_feedforward, 
                mask_ratio=self.mask_ratio,
                patches_per_timestamp_spect=self.patches_per_timestamp_spect,
                n_vectors=self.n_vectors,
                n_scalars=self.n_scalars,
                device=self.transformer_magnetosheath.device  # Set device for the model
            )


            return self.transformer_magnetosheath.fit(train_data_sheath, 
                                           test_data_sheath, 
                                           epochs=epochs, 
                                           batch_size=batch_size, 
                                           lr=learning_rate)
        else:
            raise ValueError("model_name must be either 'magnetotail' or 'magnetosheath'")
    
    def predict(self, data):
        # Data preparation for both models
        prepared_data_tail, _ = self.transformer_magnetotail.prepare_data(
            data, 
            data_types=self.data_types,
            test_fraction=0,
            calc_mean_std=False,  # Use pre-calculated means and stds
            rand_perm=False,
            timestamps=self.timestamps,
            window_average=self.window_average_data
        )
        
        prepared_data_sheath, _ = self.transformer_magnetosheath.prepare_data(
            data, 
            data_types=self.data_types,
            test_fraction=0,
            calc_mean_std=False,  # Use pre-calculated means and stds
            rand_perm=False,
            timestamps=self.timestamps,
            window_average=self.window_average_data
        )
        
        # Set models to evaluation mode for inference
        self.transformer_magnetotail.eval()
        self.transformer_magnetosheath.eval()
        
        # Predict using both models without gradient computation
        with torch.no_grad():
            pred_spect_tail, pred_vecs_tail, pred_scalars_tail, _, mask_spect_tail, mask_vectors_tail, mask_scalars_tail = \
                self.transformer_magnetotail.forward(
                    prepared_data_tail["spectrograms"], 
                    prepared_data_tail["vectors"],
                    prepared_data_tail["scalars"],
                    const_mask=True,
                    evaluate=True
                )
            
            pred_spect_sheath, pred_vecs_sheath, pred_scalars_sheath, _, mask_spect_sheath, mask_vectors_sheath, mask_scalars_sheath  = \
                self.transformer_magnetosheath.forward(
                    prepared_data_sheath["spectrograms"],
                    prepared_data_sheath["vectors"],
                    prepared_data_sheath["scalars"],
                    const_mask=True,
                    evaluate=True
                )


        # Put tail results into a dictionary
        results_tail_model = {
            'pred_spect': pred_spect_tail,
            'pred_vecs': pred_vecs_tail,
            'pred_scalars': pred_scalars_tail,
            'mask_spect': mask_spect_tail,
            'mask_vectors': mask_vectors_tail,
            'mask_scalars': mask_scalars_tail
        }

        # Put sheath results into a dictionary
        results_sheath_model = {
            'pred_spect': pred_spect_sheath,
            'pred_vecs': pred_vecs_sheath,
            'pred_scalars': pred_scalars_sheath,
            'mask_spect': mask_spect_sheath,
            'mask_vectors': mask_vectors_sheath,
            'mask_scalars': mask_scalars_sheath
        }

        return results_tail_model, results_sheath_model, prepared_data_sheath, prepared_data_tail
    
    def evaluate_accuracy(self, sections):

        region_true_all = []
        region_pred_all = []

        for section_key in sections.keys():
            data = {section_key: sections[section_key]}    

            if data[section_key]['region'] not in ['magnetotail', 'magnetosheath']:
                print(f"Skipping section {section_key} with unknown region {data[section_key]['region']}.")
                continue

            # Make prediction using the double model
            results_tail_model, results_sheath_model, prepared_data_sheath, prepared_data_tail = self.predict(data)


            # Extract average and std reconstruction errors for reference
            std_error_tail_stored = self.transformer_magnetotail.std_recon_error
            avg_error_tail_stored = self.transformer_magnetotail.avg_recon_error
            std_error_sheath_stored = self.transformer_magnetosheath.std_recon_error
            avg_error_sheath_stored = self.transformer_magnetosheath.avg_recon_error

            ### Calculate reconstruction error per timestamp ###
            # Initialize lists to store errors for each data type
            error_tail_list = []
            error_sheath_list = []

            # Loop through each tensor in the lists
            for i in range(len(results_tail_model['pred_spect'])):
                # Use loss function to calculate the reconstruction error for tail model
                _, error_tail_single = self.transformer_magnetotail.loss_function(
                results_tail_model['pred_spect'][i], 
                prepared_data_tail['spectrograms'],             
                results_tail_model['pred_vecs'][i], 
                prepared_data_tail['vectors'], 
                results_tail_model['pred_scalars'][i], 
                prepared_data_tail['scalars'], 
                results_tail_model['mask_spect'][i],
                results_tail_model['mask_vectors'][i],
                results_tail_model['mask_scalars'][i]
                )
                error_tail_list.append(error_tail_single)

                # Use loss function to calculate the reconstruction error for sheath model
                _, error_sheath_single = self.transformer_magnetosheath.loss_function(
                results_sheath_model['pred_spect'][i], 
                prepared_data_sheath['spectrograms'],             
                results_sheath_model['pred_vecs'][i], 
                prepared_data_sheath['vectors'], 
                results_sheath_model['pred_scalars'][i], 
                prepared_data_sheath['scalars'], 
                results_sheath_model['mask_spect'][i],
                results_sheath_model['mask_vectors'][i],
                results_sheath_model['mask_scalars'][i]
                )
                error_sheath_list.append(error_sheath_single)

            # Calculate average errors
            error_tail_model = torch.stack(error_tail_list).mean(dim=0)
            error_sheath_model = torch.stack(error_sheath_list).mean(dim=0)

            # Normalize errors by their respective mean and std, then calculate difference
            error_tail_normalized = (error_tail_model - std_error_tail_stored )/ avg_error_tail_stored
            error_sheath_normalized = (error_sheath_model - std_error_sheath_stored) / avg_error_sheath_stored
            reconstruction_error_diff_normalized = error_tail_normalized - error_sheath_normalized

            # Create region prediction based on which model has lower reconstruction error
            region_pred = torch.where(reconstruction_error_diff_normalized < 0, 0, 1)  # 0 for magnetotail, 1 for magnetosheath
            
            region_label = 1 if data[section_key]['region'] == 'magnetosheath' else 0
            region_true = torch.full_like(region_pred, region_label)

            region_true_all.extend(region_true.cpu().numpy().tolist())
            region_pred_all.extend(region_pred.cpu().numpy().tolist())

        # Stack
        region_true_all = np.array(region_true_all)
        region_pred_all = np.array(region_pred_all)

        # Use sklearn to calculate accuracy
        accuracy = accuracy_score(region_true_all, region_pred_all)
        class_report = classification_report(region_true_all, region_pred_all, target_names=['magnetotail', 'magnetosheath'], output_dict=True)
                                                                                             
                                                                                             
        return accuracy, class_report

    def plot_results(self, data, times=None, figsize=(12, 8), plot=True, crossing_times=None, window_average_errors=3):
        """Plot the results of the double transformer model.
        
        Args:
            data (dict): The input data used for prediction, is a single section.
            times (list or None): List of timestamps corresponding to the data. If None, will not plot time.
            figsize (tuple): Size of the figure for plotting.
        """
        if times is not None:
            # Ensure times is a tensor
            if isinstance(times, torch.Tensor):
                # Time data
                
                time_ns = times.cpu().numpy().astype('int64')
                time_s = time_ns / 1e9
                times_datetime = [datetime.fromtimestamp(ts, timezone.utc) for ts in time_s]
                times_np = [np.datetime64(int(ts), 'ns') for ts in time_ns]

                start_time_np = times_np[0]
                end_time_np = times_np[-1]


                
            else:
                # Assume times is a list of datetime objects
                times_np = times
                times_datetime = times

            # Only keep true crossings within the time range
            if crossing_times is not None:
                crossing_times = [ct for ct in crossing_times if start_time_np <= ct <= end_time_np]

            else:
                crossing_times = []
 
        # Make prediction using the double model
        results_tail_model, results_sheath_model, prepared_data_sheath, prepared_data_tail = self.predict(data)

        # Extract average and std reconstruction errors for reference
        std_error_tail_stored = self.transformer_magnetotail.std_recon_error
        avg_error_tail_stored = self.transformer_magnetotail.avg_recon_error
        std_error_sheath_stored = self.transformer_magnetosheath.std_recon_error
        avg_error_sheath_stored = self.transformer_magnetosheath.avg_recon_error

        ### Calculate reconstruction error per timestamp ###
        # Initialize lists to store errors for each data type
        error_tail_list = []
        error_sheath_list = []

        # Loop through each tensor in the lists
        for i in range(len(results_tail_model['pred_spect'])):
            # Use loss function to calculate the reconstruction error for tail model
            _, error_tail_single = self.transformer_magnetotail.loss_function(
            results_tail_model['pred_spect'][i], 
            prepared_data_tail['spectrograms'],             
            results_tail_model['pred_vecs'][i], 
            prepared_data_tail['vectors'], 
            results_tail_model['pred_scalars'][i], 
            prepared_data_tail['scalars'], 
            results_tail_model['mask_spect'][i],
            results_tail_model['mask_vectors'][i],
            results_tail_model['mask_scalars'][i]
            )
            error_tail_list.append(error_tail_single)

            # Use loss function to calculate the reconstruction error for sheath model
            _, error_sheath_single = self.transformer_magnetosheath.loss_function(
            results_sheath_model['pred_spect'][i], 
            prepared_data_sheath['spectrograms'],             
            results_sheath_model['pred_vecs'][i], 
            prepared_data_sheath['vectors'], 
            results_sheath_model['pred_scalars'][i], 
            prepared_data_sheath['scalars'], 
            results_sheath_model['mask_spect'][i],
            results_sheath_model['mask_vectors'][i],
            results_sheath_model['mask_scalars'][i]
            )
            error_sheath_list.append(error_sheath_single)

        # Calculate average errors
        error_tail_model = torch.stack(error_tail_list).mean(dim=0)
        error_sheath_model = torch.stack(error_sheath_list).mean(dim=0)

        # Calculate actual min and max values for normalization bounds
        min_error_tail = torch.min(error_tail_model).item()
        max_error_tail = torch.max(error_tail_model).item()
        
        min_error_sheath = torch.min(error_sheath_model).item()
        max_error_sheath = torch.max(error_sheath_model).item()

        # Calculate relative difference between reconstruction errors
        reconstruction_error_diff_relative = (error_tail_model/(max_error_tail - min_error_tail) - 
            error_sheath_model/(max_error_sheath - min_error_sheath))

        # Calculate reconstruction error difference using standard deviation and mean normalization
        mean_error_tail = torch.mean(error_tail_model)
        std_error_tail = torch.std(error_tail_model)
        mean_error_sheath = torch.mean(error_sheath_model)
        std_error_sheath = torch.std(error_sheath_model)
        
        # Normalize errors by their respective mean and std, then calculate difference
        error_tail_normalized = (error_tail_model - std_error_tail_stored )/ avg_error_tail_stored
        error_sheath_normalized = (error_sheath_model - std_error_sheath_stored) / avg_error_sheath_stored
        reconstruction_error_diff_normalized = error_tail_normalized - error_sheath_normalized

        # Convert errors to numpy for plotting
        error_tail_np = error_tail_model.cpu().numpy()
        error_sheath_np = error_sheath_model.cpu().numpy()
        error_diff_relative_np = reconstruction_error_diff_relative.cpu().numpy()
        error_diff_np = (error_tail_np - error_sheath_np)
        error_ratio = error_tail_np / (error_sheath_np + 1e-6)  # Avoid division by zero
        # Find zero points in the normalized, relative, and absolute error differences
        # Find zero crossings in the error differences




        if window_average_errors > 1:
            # Apply moving average smoothing to the error differences
            def moving_average(data, window_size):
                smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                return smoothed

            error_tail_np = moving_average(error_tail_np, window_average_errors)
            error_sheath_np = moving_average(error_sheath_np, window_average_errors)
            error_diff_relative_np = moving_average(error_diff_relative_np, window_average_errors)
            reconstruction_error_diff_normalized = moving_average(reconstruction_error_diff_normalized.cpu().numpy(), window_average_errors)
            error_diff_np = moving_average(error_diff_np, window_average_errors)

        else:
            reconstruction_error_diff_normalized = reconstruction_error_diff_normalized.cpu().numpy()

        e_diff = reconstruction_error_diff_normalized
        e_tail_norm = error_tail_normalized.cpu().numpy()
        e_sheath_norm = error_sheath_normalized.cpu().numpy()

        # # Example: 24-point moving average for background trend 
        # e_stacked = np.vstack([e_sheath_norm, e_tail_norm])
        # # Ensure errors are positive by clipping to (0, max) range
        # e_sheath_norm_pos = np.clip(e_sheath_norm, 0, np.max(e_sheath_norm))
        # e_tail_norm_pos = np.clip(e_tail_norm, 0, np.max(e_tail_norm))
        # b_gate = np.sqrt(e_sheath_norm_pos * e_tail_norm_pos) # Geometric mean as a simple background estimate
        # gate_window = 24
        
        # # Calculate the moving average
        # b_gate_m = np.convolve(b_gate, np.ones(gate_window)/gate_window, mode='valid')
        # pad_width = (len(b_gate) - len(b_gate_m)) // 2
        # b_gate_m = np.pad(b_gate_m, (pad_width, len(b_gate) - len(b_gate_m) - pad_width), mode='edge')

        # # Calculate the moving standard deviation
        # squared_diff = (b_gate - b_gate_m) ** 2
        # b_gate_std = np.sqrt(np.convolve(squared_diff, np.ones(gate_window)/gate_window, mode='valid'))
        # b_gate_std = np.pad(b_gate_std, (pad_width, len(b_gate) - len(b_gate_std) - pad_width), mode='edge')

        # # Make rolling 0.90 quantile based on mean and std
        # b_gate_q = b_gate_m + 0.5 * b_gate_std  # 90% quantile for normal distribution

        # Find zero crossings for all three error difference types
        zero_crossings_relative, zero_crossings_relative_interpolated = find_zero_crossings(error_diff_relative_np)
        zero_crossings_normalized, zero_crossings_normalized_interpolated = find_zero_crossings(reconstruction_error_diff_normalized, e_tail_norm, e_sheath_norm) # , b_gate, b_gate_q
        zero_crossings_absolute, zero_crossings_absolute_interpolated = find_zero_crossings(error_diff_np)

        # Store datetime moments if times are given
        if times is not None:
            # Calculate time deltas between consecutive timestamps for interpolation
            time_deltas = [times_np[i+1] - times_np[i] for i in range(len(times_np)-1)]
            
            # Time offset to account for the sliding window (timestamps-1) * 5 minutes
            time_offset = np.timedelta64(int(self.data_sampling_rate/2 * (self.timestamps - 1) * 60), 's')
            
            # Add window averaging of data offset if applicable
            if self.window_average_data > 1:
                time_offset += np.timedelta64(int(self.data_sampling_rate/2 * (self.window_average_data - 1) * 60), 's')

            # Add window averaging of errors offset if applicable
            if window_average_errors > 1:
                time_offset += np.timedelta64(int(self.data_sampling_rate/2 * (window_average_errors - 1) * 60), 's')
            
            # Apply interpolation to each set of zero crossings
            zero_crossing_times_relative = [
                interpolate_crossing_time(i, times_np, time_deltas, time_offset) for i in zero_crossings_relative_interpolated
            ]
            zero_crossing_times_normalized = [
                interpolate_crossing_time(i, times_np, time_deltas, time_offset) for i in zero_crossings_normalized_interpolated
            ]
            zero_crossing_times_absolute = [
                interpolate_crossing_time(i, times_np, time_deltas, time_offset) for i in zero_crossings_absolute_interpolated
            ]
        else:
            zero_crossing_times_relative = None
            zero_crossing_times_normalized = None
            zero_crossing_times_absolute = None
        
        if plot:
            if times is None:
                times = data[list(data.keys())[0]]['times']
                if self.timestamps % 2 == 0:
                    times = times[(self.timestamps//2-1):-(self.timestamps//2)]
                else:
                    times = times[(self.timestamps//2):-(self.timestamps//2)]
                times_datetime = times.numpy().astype('datetime64[ns]')
                start_time_np = times_datetime[0]
                end_time_np = times_datetime[-1]

            # Create subplots for all visualizations (4 rows, 1 column)
            fig, axes = plt.subplots(5, 1, figsize=figsize)
            ax1, ax2, ax3, ax4, ax5 = axes
            
            # Set up x-axis data
            if times_datetime is not None:
                x_data = mdates.date2num(times_datetime)
                xlabel = 'Time'
            else:
                x_data = range(len(error_tail_np))
                xlabel = 'Sample Index'
            
            # Plot 1: Tail Model Reconstruction Error
            ax1.plot(x_data, e_tail_norm, label='Tail Model Error', color='blue')
            ax1.set_title('Tail Model Reconstruction Error')
            ax1.set_ylabel('Reconstruction Error')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Sheath Model Reconstruction Error
            ax2.plot(x_data, e_sheath_norm, label='Sheath Model Error', color='orange')
            ax2.set_title('Sheath Model Reconstruction Error')
            ax2.set_ylabel('Reconstruction Error')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Relative Difference Between Models
            ax3.plot(x_data[:-2], error_diff_relative_np, label='Relative Difference', color='green')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Relative Difference Between Model Errors')
            ax3.set_ylabel('Relative Error Difference')
            ax3.grid(True, alpha=0.3)

            # Plot 4: Normalized Error Difference
            ax4.plot(x_data[:-2], reconstruction_error_diff_normalized, label='Normalized Error Difference', color='brown')
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_title('Normalized Error Difference Between Models')
            ax4.set_ylabel('Normalized Error Difference')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Error Difference Between Models
            ax5.plot(x_data[:-2], error_diff_np, label='Error Difference', color='purple')
            ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax5.set_title('Error Difference Between Models')
            ax5.set_ylabel('Error Difference')
            ax5.grid(True, alpha=0.3)


            # Plot on each subplot the true crossings
            for ax in axes:
                if crossing_times is not None:
                    for ct in crossing_times:
                        if times is not None:
                            if start_time_np <= ct <= end_time_np:
                                ax.axvline(x=mdates.date2num(ct), color='black', linestyle=':', alpha=0.3)
                        else:
                            # If no times are given, we cannot plot true crossings
                            pass
            
            # Format x-axis for all subplots
            for ax in axes:
                ax.set_xlabel(xlabel)
                if times is not None:
                    ax.xaxis_date()
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))
            
            if times is not None:
                fig.autofmt_xdate()

            plt.tight_layout()
            plt.show()

        results = {
            'zero_crossing_times_relative': zero_crossing_times_relative,
            'zero_crossing_times_normalized': zero_crossing_times_normalized,
            'zero_crossing_times_absolute': zero_crossing_times_absolute,
            'true_crossings': crossing_times
        }

        return results
    
    def get_results(self, data, crossing_times=None, window_average_errors=3):
        """Get prediction results without plotting, for multiple sections contained in data dictionary.
        
        Args:
            data (dict): The input data used for prediction, contains multiple sections.
            
        Returns:
            results (dict): Dictionary containing total prediction results combining all sections.
        """

        All_results = []

        for section_key in data.keys():
            section_data = data[section_key]
            section_time = section_data['times']

            # Use plot results function with plot=False to get results without plotting
            section_results = self.plot_results(
                {section_key: section_data}, 
                section_time, 
                crossing_times=crossing_times, 
                plot=False, 
                window_average_errors=window_average_errors
            )
            
            All_results.append(section_results)

        # Gather all crossing times from all sections
        all_zero_crossings_relative = []
        all_zero_crossings_normalized = []
        all_zero_crossings_absolute = []
        all_true_crossings = []

        for res in All_results:
            all_zero_crossings_relative.extend(res['zero_crossing_times_relative'])
            all_zero_crossings_normalized.extend(res['zero_crossing_times_normalized'])
            all_zero_crossings_absolute.extend(res['zero_crossing_times_absolute'])
            all_true_crossings.extend(res['true_crossings'])


        results = {
            'zero_crossing_times_relative': all_zero_crossings_relative,
            'zero_crossing_times_normalized': all_zero_crossings_normalized,
            'zero_crossing_times_absolute': all_zero_crossings_absolute,
            'true_crossings': all_true_crossings
        }

        return results
    
    def get_model_score(self, results, time_window_minutes=30, method='hungarian'):
        """ 
        Calculate performance scores based on crossing detection results. Crossings within a specified time window are considered matches.
        
        Parameters:
        results (dict): Results from show_results function containing true_crossings and predicted_crossings
        time_window_minutes (int): Time window in minutes to consider crossings as matches (default: 30)
        
        Returns:
        dict: Dictionary containing TPR, FPR, FNR and related metrics, plus a pandas DataFrame
        """
        
        # Extract crossing times
        true_crossings = results['true_crossings']
        predicted_crossings_relative = results['zero_crossing_times_relative']
        predicted_crossings_normalized = results['zero_crossing_times_normalized']
        predicted_crossings_absolute = results['zero_crossing_times_absolute']
    
        # Calculate metrics for each method
        tp_rel, fp_rel, fn_rel, tp_time_diffs_minutes_rel, tp_time_diffs_minutes_rel_signed, in_range_rel, out_range_rel, removed_true = calculate_metrics(true_crossings, predicted_crossings_relative, time_window=time_window_minutes, method=method)
        tp_norm, fp_norm, fn_norm, tp_time_diffs_minutes_norm, tp_time_diffs_minutes_norm_signed, in_range_norm, out_range_norm, removed_true = calculate_metrics(true_crossings, predicted_crossings_normalized, time_window=time_window_minutes, method=method)
        tp_abs, fp_abs, fn_abs, tp_time_diffs_minutes_abs, tp_time_diffs_minutes_abs_signed, in_range_abs, out_range_abs, removed_true = calculate_metrics(true_crossings, predicted_crossings_absolute, time_window=time_window_minutes, method=method)
    
        # Calculate rates for each method
        tpr_rel, fpr_rel, fnr_rel, prec_rel, f1_rel, f0_5_rel = calculate_rates(tp_rel, fp_rel, fn_rel)
        tpr_norm, fpr_norm, fnr_norm, prec_norm, f1_norm, f0_5_norm = calculate_rates(tp_norm, fp_norm, fn_norm)
        tpr_abs, fpr_abs, fnr_abs, prec_abs, f1_abs, f0_5_abs = calculate_rates(tp_abs, fp_abs, fn_abs)
        
        # Calculate mean and std of true positive time accuracy for each method
        tp_time_accuracy_mean_rel = np.mean(tp_time_diffs_minutes_rel) if tp_time_diffs_minutes_rel else None
        tp_time_accuracy_std_rel = np.std(tp_time_diffs_minutes_rel) if tp_time_diffs_minutes_rel else None
        tp_time_accuracy_mean_norm = np.mean(tp_time_diffs_minutes_norm) if tp_time_diffs_minutes_norm else None
        tp_time_accuracy_std_norm = np.std(tp_time_diffs_minutes_norm) if tp_time_diffs_minutes_norm else None
        tp_time_accuracy_mean_abs = np.mean(tp_time_diffs_minutes_abs) if tp_time_diffs_minutes_abs else None
        tp_time_accuracy_std_abs = np.std(tp_time_diffs_minutes_abs) if tp_time_diffs_minutes_abs else None

        # Calculate mean and std of signed true positive time accuracy for each method
        tp_time_accuracy_mean_rel_signed = np.mean(tp_time_diffs_minutes_rel_signed) if tp_time_diffs_minutes_rel_signed else None
        tp_time_accuracy_std_rel_signed = np.std(tp_time_diffs_minutes_rel_signed) if tp_time_diffs_minutes_rel_signed else None
        tp_time_accuracy_mean_norm_signed = np.mean(tp_time_diffs_minutes_norm_signed) if tp_time_diffs_minutes_norm_signed else None
        tp_time_accuracy_std_norm_signed = np.std(tp_time_diffs_minutes_norm_signed) if tp_time_diffs_minutes_norm_signed else None
        tp_time_accuracy_mean_abs_signed = np.mean(tp_time_diffs_minutes_abs_signed) if tp_time_diffs_minutes_abs_signed else None
        tp_time_accuracy_std_abs_signed = np.std(tp_time_diffs_minutes_abs_signed) if   tp_time_diffs_minutes_abs_signed else None



        # Create DataFrame with all results
        df_data = {
            'Metric': ['True Positive Rate', 'False Positive Rate', 'False Negative Rate', 
                  'True Positives','In Range Positives', 'Out Range Predicted Positives','False Positives', 'False Negatives',
                  'Precision', 'F1 Score', 'F0.5 Score',
                  'Total True Crossings', 'Total Predicted Crossings',
                  'Time Window (min)', 'TP Time Accuracy Mean (min)', 
                  'TP Time Accuracy Std (min)', 'TP Time Accuracy Mean Signed (min)', 
                  'TP Time Accuracy Std Signed (min)'],
            'Relative': [f"{tpr_rel:.3f}", f"{fpr_rel:.3f}", f"{fnr_rel:.3f}",
                int(tp_rel), int(in_range_rel), int(out_range_rel),int(fp_rel), int(fn_rel),
                f"{prec_rel:.3f}", f"{f1_rel:.3f}", f"{f0_5_rel:.3f}",
                len(true_crossings), len(predicted_crossings_relative), time_window_minutes,
                f"{tp_time_accuracy_mean_rel:.2f}" if tp_time_accuracy_mean_rel is not None else "N/A",
                f"{tp_time_accuracy_std_rel:.2f}" if tp_time_accuracy_std_rel is not None else "N/A",
                f"{tp_time_accuracy_mean_rel_signed:.2f}" if tp_time_accuracy_mean_rel_signed is not None else "N/A",
                f"{tp_time_accuracy_std_rel_signed:.2f}" if tp_time_accuracy_std_rel_signed is not None else "N/A"],
            'Normalized': [f"{tpr_norm:.3f}", f"{fpr_norm:.3f}", f"{fnr_norm:.3f}",
                  int(tp_norm), int(in_range_norm), int(out_range_norm), int(fp_norm), int(fn_norm),
                  f"{prec_norm:.3f}", f"{f1_norm:.3f}", f"{f0_5_norm:.3f}",
                  len(true_crossings), len(predicted_crossings_normalized), time_window_minutes,
                  f"{tp_time_accuracy_mean_norm:.2f}" if tp_time_accuracy_mean_norm is not None else "N/A",
                  f"{tp_time_accuracy_std_norm:.2f}" if tp_time_accuracy_std_norm is not None else "N/A",
                  f"{tp_time_accuracy_mean_norm_signed:.2f}" if tp_time_accuracy_mean_norm_signed is not None else "N/A",
                  f"{tp_time_accuracy_std_norm_signed:.2f}" if tp_time_accuracy_std_norm_signed is not None else "N/A"],
            'Absolute': [f"{tpr_abs:.3f}", f"{fpr_abs:.3f}", f"{fnr_abs:.3f}",
                int(tp_abs), int(in_range_abs), int(out_range_abs), int(fp_abs), int(fn_abs),
                f"{prec_abs:.3f}", f"{f1_abs:.3f}", f"{f0_5_abs:.3f}",
                len(true_crossings), len(predicted_crossings_absolute), time_window_minutes,
                f"{tp_time_accuracy_mean_abs:.2f}" if tp_time_accuracy_mean_abs is not None else "N/A",
                f"{tp_time_accuracy_std_abs:.2f}" if tp_time_accuracy_std_abs is not None else "N/A",
                f"{tp_time_accuracy_mean_abs_signed:.2f}" if tp_time_accuracy_mean_abs_signed is not None else "N/A",
                f"{tp_time_accuracy_std_abs_signed:.2f}" if tp_time_accuracy_std_abs_signed is not None else "N/A"]
        }
        
        df = pd.DataFrame(df_data)
        
        dict_all = {'Metric': ['True Positive Rate', 'False Positive Rate', 'False Negative Rate', 
                        'True Positives','In Range Positives', 'Out Range Predicted Positives','False Positives', 'False Negatives',
                        'Precision', 'F1 Score', 'F0.5 Score',
                        'Total True Crossings', 'Total Predicted Crossings',
                        'Time Window (min)', 'TP Time Accuracy Mean (min)',
                        'TP Time Accuracy Std (min)', 'TP Time Accuracy Mean Signed (min)',
                        'TP Time Accuracy Std Signed (min)'],
                    
                    'Relative': np.array([tpr_rel, fpr_rel, fnr_rel,
                        int(tp_rel), int(in_range_rel), int(out_range_rel),int(fp_rel), int(fn_rel),
                        prec_rel, f1_rel, f0_5_rel,
                        len(true_crossings), len(predicted_crossings_relative), time_window_minutes,
                        tp_time_accuracy_mean_rel ,
                        tp_time_accuracy_std_rel ,
                        tp_time_accuracy_mean_rel_signed ,
                        tp_time_accuracy_std_rel_signed]),
                    'Normalized':  np.array([tpr_norm, fpr_norm, fnr_norm,
                        int(tp_norm), int(in_range_norm), int(out_range_norm), int(fp_norm), int(fn_norm),
                        prec_norm, f1_norm, f0_5_norm,
                        len(true_crossings), len(predicted_crossings_normalized), time_window_minutes,
                        tp_time_accuracy_mean_norm ,
                        tp_time_accuracy_std_norm ,
                        tp_time_accuracy_mean_norm_signed ,
                        tp_time_accuracy_std_norm_signed ]),
                    'Absolute':  np.array([tpr_abs, fpr_abs, fnr_abs, 
                        int(tp_abs), int(in_range_abs), int(out_range_abs), int(fp_abs), int(fn_abs),
                        prec_abs, f1_abs, f0_5_abs,
                        len(true_crossings), len(predicted_crossings_absolute), time_window_minutes,
                        tp_time_accuracy_mean_abs ,
                        tp_time_accuracy_std_abs ,
                        tp_time_accuracy_mean_abs_signed ,
                        tp_time_accuracy_std_abs_signed ])
                    }
        
        
        
        # Return dictionary with summary and DataFrame
        return {
            'summary': {
                'relative': {'TPR': tpr_rel, 'FPR': fpr_rel, 'FNR': fnr_rel, 'Precision': prec_rel, 'F1': f1_rel, 'F0.5': f0_5_rel},
                'normalized': {'TPR': tpr_norm, 'FPR': fpr_norm, 'FNR': fnr_norm, 'Precision': prec_norm, 'F1': f1_norm, 'F0.5': f0_5_norm},
                'absolute': {'TPR': tpr_abs, 'FPR': fpr_abs, 'FNR': fnr_abs, 'Precision': prec_abs, 'F1': f1_abs, 'F0.5': f0_5_abs}
            },
            'dataframe': df,
            'detailed_dict': dict_all,
            'time_window_minutes': time_window_minutes,
            'total_true_crossings': len(true_crossings)
        }

    def save_models(self, path_tail, path_sheath):
        """Save both transformer models to specified file paths."""
        self.transformer_magnetotail.save_model(path_tail)
        self.transformer_magnetosheath.save_model(path_sheath)
        torch.save(self.window_average_data, path_tail.replace('.pth', '_window_avg.pth'))
        
    def load_models(self, path_tail, path_sheath):
        """Load both transformer models from specified file paths."""
        self.transformer_magnetotail.load_model(path_tail)
        self.transformer_magnetosheath.load_model(path_sheath)
        if os.path.exists(path_tail.replace('.pth', '_window_avg.pth')):
            self.window_average_data = torch.load(path_tail.replace('.pth', '_window_avg.pth'))
            
def interpolate_crossing_time(crossing_index, times_np, time_deltas, time_offset):
    """Interpolate exact crossing time based on fractional index."""
    int_index = int(crossing_index)

    if crossing_index == int_index:
        # Exact integer index - no interpolation needed
        return times_np[int_index] + time_offset
    
    else:
        # Fractional index - interpolate between two timestamps
        fraction = crossing_index - int_index
        interpolated_time = times_np[int_index] + time_deltas[int_index] * fraction
        return interpolated_time + time_offset            

def find_zero_crossings(data, e_tail=None, e_sheath=None):
    """Find indices where the data crosses zero."""
    zero_crossings = []
    zero_crossings_interpolated = []
    
    for i in range(len(data) - 1):
        if (data[i] >= 0 and data[i+1] < 0) or (data[i] < 0 and data[i+1] >= 0):

            # Remove crossings where both e_tail and e_sheath are above 3 standard deviations(they're already normalized)
            if (e_tail is not None) and (e_sheath is not None) and min([e_tail[i], e_sheath[i], e_tail[i+1], e_sheath[i+1]]) >  2.5:
                continue

            # # Remove crossings if neither e_tail and e_sheath are high
            # if (e_tail is not None) and (e_sheath is not None):
            #     start_idx = max(0, i-10)
            #     end_idx = min(len(e_tail), i+11)
            #     if max(max(e_tail[start_idx:end_idx]), max(e_sheath[start_idx:end_idx])) < 4.5:
            #         continue

            zero_crossings.append(i)
            
            # Linear interpolation to find exact zero crossing
            if data[i+1] != data[i]:  # Avoid division by zero
                # Interpolated position: i + fraction where zero occurs
                fraction = -data[i] / (data[i+1] - data[i])
                interpolated_index = i + fraction
            else:
                interpolated_index = i  # Fallback if slope is zero
                
            zero_crossings_interpolated.append(interpolated_index)
    
    return zero_crossings, zero_crossings_interpolated

def calculate_metrics(true_crossings, predicted_crossings, time_window=30, method='hungarian'):
    """Calculate TP, FP, FN for a set of predicted crossings."""
    time_window = np.timedelta64(time_window, 'm')  # Convert minutes to timedelta64

    if not true_crossings or not predicted_crossings:
        return 0, len(predicted_crossings), len(true_crossings)
    
    true_matched = set()
    predicted_matched = set()
    tp_time_diffs_minutes = []
    tp_time_diffs_minutes_signed = []
    
    # Find matches within time window - ensure one-to-one matching with closest pairs
    unmatched_true = list(range(len(true_crossings)))
    unmatched_pred = list(range(len(predicted_crossings)))
    
    if method == 'hungarian':
        results = match_crossings_hungarian(true_crossings, predicted_crossings, time_window, unmatched_true, unmatched_pred, removal_window_minutes=10)
    elif method == 'greedy':
        results = match_crossings_greedy(true_crossings, predicted_crossings, time_window, unmatched_true, unmatched_pred)
    
    

    tp = len(results.true_matched)
    fp = len(predicted_crossings) - len(results.pred_matched)
    fn = len(true_crossings) - len(results.true_matched) - len(results.removed_true_crossings)
    in_range_count_true = len(results.in_range_true)
    out_range_count_pred = len(results.out_range_pred)

    
    return tp, fp, fn, results.tp_time_diffs_minutes, results.tp_time_diffs_minutes_signed, in_range_count_true, out_range_count_pred, results.removed_true_crossings
        
def calculate_rates(tp, fp, fn):
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0  # False Positive Rate
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    f0_5 = (1 + 0.5**2) * (precision * tpr) / ((0.5**2 * precision) + tpr) if ((0.5**2 * precision) + tpr) > 0 else 0
    return tpr, fpr, fnr, precision, f1, f0_5