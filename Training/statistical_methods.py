import sys
from pathlib import Path
from copy import deepcopy
sys.path.append(str(Path().resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.lines import Line2D
from tqdm import tqdm
from matplotlib.colors import LogNorm

from Helper.preprocess_functions import average_window
from Helper.crossoptim import match_crossings_hungarian, match_crossings_greedy
from Helper.preprocess_functions import remove_outliers_with_local_interpolation, remove_outliers_with_interpolation, downweight_variable
from sklearn.ensemble import GradientBoostingClassifier

def convert_crossings_to_regions(dataset, samples, crossings, section_keys=None):
    """
    Convert boundary crossings to regions between crossings for multiple sections.
    
    Parameters:
    samples (dict): Dictionary containing the data samples. 
    crossings (list): List of datetime objects where boundary crossings occur.
    section_keys (list or str): Key(s) to access the specific section(s) in the dataset.
    dataset (MagnetotailDataset): The dataset object containing the data.
    
    Returns:
    samples (dict): Updated 'region' key in samples with region labels.
    """
    # Handle single section case
    if isinstance(section_keys, str):
        section_keys = [section_keys]
    elif section_keys is None:
        section_keys = list(samples.keys())
    
    # Convert datetime objects to numpy.datetime64[ns] for comparison
    crossings_ns = [np.datetime64(crossing, 'ns') for crossing in crossings]
    crossings_ns.sort()
    
    # Get section times dictionary
    section_times_dict = dataset.get_section_times()
    
    # Process each section separately
    for section_key in tqdm(section_keys):
        section_times = section_times_dict[section_key]
        section_length = len(section_times)
        section_start_time = section_times[0]
        
        # Initialize region binary array for this section
        section_regions_binary = np.zeros(section_length, dtype=int)
        
        # Count crossings before this section starts
        crossings_before_section = sum(1 for crossing in crossings_ns if crossing < section_start_time)
        
        # Apply crossings within this section
        for i in range(section_length):
            time_point = section_times[i]
            
            # Count crossings from the beginning up to this time point
            total_crossings_before = sum(1 for crossing in crossings_ns if crossing <= time_point)
            
            # Determine region based on total number of crossings
            if total_crossings_before % 2 == 0:
                section_regions_binary[i] = 0  # magnetosheath
            else:
                section_regions_binary[i] = 1  # magnetotail
        
        # Add binary regions to samples dictionary for this specific section
        samples[section_key]['region_binary'] = section_regions_binary
    
    return samples

class HysteresisGradientBoostedTree:
    """
    Gradient boosted tree model with hysteresis for region prediction.
    """
    
    def __init__(self, feature_keys=['ion_density', 'ion_avgtemp', 'plasma_beta', 'magnetic_field_gsm_x'], hysteresis_threshold=0.1, time_hysteresis=None, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, min_samples_leaf=100):
        """
        Initialize the gradient boosted tree with hysteresis.
        
        Parameters:
        feature_keys (list): List of keys to use as features from the data dictionary
        hysteresis_threshold (float): Threshold for hysteresis effect (0.0 to 0.5)
        n_estimators (int): Number of boosting stages to be run
        max_depth (int): Maximum depth of the individual regression estimators
        learning_rate (float): Learning rate shrinks the contribution of each tree
        random_state (int): Random state for reproducibility
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node
        """
        
        self.feature_keys = feature_keys
        self.hysteresis_threshold = hysteresis_threshold
        self.tree = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf
        )
        self.last_prediction = None
        self.is_fitted = False
        self.time_hysteresis = time_hysteresis  # Number of consecutive timestamps required for region change
        self.use_preprocessing = True  # Whether to apply preprocessing steps during feature extraction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.window_size = 8
        self.threshold = 4
        self.no_overlap = 2
    
    def calculate_mean_std(self, data, window_average=1):
        """
        Calculate mean and standard deviation for normalization across all sections and data types.
        
        Args:
            data (dict): Dictionary containing the data for each section.
            window_average (int): Window size for averaging.
        """
        self.means = {}
        self.stds = {}
        
        # Process each data type
        for data_type in self.feature_keys:
            all_values = []
            
            # Collect all values across all sections for this data type
            for section_key in data.keys():
                section = data[section_key]
                
                if data_type not in section:
                    continue
                
                section_data = section[data_type]
                section_length = len(section_data)
                
                # Process each timestamp
                for timestamp_idx in range(section_length):
                    timestamp_data = section_data[timestamp_idx]
                    
                    # Apply window averaging if needed
                    if window_average > 1 and timestamp_idx >= window_average - 1:
                        start_idx = max(0, timestamp_idx - window_average + 1)
                        timestamp_data = section_data[start_idx:timestamp_idx + 1].mean(dim=0)
                    
                    # Process based on data type and shape
                    if len(timestamp_data.shape) == 1 and timestamp_data.shape[0] > 3:  # Spectrogram data
                        # Apply log transform first
                        log_data = torch.log(timestamp_data + 1e-10)
                        all_values.append(log_data)
                        
                    elif len(timestamp_data.shape) == 1 and timestamp_data.shape[0] <= 3:  # Vector data
                        # Apply log transform for specific data types
                        if data_type in ['ion_avgtemp', 'ion_density', 'plasma_beta', 'ion_vthermal', 
                                       'electron_avgtemp', 'electron_density', 'electron_vthermal']:
                            timestamp_data = torch.log(timestamp_data + 1e-10)
                        all_values.append(timestamp_data)
                        
                    elif len(timestamp_data.shape) == 0 or timestamp_data.numel() == 1:  # Scalar data
                        # Apply log transform for specific data types
                        if data_type in ['ion_avgtemp', 'ion_density', 'plasma_beta', 'ion_vthermal',
                                       'electron_avgtemp', 'electron_density', 'electron_vthermal']:
                            timestamp_data = torch.log(timestamp_data + 1e-10)
                        all_values.append(timestamp_data)
            
            # Calculate mean and std if we have values
            if all_values:
                # Stack all values
                stacked_values = torch.stack(all_values)
                
                # Calculate mean and std across all samples
                if len(stacked_values.shape) > 1:  # Multi-dimensional data (like spectrograms)
                    self.means[data_type] = stacked_values.mean(dim=0)
                    self.stds[data_type] = stacked_values.std(dim=0)
                else:  # Scalar data
                    self.means[data_type] = stacked_values.mean()
                    self.stds[data_type] = stacked_values.std()
    
    def prepare_features(self, 
                        data, 
                        test_fraction=0.15, 
                        rand_perm=True, 
                        calc_mean_std=True, 
                        window_average=1,
                        inference=False
                        ):
        """
        Prepares the data for classification using gradient boosted trees.
        
        Args:
            data (dict): Dictionary containing the data for each modality, per section.
            test_fraction (float): Fraction of data to use for testing.
            rand_perm (bool): Whether to randomly permute the data.
            calc_mean_std (bool): Whether to calculate and apply normalization.
            window_average (int): Window size for averaging.

        Returns:
            tuple: A tuple containing:
                - train_features (torch.Tensor): [N_train, n_features] - Training features
                - train_labels (torch.Tensor): [N_train] - Training labels (0=magnetosheath, 1=magnetotail)
                - test_features (torch.Tensor): [N_test, n_features] - Test features  
                - test_labels (torch.Tensor): [N_test] - Test labels (0=magnetosheath, 1=magnetotail)
        """
        
        # 1. Calculate normalization variables if needed
        if calc_mean_std:
            self.calculate_mean_std(
                data, 
                window_average=window_average, 
            )

        # Lists to store all features and labels
        all_features = []
        all_labels = []

        # 2. Process each section
        for section_key in data.keys():
            section = data[section_key]
            
            if section['region'] not in ['magnetotail', 'magnetosheath'] and not inference:
                continue

            # Get region label (0=magnetosheath, 1=magnetotail)
            region_label = 1 if section['region'] == 'magnetotail' else 0
            
            # Get the length of data in this section
            first_data_type = self.feature_keys[0]
            section_length = len(section[first_data_type])
            
            # Process each timestamp individually
            for timestamp_idx in range(section_length):
                features_for_timestamp = []
                
                # Process each data type
                for data_type in self.feature_keys:
                    if data_type not in section:
                        continue
                        
                    # Get data for this timestamp
                    timestamp_data = section[data_type][timestamp_idx].to(self.device)
                    
                    # Apply window averaging if needed (though with single timestamps this may not be relevant)
                    if window_average > 1 and timestamp_idx >= window_average - 1:
                        # Take average of current and previous timestamps
                        start_idx = max(0, timestamp_idx - window_average + 1)
                        timestamp_data = section[data_type][start_idx:timestamp_idx + 1].mean(dim=0).to(self.device)
                    
                    # Determine data type and process accordingly
                    if len(timestamp_data.shape) == 1 and timestamp_data.shape[0] > 3:  # Spectrogram data
                        # Take logarithm
                        timestamp_data = torch.log(timestamp_data + 1e-10)
                        
                        # Normalize if means/stds available
                        if data_type in self.means:
                            mean = self.means[data_type].to(self.device)
                            std = self.stds[data_type].to(self.device)
                            timestamp_data = (timestamp_data - mean) / (std + 1e-10)
                        
                        # Calculate mean and std across energy bins
                        flux_mean = timestamp_data.mean()
                        flux_std = timestamp_data.std()
                        features_for_timestamp.extend([flux_mean, flux_std])
                        
                    elif len(timestamp_data.shape) == 1 and timestamp_data.shape[0] <= 3:  # Vector data
                        # Apply outlier removal and log transform if needed
                        if data_type in ['ion_avgtemp', 'ion_density', 'ion_vthermal', 'electron_avgtemp', 'electron_density', 'electron_vthermal', 'plasma_beta', 'magnetic_field_gsm_magnitude', 'magnetic_field_gsm_x']:
                            # Note: outlier removal function expects 1D data, may need adjustment for single values
                            pass
                        
                        if data_type in ['ion_avgtemp', 'ion_density', 'plasma_beta', 'ion_vthermal', 'electron_avgtemp', 'electron_density', 'electron_vthermal']:
                            timestamp_data = torch.log(timestamp_data + 1e-10)
                        
                        # Normalize if means/stds available
                        if data_type in self.means:
                            mean = self.means[data_type].to(self.device)
                            std = self.stds[data_type].to(self.device)
                            timestamp_data = (timestamp_data - mean) / (std + 1e-10)
                        
                        # Calculate magnitude for vector data
                        magnitude = torch.norm(timestamp_data)
                        features_for_timestamp.append(magnitude)
                        
                    elif len(timestamp_data.shape) == 0 or timestamp_data.numel() == 1:  # Scalar data
                        # Apply outlier removal and log transform if needed
                        if data_type in ['ion_avgtemp', 'ion_density', 'ion_vthermal', 'electron_avgtemp', 'electron_density', 'electron_vthermal', 'plasma_beta', 'magnetic_field_gsm_magnitude', 'magnetic_field_gsm_x']:
                            pass  # Skip outlier removal for single values
                        
                        if data_type in ['ion_avgtemp', 'ion_density', 'plasma_beta', 'ion_vthermal', 'electron_avgtemp', 'electron_density', 'electron_vthermal']:
                            timestamp_data = torch.log(timestamp_data + 1e-10)
                        
                        # Normalize if means/stds available
                        if data_type in self.means:
                            mean = self.means[data_type].to(self.device)
                            std = self.stds[data_type].to(self.device)
                            timestamp_data = (timestamp_data - mean) / (std + 1e-10)
                        
                        # Add scalar value directly
                        features_for_timestamp.append(timestamp_data)
                
                # Convert features to tensor and add to lists
                if features_for_timestamp:
                    features_tensor = torch.stack([f if f.dim() == 0 else f.squeeze() for f in features_for_timestamp])
                    all_features.append(features_tensor)
                    all_labels.append(region_label)
        
        # Convert to tensors
        all_features = torch.stack(all_features)  # [N, n_features]
        all_labels = torch.tensor(all_labels, device=self.device)  # [N]
        
        # Randomly permute if specified
        if rand_perm:
            perm = torch.randperm(all_features.shape[0])
            all_features = all_features[perm]
            all_labels = all_labels[perm]
        
        # Split into train and test sets
        total_samples = all_features.shape[0]
        test_size = int(total_samples * test_fraction)
        train_size = total_samples - test_size
        
        train_features = all_features[:train_size].cpu().numpy()
        train_labels = all_labels[:train_size].cpu().numpy()
        test_features = all_features[train_size:].cpu().numpy()
        test_labels = all_labels[train_size:].cpu().numpy()
        
        return train_features, train_labels, test_features, test_labels

    def fit(self, samples, crossing_times_ns, window_average=1, exclude_sections=None, use_preprocessing=True):
        """
        Train the gradient boosted tree model.
        
        Parameters:
        samples (dict): Dictionary containing features and 'region_binary' target
        exclude_sections (list): List of section keys to exclude from training
        """
        
        self.use_preprocessing = use_preprocessing

        X, y, _, _ = self.prepare_features(samples, window_average=window_average)
        self.window_average=window_average
        
        # Fit the gradient boosted tree
        self.tree.fit(X, y)
        self.is_fitted = True
    
    def predict_with_hysteresis(self, samples):
        """
        Predict regions with hysteresis effect.
        
        Parameters:
        samples (dict): Dictionary containing features
        time_hysteresis (int): Number of consecutive timestamps required for region change
        
        Returns:
        np.ndarray: Predictions with hysteresis applied
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _, _, _ = self.prepare_features(samples, test_fraction=0.0, rand_perm=False)
        if not isinstance(X, np.ndarray):
            X = X.numpy()
        
        # Get probability predictions
        probabilities = self.tree.predict_proba(X)
        predictions = np.zeros(len(X), dtype=int)
        
        if self.time_hysteresis is None:
            # Original hysteresis implementation
            for i in range(len(X)):
                prob_magnetotail = probabilities[i, 1] if probabilities.shape[1] > 1 else 0
                
                if self.last_prediction is None:
                    # First prediction without hysteresis
                    predictions[i] = 1 if prob_magnetotail > 0.5 else 0
                else:
                    # Apply hysteresis
                    if self.last_prediction == 0:  # Currently in magnetosheath
                        # Need higher threshold to switch to magnetotail
                        predictions[i] = 1 if prob_magnetotail > (0.5 + self.hysteresis_threshold) else 0
                    else:  # Currently in magnetotail
                        # Need lower threshold to switch to magnetosheath
                        predictions[i] = 0 if prob_magnetotail < (0.5 - self.hysteresis_threshold) else 1
                
                self.last_prediction = predictions[i]
        else:
            # Time-based hysteresis implementation
            consecutive_count = 0
            pending_change = None
            change_start_idx = None
            
            for i in range(len(X)):
                prob_magnetotail = probabilities[i, 1] if probabilities.shape[1] > 1 else 0
                
                if self.last_prediction is None:
                    # First prediction without hysteresis
                    predictions[i] = 1 if prob_magnetotail > 0.5 else 0
                    self.last_prediction = predictions[i]
                    consecutive_count = 0
                    continue
                
                # Determine what the new prediction would be
                if consecutive_count == 0:
                    # Use hysteresis threshold only for the first detection
                    if self.last_prediction == 0:  # Currently in magnetosheath
                        new_prediction = 1 if prob_magnetotail > (0.5 + self.hysteresis_threshold) else 0
                    else:  # Currently in magnetotail
                        new_prediction = 0 if prob_magnetotail < (0.5 - self.hysteresis_threshold) else 1
                else:
                    # For subsequent detections, use standard 0.5 threshold
                    new_prediction = 1 if prob_magnetotail > 0.5 else 0
                
                # Check if this would be a region change
                if new_prediction != self.last_prediction:
                    if pending_change is None or pending_change != new_prediction:
                        # Start counting for a new potential change
                        pending_change = new_prediction
                        consecutive_count = 1
                        change_start_idx = i
                    else:
                        # Continue counting for the same change
                        consecutive_count += 1
                    
                    # Check if we've reached the required consecutive count
                    if consecutive_count >= self.time_hysteresis:
                        # Apply the change to all timestamps from start to current
                        for j in range(change_start_idx, i + 1):
                            predictions[j] = pending_change
                        self.last_prediction = pending_change
                        consecutive_count = 0
                        pending_change = None
                        change_start_idx = None
                    else:
                        # Not enough consecutive points yet, keep current prediction
                        predictions[i] = self.last_prediction
                else:
                    # No region change, reset counters and keep current prediction
                    consecutive_count = 0
                    pending_change = None
                    change_start_idx = None
                    predictions[i] = self.last_prediction
        
        return predictions
    
    def predict(self, samples, inference=False):
        """
        Standard prediction without hysteresis.
        
        Parameters:
        samples (dict): Dictionary containing features
        
        Returns:
        np.ndarray: Standard predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, y, _, _ = self.prepare_features(samples, window_average=self.window_average, test_fraction=0, inference=inference, rand_perm=False)
        if inference:
            return self.tree.predict(X)
        return self.tree.predict(X), y
    
    def evaluate(self, samples, use_preprocessing=True):
        """
        Evaluate model performance.
        Parameters:
        samples (dict): Dictionary containing features
        """
        self.use_preprocessing = use_preprocessing
        
        y_pred, y = self.predict(samples, inference=False)

        accuracy = accuracy_score(y, y_pred)

        report = classification_report(y, y_pred, target_names=['magnetosheath', 'magnetotail'], output_dict=True)

        return accuracy, report


    
    def reset_hysteresis(self):
        """Reset the hysteresis state."""
        self.last_prediction = None
    
    def show_results_section(self, samples, dataset, section_key, crossings=None, use_hysteresis=False, plot=True, figsize=(12, 12), legacy_predict=False, plot_variables=False):
        """
        Plot the predicted regions vs true regions with two subplots.
        
        Parameters:
        samples (dict): Dictionary containing features and true labels
        dataset (MagnetotailDataset): Dataset object to get times
        section_key (str): Key to access the specific section in the dataset
        crossings (list): List of datetime objects where boundary crossings occur
        use_hysteresis (bool): Whether to use hysteresis in prediction
        figsize (tuple): Figure size for the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before showing results")
        
        # Get predictions
        if use_hysteresis:
            y_pred = self.predict_with_hysteresis(samples)
        else:
            y_pred = self.predict(samples, inference=True)
        
        times = dataset.get_section_times()[section_key]



        if legacy_predict:
            # Plot predicted crossings (detect transitions in predictions)
            pred_crossings = []
            for i in range(1, len(y_pred)):
                if y_pred[i] != y_pred[i-1]:
                    pred_crossings.append(times[i])

        else:
            # Alternative prediction method: detect crossings based on region balance within sliding windows
            pred_crossings = []
            window_size = self.window_size
            threshold = self.threshold
            no_overlap = self.no_overlap # Number of timestamps required between crossings
            
            if no_overlap:
                # Non-overlapping window approach
                last_crossing_idx = -no_overlap  # Initialize to allow first window
                
                for i in range(window_size//2, len(y_pred) - window_size//2):
                    # Skip if within no_overlap distance of last crossing
                    if i - last_crossing_idx < no_overlap:
                        continue
                        
                    window_start = i - window_size//2
                    window_end = i + window_size//2
                    window_predictions = y_pred[window_start:window_end]
                    
                    # Count predictions for each region in the window
                    magnetosheath_count = np.sum(window_predictions == 0)
                    magnetotail_count = np.sum(window_predictions == 1)
                    
                    # Check if we have exactly the threshold for each region
                    if magnetosheath_count >= threshold and magnetotail_count >= threshold:
                        # Check if this is a transition point (different regions on either side)
                        left_region = y_pred[window_start:i]
                        right_region = y_pred[i:window_end]
                        
                        left_dominant = 1 if np.sum(left_region == 1) >= len(left_region)/2 else 0
                        right_dominant = 1 if np.sum(right_region == 1) > len(right_region)/2 else 0
                        
                        # If dominant regions are different, mark as crossing
                        if left_dominant != right_dominant:
                            pred_crossings.append(times[i])
                            last_crossing_idx = i  # Update last crossing position
            else:
                # Original overlapping window approach
                for i in range(window_size//2, len(y_pred) - window_size//2):
                    window_start = i - window_size//2
                    window_end = i + window_size//2
                    window_predictions = y_pred[window_start:window_end]
                    
                    # Count predictions for each region in the window
                    magnetosheath_count = np.sum(window_predictions == 0)
                    magnetotail_count = np.sum(window_predictions == 1)
                    
                    # Check if we have exactly the threshold for each region
                    if magnetosheath_count >= threshold and magnetotail_count >= threshold:
                        # Check if this is a transition point (different regions on either side)
                        left_region = y_pred[window_start:i]
                        right_region = y_pred[i:window_end]
                        
                        left_dominant = 1 if np.sum(left_region == 1) >= len(left_region)/2 else 0
                        right_dominant = 1 if np.sum(right_region == 1) > len(right_region)/2 else 0
                        
                        # If dominant regions are different, mark as crossing
                        if left_dominant != right_dominant:
                            pred_crossings.append(times[i] + np.timedelta64(int(5 * (self.window_average - 1) * 60), 's'))

        start_time = times[0]
        end_time = times[-1]
        
        # Plot true crossings (detect transitions in y_true)
        true_crossings = crossings[(crossings > start_time) & (crossings < end_time)] if crossings is not None else []
        

        if plot:
            if plot_variables:
                fig, axs = plt.subplots(6, 1, figsize=figsize, sharex=True)
            else:
                # Create figure with two subplots
                fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

            if plot_variables:
                for i, var in enumerate(self.feature_keys):

                    values = samples[section_key][var].cpu().numpy()
                    
                    if var in ['ion_density', 'ion_avgtemp', 'ion_vthermal']:

                        # Plot the variable
                        axs[i].plot(times, values, label=var, color='black', alpha=0.7)

                    if var == 'ion_density':
                        axs[i].set_ylabel('n$_i$ (cm$^{-3}$)')
                    elif var == 'ion_avgtemp':
                        axs[i].set_ylabel('T$_i$ (eV)')
                    elif var == 'ion_vthermal':
                        axs[i].set_ylabel('v$_{th,i}$ (km/s)')
                    elif var == 'ion_eflux':
                        axs[i].set_ylabel('Ion eflux')

        
                    if var == 'ion_eflux':
                        # Create a meshgrid for the spectrogram
                        T, F = np.meshgrid(times, np.arange(values.shape[1]))
                        pcm = axs[i].pcolormesh(T, F, values.T, shading='auto', cmap='jet', 
                        norm=LogNorm(vmin=1e2, vmax=1e8))
                        
                        # Add colorbar with controlled width using constrained layout approach
                        pos = axs[i].get_position()
                        cbar_ax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
                        fig.colorbar(pcm, cax=cbar_ax, label='eV/(cm^2-s-sr-eV)')

                        axs[i].set_ylabel("Energy Bins (eV)")
            
                        # Set logarithmic energy values for the 31 bins
                        energy_values = np.logspace(np.log10(5), np.log10(25000), values.shape[1])
                        
                        # Define specific energy levels for tick labels
                        target_energies = [1e1, 1e2, 1e3, 1e4]
                        
                        # Find indices closest to target energies
                        tick_indices = []
                        tick_labels = []
                        for target in target_energies:
                            idx = np.argmin(np.abs(energy_values - target))
                            tick_indices.append(idx)
                            # Use the actual target energy for the label, not the closest bin
                            tick_labels.append(f'$10^{int(np.log10(target))}$')

                        # Set ticks and labels
                        axs[i].set_yticks(tick_indices)
                        axs[i].set_yticklabels(tick_labels)
            

            # First subplot: Region comparison
            axs[-2].plot(times, y_pred, 'k', linewidth=2, label='Predicted Regions', alpha=0.7)
            
            # Fill regions for better visualization
            axs[-2].fill_between(times, 0, y_pred, where=(y_pred == 1), alpha=0.3, color='red', label='Predicted Magnetotail')
            
            axs[-2].set_ylabel('Region')
            axs[-2].set_ylim(-0.1, 1.1)
            axs[-2].grid(True, alpha=0.3)
            axs[-2].legend()


                    
            # Create legend handles for crossings
            legend_elements = [Line2D([0], [0], color='gray', linestyle='-', alpha=0.5, label='Boundary Level')]

            if crossings is not None:
                legend_elements.append(Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='True Crossings'))

            legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Predicted Crossings'))

            for crossing in true_crossings:
                axs[-1].axvline(x=crossing, color='blue', linestyle='-', alpha=0.7, linewidth=2)
        
            # Plot predicted crossings
            for crossing in pred_crossings:
                axs[-1].axvline(x=crossing, color='red', linestyle='--', alpha=0.7, linewidth=2)        
            
            # Second subplot: Crossings comparison
            axs[-1].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Boundary Level')
            axs[-1].set_xlabel('Time')
            axs[-1].set_ylabel('Crossings')
            axs[-1].set_ylim(0, 1)
            axs[-1].grid(True, alpha=0.3)
            axs[-1].legend(handles=legend_elements)
        
            if plot_variables:
                plt.subplots_adjust(hspace=0.08)
            plt.show()

        results = {
            'true_crossings': true_crossings,
            'predicted_crossings': pred_crossings
        }
        
        return results
     
    def get_results(self, samples, dataset, crossings=None, use_hysteresis=False, legacy_predict=False):
        """
        Get Results for multiple sections.
        Parameters:
            samples (dict): Dictionary containing sections, with each section containing features and true labels
            dataset (MagnetotailDataset): Dataset object to get times
            crossings (list): List of datetime objects where boundary crossings occur
            use_hysteresis (bool): Whether to use hysteresis in prediction

        Returns:
            dict: Results including accuracy, true crossings, and predicted crossings
        
        """

        all_results = {}
        overall_accuracy = 0
        total_samples = 0

        for section_key in samples.keys():
            # Reset hysteresis for each section
            self.reset_hysteresis()
            
            # Get results for this section without plotting
            section_results = self.show_results_section(
                samples={section_key: samples[section_key]}, 
                dataset=dataset, 
                section_key=section_key, 
                crossings=crossings, 
                use_hysteresis=use_hysteresis, 
                plot=False,
                legacy_predict=legacy_predict
            )
            
            all_results[section_key] = section_results
            
            # Calculate weighted accuracy
            section_samples = len(samples[section_key]['times'])
            total_samples += section_samples

        # Calculate overall accuracy
        if total_samples > 0:
            overall_accuracy /= total_samples

        # Combine all crossings
        all_true_crossings = []
        all_predicted_crossings = []
        for section_results in all_results.values():
            all_true_crossings.extend(section_results['true_crossings'])
            all_predicted_crossings.extend(section_results['predicted_crossings'])

        return {
            'section_results': all_results,
            'true_crossings': all_true_crossings,
            'predicted_crossings': all_predicted_crossings
        }
    
    def get_model_score(self, results, time_window_minutes=30):
        """
        Calculate performance scores based on crossing detection results. Crossings within a specified time window are considered matches.
        
        Parameters:
        results (dict): Results from show_results function containing true_crossings and predicted_crossings
        time_window_minutes (int): Time window in minutes to consider crossings as matches (default: 30)
        
        Returns:
        dict: Dictionary containing TPR, FPR, FNR and related metrics, plus a pandas DataFrame
        """
        true_crossings = results['true_crossings']
        predicted_crossings = results['predicted_crossings']
        
        if len(true_crossings) == 0 and len(predicted_crossings) == 0:
            metrics = {
                'true_positive_rate': 1.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_true_crossings': 0,
                'total_predicted_crossings': 0,
                'tp_time_accuracy_mean_minutes': None,
                'tp_time_accuracy_std_minutes': None
            }
            
            # Create DataFrame for display
            df = pd.DataFrame([metrics])
            metrics['results_dataframe'] = df
            return metrics
        
        # # Convert time window to timedelta
        # time_window = np.timedelta64(time_window_minutes, 'm')
        
        # Convert to numpy datetime64 for easier comparison
        true_crossings_np = np.array([np.datetime64(tc) for tc in true_crossings])
        predicted_crossings_np = np.array([np.datetime64(pc) for pc in predicted_crossings])
        
        # Calculate True Positives: true crossings that have a predicted crossing within time window
        true_positives = 0
        matched_predicted = set()
        tp_time_diffs_minutes = []  # Store time differences for true positives


        true_positives, false_positives, false_negatives, tp_time_diffs_minutes, tp_time_diffs_minutes_signed, in_range_count_true, out_range_count_pred, _ = calculate_metrics_test(
            true_crossings_np, 
            predicted_crossings_np, 
            time_window=time_window_minutes
        )
        
        true_positive_rate, false_positive_rate, false_negative_rate, precision, f1_score, f0_5_score = calculate_rates(true_positives, false_positives, false_negatives)


        ### Old code for one-to-one matching ###
        
        # for true_crossing in true_crossings_np:
        #     if len(predicted_crossings_np) > 0:
        #         # Find closest predicted crossing
        #         time_diffs = np.abs(predicted_crossings_np - true_crossing)
        #         min_diff_idx = np.argmin(time_diffs)
        #         min_diff = time_diffs[min_diff_idx]
                
        #         # Check if within time window and not already matched
        #         if min_diff <= time_window and min_diff_idx not in matched_predicted:
        #             true_positives += 1
        #             matched_predicted.add(min_diff_idx)
        #             # Convert time difference to minutes
        #             min_diff_minutes = min_diff / np.timedelta64(1, 'm')
        #             tp_time_diffs_minutes.append(min_diff_minutes)
        
        # # Calculate False Positives: predicted crossings not matched to any true crossing
        # false_positives = len(predicted_crossings) - len(matched_predicted)
        
        # # Calculate False Negatives: true crossings not matched to any predicted crossing
        # false_negatives = len(true_crossings) - true_positives
        
        # Calculate rates
        total_true = len(true_crossings)
        total_predicted = len(predicted_crossings)
        
        # Calculate mean and std of true positive time accuracy
        tp_time_accuracy_mean = np.mean(tp_time_diffs_minutes) if tp_time_diffs_minutes else None
        tp_time_accuracy_std = np.std(tp_time_diffs_minutes) if tp_time_diffs_minutes else None
        
        metrics = {
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positives': true_positives,
            'in_range_positives': in_range_count_true,
            'out_range_positives': out_range_count_pred,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'f1_score': f1_score,
            'f0_5_score': f0_5_score,
            'total_true_crossings': total_true,
            'total_predicted_crossings': total_predicted,
            'time_window_minutes': time_window_minutes,
            'tp_time_accuracy_mean_minutes': tp_time_accuracy_mean,
            'tp_time_accuracy_std_minutes': tp_time_accuracy_std
        }
        
        # Create a nicely formatted DataFrame for display
        df_data = {
            'Metric': ['True Positive Rate', 'False Positive Rate', 'False Negative Rate', 
                  'True Positives', 'In Range Positives', 'Out Range Predicted Positives', 'False Positives', 'False Negatives',
                  'Precision', 'F1 Score', 'F0.5 Score',
                  'Total True Crossings', 'Total Predicted Crossings',
                  'Time Window (min)', 'TP Time Accuracy Mean (min)', 'TP Time Accuracy Std (min)'],
            'Value': [f"{true_positive_rate:.3f}", f"{false_positive_rate:.3f}", f"{false_negative_rate:.3f}",
                 true_positives, in_range_count_true, out_range_count_pred,false_positives, false_negatives,
                 f"{precision:.3f}", f"{f1_score:.3f}", f"{f0_5_score:.3f}",
                 total_true, total_predicted, time_window_minutes,
                 f"{tp_time_accuracy_mean:.2f}" if tp_time_accuracy_mean is not None else "N/A",
                 f"{tp_time_accuracy_std:.2f}" if tp_time_accuracy_std is not None else "N/A"]
        }
        
        df = pd.DataFrame(df_data)
        metrics['results_dataframe'] = df
        
        return metrics

def calculate_metrics(true_crossings, predicted_crossings, time_window=30):
    """Calculate TP, FP, FN for a set of predicted crossings."""
    time_window = np.timedelta64(time_window, 'm')  # Convert minutes to timedelta64

    
    true_matched = set()
    predicted_matched = set()
    tp_time_diffs_minutes = []
    tp_time_diffs_minutes_signed = []
    
    # Find matches within time window - ensure one-to-one matching with closest pairs
    unmatched_true = list(range(len(true_crossings)))
    unmatched_pred = list(range(len(predicted_crossings)))
    
    # Create list of all valid matches within time window
    valid_matches = []
    for i in unmatched_true:
        for j in unmatched_pred:
            time_diff = abs(true_crossings[i] - predicted_crossings[j])
            if time_diff <= time_window:
                valid_matches.append((i, j, time_diff))

    # Create a list of all true crossings that are within the time window of predicted crossings
    in_range = []
    for i in unmatched_true:
        time_diffs = []
        for j in unmatched_pred:
            time_diffs.append(abs(predicted_crossings[j] - true_crossings[i]))

        if any(time_diffs <= time_window):
            in_range.append(i)

    
    # Create a list of all predicted crossings that are outside the time window of any true crossing
    out_range = []
    for j in unmatched_pred:
        time_diffs = []
        for i in unmatched_true:
            time_diffs.append(abs(predicted_crossings[j] - true_crossings[i]))

        if all(time_diffs > time_window):
            out_range.append(j)
    
    # Sort matches by time difference (closest first)
    valid_matches.sort(key=lambda x: x[2])
    
    # Assign matches greedily, ensuring one-to-one mapping
    for true_idx, pred_idx, time_diff in valid_matches:
        if true_idx in unmatched_true and pred_idx in unmatched_pred:
            true_matched.add(true_idx)
            predicted_matched.add(pred_idx)
            unmatched_true.remove(true_idx)
            unmatched_pred.remove(pred_idx)
            tp_time_diffs_minutes.append(time_diff / np.timedelta64(1, 'm'))  # Convert to minutes
            tp_time_diffs_minutes_signed.append((predicted_crossings[pred_idx] - true_crossings[true_idx]) / np.timedelta64(1, 'm'))  # Signed difference in minutes
    
    tp = len(true_matched)
    fp = len(predicted_crossings) - len(predicted_matched)
    fn = len(true_crossings) - len(true_matched)
    in_range_count_true = len(in_range)
    out_range_count_pred = len(out_range)
    
    return tp, fp, fn, tp_time_diffs_minutes, tp_time_diffs_minutes_signed, in_range_count_true, out_range_count_pred

def calculate_metrics_test(true_crossings, predicted_crossings, time_window=30, method='hungarian'):
    """Calculate TP, FP, FN for a set of predicted crossings."""
    time_window = np.timedelta64(time_window, 'm')  # Convert minutes to timedelta64

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

def moving_average_with_padding(data, window_size):
    """
    Apply moving average to data while maintaining the same length by padding.
    
    Args:
        data: torch.Tensor - input data
        window_size: int - size of the moving average window
    
    Returns:
        torch.Tensor - smoothed data with same length as input
    """
    if window_size <= 1:
        return data
    
    # Convert to numpy for easier handling
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data
    
    # Apply moving average using convolution
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data_np, kernel, mode='same')
    
    # Convert back to torch tensor if input was tensor
    if isinstance(data, torch.Tensor):
        return torch.tensor(smoothed, dtype=data.dtype)
    return smoothed
