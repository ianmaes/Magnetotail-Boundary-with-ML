import sys
import os
from pathlib import Path
sys.path.append(str(Path().resolve()))
sys.path.append(str(Path().resolve().parent))
sys.path.append(os.path.dirname(__file__))  # adds current folder

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional
from pathlib import Path
from Data.data_class import plot_electron_spectrogram
from IPython.display import clear_output
from tqdm import tqdm

from Helper.preprocess_functions import remove_outliers_with_local_interpolation, downweight_variable, remove_outliers_with_interpolation

class MagnetotailDataset(Dataset):
    """
    PyTorch Dataset for loading magnetotail data from HDF5 files generated 
    by the extract_to_HDF5 function.
    """
    
    def __init__(self, 
                 hdf5_file: str,
                 session_ids: Optional[List[str]] = None,
                 variables: Optional[List[str]] = None,
                 transform=None):
        """
        Initialize the dataset.
        
        Parameters:
            hdf5_file (str): Path to the HDF5 file containing the data.
            session_ids (List[str], optional): List of specific session IDs to load.
                                              If None, loads all sessions.
            variables (List[str], optional): List of variable names to load. 
                                            If None, loads all variables.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.hdf5_file = hdf5_file
        self.transform = transform
        
        # Load metadata and prepare index mapping
        self.sessions = []
        self.index_mapping = []
        self._load_metadata(session_ids, variables)
        self.trainable_indices = np.array(self.index_mapping)
        self.initial_cutoffs = []
        self.final_cutoffs = []
        self.trainable_samples = None
        self.skip_calculations = False
        # Load section to region map if it exists
        self.section_to_region_map = self._load_section_to_region_map()

        # Initialize sections
        self.set_trainable_samples(write_to_file=False)

        if self.skip_calculations:
            print("Skipping further calculations due to previous warnings.")
            return

        # Compute plasma beta for each sample
        self._compute_plasma_beta()
        self._extract_magnetic_field_x()
        self._compute_magnetic_field_magnitude()
        self._compute_velocity_magnitude()
        self._extract_velocity_x()
        
    def _compute_plasma_beta(self):
        """
        Compute plasma beta for each sample in the dataset.
        
        Returns:
            List of plasma beta values for each sample, and stores the results as a new variable in the dataset.
        """

        trainable_samples = self.get_trainable_samples()

        print("Computing plasma beta for each section...")

        for section_key, section_data in tqdm(trainable_samples.items()):
            # Check if necessary variables are present
            if 'ion_avgtemp' in section_data and 'electron_avgtemp' in section_data and 'ion_density' in section_data and 'electron_density' in section_data:
                
                # Get the necessary variables and convert to SI units
                ion_avgtemp      = section_data['ion_avgtemp'] * 1.60218e-19         # eV to SI units (J)
                electron_avgtemp = section_data['electron_avgtemp'] * 1.60218e-19    # eV to SI units (J)
                ion_density      = section_data['ion_density'] * 10 ** 6             # n per cm^-3 to SI units (m^-3)
                electron_density = section_data['electron_density'] * 10 ** 6        # n per cm^-3 to SI units (m^-3)

                # Remove outliers of all variables
                ion_avgtemp      = remove_outliers_with_local_interpolation(ion_avgtemp, 2)
                electron_avgtemp = remove_outliers_with_local_interpolation(electron_avgtemp, 2)
                ion_density      = remove_outliers_with_local_interpolation(ion_density, 2)
                electron_density = remove_outliers_with_local_interpolation(electron_density, 2)

                # Get the total magnetic field strength in Tesla
                if 'magnetic_field_gsm' in section_data:

                    # Get magnetic field in GSM coordinates
                    magnetic_field = section_data['magnetic_field_gsm']

                    # Compute total magnetic field strength: |B| = sqrt(Bx^2 + By^2 + Bz^2)
                    B_total = torch.sqrt(torch.sum(magnetic_field**2, dim=-1)) * 1e-9  # Convert from nT to T

                    # Remove outliers in the magnetic field data
                    B_total = remove_outliers_with_local_interpolation(B_total, 2)

                    # Downweight the magnetic field variable to reduce noise
                    B_total, _, _ = downweight_variable(B_total, p=2.0, win=5, smoother_win=15, mode='rational', wmin=0.01, use_forward_diff=True)

                    # Minimum threshold for B_total in Tesla
                    min_B = 3e-9  

                    # Set a minimum threshold for B_total to avoid division by zero
                    B_total = torch.where(B_total < min_B, torch.tensor(min_B, dtype=B_total.dtype, device=B_total.device), B_total)


                μ_0 = 4 * np.pi * 1e-7  # Permeability of free space in T*m/A
                k_b = 1.380649e-23      # Boltzmann constant in J/K (unused here, but included for completeness)

                # Calculate plasma beta: β = n * T / (B^2 / (2 * μ_0))
                plasma_beta = (ion_density * ion_avgtemp + electron_density * electron_avgtemp) / (B_total**2 / (2 * μ_0))


                if len(plasma_beta) == 1:
                    print(f"Warning: Plasma beta calculation resulted in a single value for section {section_key}. This may indicate insufficient data points.")

                # Remove outliers in the magnetic field data
                plasma_beta = remove_outliers_with_local_interpolation(plasma_beta, window_size=50, n_std=2)

                # Store plasma beta in the section data
                section_data['plasma_beta'] = plasma_beta

                # Downweight plasma beta to reduce noise for logged version
                plasma_beta, _, _ = downweight_variable(plasma_beta, kappa_pct=80, p=2.0, smoother_win=20, mode='rational', wmin=0.01, use_forward_diff=True)

                # Store logged plasma beta in the section data
                section_data['log_plasma_beta'] = torch.log10(plasma_beta + 1e-10)

            else:
                print(f"Missing required variables for plasma beta calculation in section {section_key}. Skipping this section.")
                section_data['plasma_beta'] = None

    def _extract_magnetic_field_x(self):
        """
        Extract the x-component of the magnetic field in GSM coordinates for each sample.
        
        Returns:
            List of magnetic field x-component values for each sample, and stores the results as a new variable in the dataset.
        """

        trainable_samples = self.get_trainable_samples()

        print("Extracting magnetic field x-component for each section...")

        for section_key, section_data in tqdm(trainable_samples.items()):
            # Check if magnetic_field_gsm variable is present
            if 'magnetic_field_gsm' in section_data:

                # Get magnetic field in GSM coordinates
                magnetic_field = section_data['magnetic_field_gsm']

                # Extract the x-component (first column)
                Bx = magnetic_field[:, 0]

            # Store Bx in the section data
            section_data['magnetic_field_gsm_x'] = torch.abs(Bx)

    def _extract_magnetic_field_y(self):
        """
        Extract the y-component of the magnetic field in GSM coordinates for each sample.
        
        Returns:
            List of magnetic field y-component values for each sample, and stores the results as a new variable in the dataset.
        """

        trainable_samples = self.get_trainable_samples()

        print("Extracting magnetic field y-component for each section...")

        for section_key, section_data in tqdm(trainable_samples.items()):
            # Check if magnetic_field_gsm variable is present
            if 'magnetic_field_gsm' in section_data:

                # Get magnetic field in GSM coordinates
                magnetic_field = section_data['magnetic_field_gsm']

                # Extract the y-component (second column)
                By = magnetic_field[:, 1]

                # Store By in the section data
                section_data['magnetic_field_gsm_y'] = By

            else:
                print(f"Missing 'magnetic_field_gsm' variable in section {section_key}. Skipping this section.")
                section_data['magnetic_field_gsm_y'] = None

    def _extract_magnetic_field_z(self):
        """
        Extract the z-component of the magnetic field in GSM coordinates for each sample.
        
        Returns:
            List of magnetic field z-component values for each sample, and stores the results as a new variable in the dataset.
        """

        trainable_samples = self.get_trainable_samples()

        print("Extracting magnetic field z-component for each section...")

        for section_key, section_data in tqdm(trainable_samples.items()):
            # Check if magnetic_field_gsm variable is present
            if 'magnetic_field_gsm' in section_data:

                # Get magnetic field in GSM coordinates
                magnetic_field = section_data['magnetic_field_gsm']

                # Extract the z-component (third column)
                Bz = magnetic_field[:, 2]

                # Store Bz in the section data
                section_data['magnetic_field_gsm_z'] = Bz

            else:
                print(f"Missing 'magnetic_field_gsm' variable in section {section_key}. Skipping this section.")
                section_data['magnetic_field_gsm_z'] = None

    def _compute_magnetic_field_magnitude(self):
        """
        Compute the magnitude of the magnetic field in GSM coordinates for each sample.
        
        Returns:
            List of magnetic field magnitude values for each sample, and stores the results as a new variable in the dataset.
        """

        trainable_samples = self.get_trainable_samples()

        print("Computing magnetic field magnitude for each section...")

        for section_key, section_data in tqdm(trainable_samples.items()):
            # Check if magnetic_field_gsm variable is present
            if 'magnetic_field_gsm' in section_data:

                # Get magnetic field in GSM coordinates
                magnetic_field = section_data['magnetic_field_gsm']

                # Compute total magnetic field strength: |B| = sqrt(Bx^2 + By^2 + Bz^2)
                B_total = torch.sqrt(torch.sum(magnetic_field**2, dim=-1))

                # Store B_total in the section data
                section_data['magnetic_field_gsm_magnitude'] = B_total

            else:
                print(f"Missing 'magnetic_field_gsm' variable in section {section_key}. Skipping this section.")
                section_data['magnetic_field_gsm_magnitude'] = None
 
    def _compute_velocity_magnitude(self):
        
        trainable_samples = self.get_trainable_samples()

        print("Computing ion velocity magnitude for each section...")

        for section_key, section_data in tqdm(trainable_samples.items()):
            # Check if magnetic_field_gsm variable is present
            if 'ion_velocity_gsm' in section_data:

                # Get ion velocity in GSM coordinates
                ion_velocity = section_data['ion_velocity_gsm']

                # Compute total magnetic field strength: |B| = sqrt(Bx^2 + By^2 + Bz^2)
                velocity = torch.sqrt(torch.sum(ion_velocity**2, dim=-1))

                # Store B_total in the section data
                section_data['ion_velocity_magnitude'] = velocity

            else:
                print(f"Missing 'ion_velocity_gsm' variable in section {section_key}. Skipping this section.")
                section_data['ion_velocity_magnitude'] = None
 
    def _extract_velocity_x(self):
        """
        Extract the x-component of the ion velocity in GSM coordinates for each sample.
        
        Returns:
            List of ion velocity x-component values for each sample, and stores the results as a new variable in the dataset.
        """

        trainable_samples = self.get_trainable_samples()

        print("Extracting ion velocity x-component for each section...")

        for section_key, section_data in tqdm(trainable_samples.items()):
            # Check if ion_velocity_gsm variable is present
            if 'ion_velocity_gsm' in section_data:

                # Get ion velocity in GSM coordinates
                ion_velocity = section_data['ion_velocity_gsm']

                # Extract the x-component (first column)
                Vx = ion_velocity[:, 0]

                # Store Vx in the section data
                section_data['ion_velocity_gsm_x'] = Vx

            else:
                print(f"Missing 'ion_velocity_gsm' variable in section {section_key}. Skipping this section.")
                section_data['ion_velocity_gsm_x'] = None
        

    def _load_section_to_region_map(self, filepath: Optional[str] = None):
        
        if filepath is not None:
            # Load from specified file stored as a numpy .npz file
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File {filepath} not found.")
            data = np.load(filepath, allow_pickle=True)
            section_to_region_map = {key: data[key].item() if data[key].shape == () else data[key].tolist() for key in data.files}
            return section_to_region_map
        
        with h5py.File(self.hdf5_file, 'r') as h5f:
            # Check if section to region mapping exists
            if 'section_to_region_map' in h5f.keys():
                
                # Read the section to region mapping group
                section_group = h5f['section_to_region_map']
                section_to_region_map = {}

                # Iterate through each key in the section group
                for k, v in section_group.items():

                    # Read dataset content; v[...] reads the data into memory
                    data = v[()]

                    # Check if the data is scalar or array and decode accordingly
                    if isinstance(data, bytes):
                        section_to_region_map[k] = data.decode('utf-8')

                    # if it is an array of bytes, decode each element
                    elif isinstance(data, (list, tuple, np.ndarray)):
                        section_to_region_map[k] = [item.decode('utf-8') for item in data]

                    else:
                        raise ValueError(f"Unexpected data type {type(data)} for key '{k}'")
                    
            # If no section to region mapping exists, return an empty dict
            else:
                section_to_region_map = {} 

            

        return section_to_region_map
    
    def write_section_to_region_map(self):
        with h5py.File(self.hdf5_file, 'r+') as h5f:

            # If section_to_region_map is not set, try to get it
            if not self.section_to_region_map:
                self.section_to_region_map = self.get_section_to_region_map()

            

            # Create a group for section to region mapping
            if 'section_to_region_map' not in h5f.keys():
                section_group = h5f.create_group('section_to_region_map')
            else:
                section_group = h5f['section_to_region_map']
                # Clear existing entries
                for key in list(section_group.keys()):
                    del section_group[key]
            
            # Write each section and its corresponding region
            for section_key, region in self.section_to_region_map.items():
                if region is None:
                    region = 'None'  # Handle None regions
                section_group.create_dataset(section_key, data=region.encode('utf-8'))

    def save_section_to_region_map(self, filepath: str):
        """
        Save the section to region mapping to a specified file in NumPy .npz format.
        
        Parameters:
            filepath (str): Path to the file where the mapping should be saved.
        """
        if not self.section_to_region_map:
            raise ValueError("No section to region mapping found. Please set regions manually or ensure the mapping exists.")
        
        # Save the mapping as a .npz file
        np.savez(filepath, **self.section_to_region_map)

    def _load_metadata(self, session_ids=None, variables=None):
        """Load metadata from the HDF5 file and prepare index mapping."""
        with h5py.File(self.hdf5_file, 'r+') as h5f:
            # Get available session IDs
            available_sessions = [k for k in h5f.keys() if k.startswith('session_')]
            
            # Filter sessions if specified
            if session_ids is not None:
                available_sessions = [s for s in available_sessions if s in session_ids]
            
            # Process each session
            for session_id in available_sessions:

                # Check if 'times' variable exists, if not, remove the session
                # This is to ensure that we only work with sessions that have valid timestamps
                while 'times' not in h5f[session_id]:
                    
                    # If 'times' is missing, print a warning and remove the session
                    print(f"Warning: Session does not contain 'times' variable. Removing it, and re-ordering.")

                    # Remove the session from the dataset
                    del h5f[session_id]

                    # Remove from in-memory sessions list
                    self.resort_sessions()

                    # Re-load available sessions after deletion
                    available_sessions = [k for k in h5f.keys() if k.startswith('session_')]

                # Access the session group
                session_group = h5f[session_id]    
                                
                # Get timestamps for this session
                times = session_group['times'][:]
                
                # Get variables in this session
                session_vars = [k for k in session_group.keys()]

                # Filter variables if specified
                if variables is not None:
                    session_vars = [v for v in session_vars if v in variables]
                
                # Skip session if no variables left after filtering
                if not session_vars:
                    continue
                
                # Store session metadata
                self.sessions.append({
                    'id': session_id,
                    'start_time': session_group.attrs.get('start_time', ''),
                    'end_time': session_group.attrs.get('end_time', ''),
                    'data_origin': session_group.attrs.get('data_origin', 'Unknown'),
                    'num_timestamps': len(times),
                    'variables': session_vars
                })
                
                # Create index mapping (session_idx, timestamp_idx) for each data point
                for i in range(len(times)):
                    self.index_mapping.append((len(self.sessions) - 1, i))
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
            idx (int or tuple): If int, index of the sample to retrieve.
                               If tuple (session_idx, time_idx), gets specific session and time.
            
        Returns:
            dict: Contains 'session_id', and a key for each variable
                 with its corresponding value at the time index.
        """
        if isinstance(idx, tuple) and len(idx) == 2:
            # Handle direct (session_idx, time_idx) access
            session_idx, time_idx = idx
        else:
            # Handle normal index access through the mapping
            session_idx, time_idx = self.index_mapping[idx]
        
        session_info = self.sessions[session_idx]
        session_id = session_info['id']
        variables = session_info['variables']
        
        with h5py.File(self.hdf5_file, 'r') as h5f:
            session_group = h5f[session_id]
            
            # Get data for all variables at this time index
            data = {}
            for var_name in variables:
                data[var_name] = torch.tensor(session_group[var_name][time_idx], dtype=torch.float32)
            
            # Create sample
            sample = {
                'session_id': session_id,
                **data
            }
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
        
    def remove_session(self, session_id: str):
        """
        Remove a session from the dataset and delete it from the HDF5 file.
                
        Parameters:
            session_id (str): The session ID to remove (e.g., 'session_001')
        """
        # First, remove from in-memory structures
        session_idx_to_remove = None
        for i, session in enumerate(self.sessions):
            if session['id'] == session_id:
                session_idx_to_remove = i
                break

        if session_idx_to_remove is None:
            raise ValueError(f"Session {session_id} not found in dataset")

        # Remove session from sessions list
        self.sessions.pop(session_idx_to_remove)

        # Update index mapping to remove entries for this session and adjust indices
        new_index_mapping = []
        for session_idx, time_idx in self.index_mapping:
            if session_idx < session_idx_to_remove:
                # Sessions before the removed one keep their indices
                new_index_mapping.append((session_idx, time_idx))
            elif session_idx > session_idx_to_remove:
                # Sessions after the removed one get decremented indices
                new_index_mapping.append((session_idx - 1, time_idx))
            # Skip entries for the removed session (session_idx == session_idx_to_remove)

        self.index_mapping = new_index_mapping

        # Update trainable indices similarly
        new_trainable_indices = []
        for session_idx, time_idx in self.trainable_indices:
            if session_idx < session_idx_to_remove:
                new_trainable_indices.append((session_idx, time_idx))
            elif session_idx > session_idx_to_remove:
                new_trainable_indices.append((session_idx - 1, time_idx))

        self.trainable_indices = np.array(new_trainable_indices)

        # Remove from HDF5 file
        with h5py.File(self.hdf5_file, 'r+') as h5f:
            if session_id in h5f:
                del h5f[session_id]

    def resort_sessions(self):
        """
        Resort sessions to remove gaps in session numbering after session removal.
        This function renumbers sessions sequentially starting from 'session_0001'.
        """
        with h5py.File(self.hdf5_file, 'r+') as h5f:
            # Get all remaining session keys and sort them
            session_keys = [k for k in h5f.keys() if k.startswith('session_')]
            session_keys.sort()
            
            # Create mapping from old to new session names
            old_to_new = {}
            for i, old_key in enumerate(session_keys):
                new_key = f"session_{i:04d}"
                old_to_new[old_key] = new_key
            
            # Rename sessions in HDF5 file if needed
            for old_key, new_key in old_to_new.items():
                if old_key != new_key:
                    # Create new group with new name
                    h5f.copy(old_key, new_key)
                    # Delete old group
                    del h5f[old_key]
            
            # Update session IDs in memory
            for i, session in enumerate(self.sessions):
                old_id = session['id']
                new_id = f"session_{i:04d}"
                session['id'] = new_id

    def get_cutoff_times(self):
        """Return the initial and final cutoffs for exclusion."""
        return self.initial_cutoffs, self.final_cutoffs
    
    def get_variable_names(self):
        """Return a list of all variable names in the dataset."""
        variable_set = set()
        for session in self.sessions:
            variable_set.update(session['variables'])
        return sorted(list(variable_set))
    
    def get_session_info(self):
        """Return information about all sessions in the dataset."""
        return self.sessions
    
    def exclude_time_ranges(self, initial_cutoffs: List[str], final_cutoffs: List[str]):
        """
        Remove trainable indices that fall within specified time ranges.
        
        Parameters:
            initial_cutoffs (List[str]): List of start times for exclusion ranges (ISO format)
            final_cutoffs (List[str]): List of end times for exclusion ranges (ISO format)
        """
        if len(initial_cutoffs) != len(final_cutoffs):
            raise ValueError("initial_cutoffs and final_cutoffs must have the same length")
        
        # Start with all current trainable indices
        valid_indices = []
        
        for idx_tuple in self.trainable_indices:
            session_idx, time_idx = idx_tuple
            
            # Get the timestamp for this index
            with h5py.File(self.hdf5_file, 'r') as h5f:
                session_id = self.sessions[session_idx]['id']
                timestamp = h5f[session_id]['times'][time_idx]
            
            timestamp_tensor = torch.tensor(timestamp)
            
            # Check if this timestamp falls within any exclusion range
            keep_index = True
            for initial_cutoff, final_cutoff in zip(initial_cutoffs, final_cutoffs):
                cutoff_start = torch.tensor(np.datetime64(initial_cutoff).astype('datetime64[ns]').astype('int64'))
                cutoff_end = torch.tensor(np.datetime64(final_cutoff).astype('datetime64[ns]').astype('int64'))
                
                # If timestamp is within this exclusion range, don't keep it
                if cutoff_start <= timestamp_tensor <= cutoff_end:
                    keep_index = False
                    break
            
            if keep_index:
                valid_indices.append(idx_tuple)
        
        self.initial_cutoffs.append(initial_cutoffs)
        self.final_cutoffs.append(final_cutoffs)
        self.trainable_indices = np.array(valid_indices)
        
    def get_trainable_indices(self):
        """Return the indices of trainable samples."""
        return self.trainable_indices
    
    def _compute_sections(self):
        """Compute and store trainable indices grouped by continuous sections."""
        
        self.sections = {}

        if len(self.trainable_indices) == 0:
            return
            
        # Group trainable indices by session first
        session_groups = {}
        for session_idx, time_idx in self.trainable_indices:
            session_id = self.sessions[session_idx]['id']
            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append((session_idx, time_idx))
        
        # Process each session and split into continuous sections
        section_counter = 0
        
        for session_id, indices in session_groups.items():
            
            # Split into continuous sections based on missing indices
            sections = []
            current_section = [indices[0]]
            
            for i in range(1, len(indices)):
                prev_session_idx, prev_time_idx = indices[i-1]
                curr_session_idx, curr_time_idx = indices[i]
                
                # If there's a gap in time indices, start a new section
                if curr_time_idx != prev_time_idx + 1:
                    sections.append(current_section)
                    current_section = [indices[i]]
                else:
                    current_section.append(indices[i])
            
            # Add the last section
            if current_section:
                sections.append(current_section)

            
            
            # Store each continuous section
            for section_idx, section_indices in enumerate(sections):
                section_key = f"section_{(section_idx + section_counter):03d}"
                self.sections[section_key] = section_indices

            section_counter += len(sections)

    def set_trainable_samples(self, write_to_file=False):
        if not hasattr(self, 'sections'):
                self._compute_sections()
            
        result = {}

        print("Setting trainable samples for each section...")

        # Pre-open the HDF5 file to avoid repeated file operations
        with h5py.File(self.hdf5_file, 'r') as h5f:
            for section_key, section_indices in tqdm(self.sections.items()):
                if not section_indices:
                    continue
                
                # Get session info for this section (all indices should be from same session)
                session_idx, _ = section_indices[0]
                session_info = self.sessions[session_idx]
                session_id = session_info['id']
                variables = session_info['variables']
                session_group = h5f[session_id]
                
                # Extract time indices for this section
                time_indices = [time_idx for _, time_idx in section_indices]
                
                # Initialize section result with variables (exclude 'region')
                section_result = {}
                
                # Check whether all variables are of the same length
                expected_length = session_info['num_timestamps']
                skipped_variables = []
                
                # Load data for each variable efficiently using array slicing
                for var_name in variables:
                    actual_length = session_group[var_name].shape[0]
                    if actual_length != expected_length:
                        print(f"Warning: Variable '{var_name}' has length {actual_length} but expected {expected_length} in session '{session_id}'. Skipping variable.")
                        skipped_variables.append(var_name)
                        self.skip_calculations = True
                        continue
                        
                    var_data = session_group[var_name][time_indices]
                    section_result[var_name] = torch.tensor(var_data, dtype=torch.float32)
                
                # Print summary of skipped variables if any
                if skipped_variables:
                    print(f"Skipped {len(skipped_variables)} variables in section '{section_key}': {skipped_variables}")
                
                # Add session_id and region
                section_result['session_id'] = session_id
                section_result['region'] = None
                
                result[section_key] = section_result

        self.trainable_samples = result

        # if write_to_file:
        #     # Store section keys with section indices
        #     with h5py.File(self.hdf5_file, 'w') as h5f:

                
        #         if session_id in h5f:
        #             del h5f[session_id]

        
        

        return result

    def get_trainable_samples(self):
        """Return all trainable samples grouped by continuous sections."""
        # If trainable_samples is already set, return it

        if self.trainable_samples is not None:
            return self.trainable_samples

        # If not set, compute and return
        else:
            result = self.set_trainable_samples()
            
            return result

    def set_regions_for_sections(self, overwrite=True, filepath: Optional[str] = None):
        """
        Set the 'region' key for each section in trainable samples.
        
        Parameters:
            regions (Dict[str, str]): Mapping of section keys to region names.
        """

        # If not section to region map is given, manual assignment is performed
        if overwrite:
            for section in self.trainable_samples:

                # Plot the variable containing ion_eflux 
                for key in self.trainable_samples[section].keys():
                    if 'ion_eflux' in key:
                        clear_output(wait=True)  # Clear previous output
                        plot_electron_spectrogram(self.trainable_samples[section][key].numpy(),
                                                    self.trainable_samples[section]['times'].numpy(),
                                                    vmin=1e2,
                                                    vmax=1e8)

                # Gather input from user
                region = input(f"Enter region for section {section}, options are [1 - magnetotail, 2 - magnetosheath]: ")

                if region not in ['magnetotail', 'magnetosheath', '1', '2']:
                    print("Invalid input. Please enter '1' for magnetotail or '2' for magnetosheath.")

                if region in ['1', 'magnetotail']:
                    self.trainable_samples[section]['region'] = 'magnetotail'
                    self.section_to_region_map[section] = 'magnetotail'
                elif region in ['2', 'magnetosheath']:    
                    self.trainable_samples[section]['region'] = 'magnetosheath'
                    self.section_to_region_map[section] = 'magnetosheath'
                else: 
                    self.trainable_samples[section]['region'] = None
                    self.section_to_region_map[section] = None
                    print('Due to invalid input, region was set to None')

        # Assignment using section to region map
        else:
            section_to_region_map = self.get_section_to_region_map(filepath)

            if not section_to_region_map:
                raise ValueError("No section to region mapping found. Please set regions manually or ensure the mapping exists.")

            for section_key in section_to_region_map.keys():
                self.trainable_samples[section_key]['region'] = section_to_region_map[section_key]

    def get_section_to_region_map(self, filepath: Optional[str] = None):

        """
        Get the section to region map.
        """

        if not hasattr(self, 'section_to_region_map'):
            self.section_to_region_map = self._load_section_to_region_map(filepath)
        
        return self.section_to_region_map

    def get_section_times(self):
        """
        Get the times variable for each section in trainable samples as numpy.datetime64 array, for plotting purposes.

        Returns:
            Dict[str, np.ndarray]: Mapping of section keys to their corresponding times as numpy.datetime64 arrays.
        """

        # Set up dictionary to hold times for each section
        times_np_dict = {}

        # Loop through each section
        for section_key in self.trainable_samples.keys():

            # Extract times tensor
            times_tensor = self.trainable_samples[section_key]['times']

            # Convert to numpy datetime64 array
            times_np = times_tensor.numpy().astype('datetime64[ns]')
            
            # Store in dictionary
            times_np_dict[section_key] = times_np

        return times_np_dict

    def delete_section(self, section_key, section_is_session=False):
        """
        Delete a section from the dataset and remove it from the HDF5 file. Delete it from the file, trainable indices, and trainable samples.
        Parameters:
            section_key (str): The section key to delete (e.g., 'section_001')
        """

        # Remove section from HDF5 file if stored
        if section_is_session:
            # From section key, get session id  
            session_id = self.trainable_samples[section_key]['session_id']
            print(f"Deleting entire session {session_id} from dataset.")
            with h5py.File(self.hdf5_file, 'r+') as h5f:
                if session_id in h5f:
                    del h5f[session_id]
   


            if section_key not in self.sections:
                raise ValueError(f"Section {section_key} not found in dataset")

            # Get indices for this section
            section_indices = self.sections[section_key]
            section_indices_set = set(section_indices)
            # Remove these indices from trainable_indices
            new_trainable_indices = [idx for idx in self.trainable_indices if tuple(idx) not in section_indices_set]

            self.trainable_indices = np.array(new_trainable_indices)

            # Remove section from sections and trainable_samples
            del self.sections[section_key]

            if self.trainable_samples and section_key in self.trainable_samples:
                del self.trainable_samples[section_key]

            # Remove section from section_to_region_map if it exists
            if self.section_to_region_map and section_key in self.section_to_region_map:
                del self.section_to_region_map[section_key]

        else:
            with h5py.File(self.hdf5_file, 'r+') as h5f:
                if section_key in h5f:
                    del h5f[section_key]
                    