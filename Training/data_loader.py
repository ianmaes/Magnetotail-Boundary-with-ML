import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
from Data.data_class import plot_electron_spectrogram
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

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
        
    def _load_metadata(self, session_ids=None, variables=None):
        """Load metadata from the HDF5 file and prepare index mapping."""
        with h5py.File(self.hdf5_file, 'r') as h5f:
            # Get available session IDs
            available_sessions = [k for k in h5f.keys() if k.startswith('session_')]
            
            # Filter sessions if specified
            if session_ids is not None:
                available_sessions = [s for s in available_sessions if s in session_ids]
            
            # Process each session
            for session_id in available_sessions:
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
        This function renumbers sessions sequentially starting from 'session_001'.
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

        for section_key, section_indices in self.sections.items():
            # Get the first sample to determine keys
            sample = self[section_indices[0]]

            # Create a list of keys excluding 'region'
            sample_keys = list(sample.keys())
            sample_keys.remove('region') if 'region' in sample_keys else None # Remove 'region' if it exists

            # Initialize a dictionary to hold results for this section
            section_result = {key: [] for key in sample_keys}

            # Collect all values for each key in this section
            for idx in section_indices:
                sample = self[idx]
                for key in section_result.keys():
                    section_result[key].append(sample[key])
            
            # Convert lists to tensors for each section
            for key in section_result.keys():
                if isinstance(section_result[key][0], torch.Tensor):
                    section_result[key] = torch.stack(section_result[key])
                else:
                    section_result[key] = section_result[key]  # Keep as list for non-tensor data
                        # Add 'region' key to section result dict

            section_result['region'] = None

            result[section_key] = section_result

        self.trainable_samples = result

        if write_to_file:
            # Store section keys with section indices
            with h5py.File(self.hdf5_file, 'w') as h5f:

                
                if session_id in h5f:
                    del h5f[session_id]

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

    def set_regions_for_sections(self, section_to_region_map=None):
        """
        Set the 'region' key for each section in trainable samples.
        
        Parameters:
            regions (Dict[str, str]): Mapping of section keys to region names.
        """

        # If not section to region map is given, manual assignment is performed
        if section_to_region_map == None:
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
                elif region in ['2', 'magnetosheath']:    
                    self.trainable_samples[section]['region'] = 'magnetosheath'
                else: 
                    self.trainable_samples[section]['region'] = None
                    print('Due to invalid input, region was set to None')

        # Assignment using section to region map
        else:
            for section_key in section_to_region_map.keys():
                self.trainable_samples[section_key] = section_to_region_map[section_key]

    def get_section_to_region_map(self):
        """
        Get the section to region map.
        """

        # Extract section to region map
        section_to_region_map = dict()

        # Loop through sections of the trainable samples
        for section_key in self.trainable_samples.keys():

            # Get the region for each section
            section_to_region_map[section_key] = self.trainable_samples[section_key]['region']

        return section_to_region_map
