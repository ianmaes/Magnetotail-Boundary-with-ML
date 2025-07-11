import sys
from pathlib import Path

# Append the parent directory of 'Data' to sys.path
sys.path.append(str(Path().resolve().parent))  # adjust as needed

import os
import h5py
import pyspedas
from Data import data_cleaning
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy
from matplotlib.colors import LogNorm
from datetime import datetime, timezone

def plot_electron_spectrogram(electron_data, times=None, energy=None, figsize=(10, 4), vmin=None, vmax=None, norm='Log', title=None):

    """
    Plots an electron energy-time spectrogram using the provided electron flux data, time stamps, and energy bins.
    Parameters
    ----------
    electron_data : np.ndarray
        2D array of electron flux values, with shape (n_times, n_energies).
    times : np.ndarray, optional
        1D array of time stamps (numpy datetime64 or similar), length n_times. If None, indices will be used.
    energy : np.ndarray, optional
        2D array of energy bin values, with shape (n_times, n_energies). If None, indices will be used.
    figsize : tuple, optional
        Size of the matplotlib figure (default is (10, 4)).
    vmin : float, optional
        Minimum value for color normalization (not currently used).
    vmax : float, optional
        Maximum value for color normalization (not currently used).
    """

    # Energy and flux data
    E = electron_data.T
    
    # Handle time data
    if times is not None:
        # Time data
        time_ns = times.astype('int64')
        time_s = time_ns / 1e9

        # Calculate edges for pcolormesh
        t = time_s
        dt = np.diff(t)

        t_edges = np.empty(t.size + 1)
        t_edges[1:-1] = (t[:-1] + t[1:]) / 2
        t_edges[0] = t[0] - dt[0] / 2 if t.size > 1 else t[0] - 0.5
        t_edges[-1] = t[-1] + dt[-1] / 2 if t.size > 1 else t[-1] + 0.5

        # Convert to matplotlib format
        t_edges_plot = mdates.date2num([datetime.fromtimestamp(ts, timezone.utc) for ts in t_edges])
        use_time_axis = True
    else:
        # Use indices instead of time
        n_times = electron_data.shape[0]
        t_edges_plot = np.arange(n_times + 1) - 0.5
        use_time_axis = False

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    pcm = ax.pcolormesh(
        t_edges_plot,
        np.arange(E.shape[0] + 1),
        np.flip(E, axis=0),
        shading='flat',
        cmap='jet',
        norm=LogNorm(vmin=vmin, vmax=vmax) if norm == 'Log' else None,
    )

    cb = fig.colorbar(pcm, ax=ax, label='Electron flux (log scale)')
    
    # Set y-axis labels based on whether energy data is provided
    if energy is not None:
        energy_keV = energy.T[:, 0]
        ax.set_ylabel('Energy (keV)')
        num_yticks = 8  # Set the desired number of y-ticks
        yticks = np.linspace(0, energy_keV.size - 1, num_yticks, dtype=int)
        yticklabels = ["{:.1e}".format(val) for val in np.flip(energy_keV)[yticks]]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_ylabel('Energy Index')
        num_yticks = 8  # Set the desired number of y-ticks
        yticks = np.linspace(0, E.shape[0] - 1, num_yticks, dtype=int)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)

    # Set title if provided
    ax.set_title(title if title else 'Ion energy-time spectrogram\n(pixel width = Δt between measurements)')

    # Set x-axis formatting based on whether times are provided
    if use_time_axis:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))
        fig.autofmt_xdate()
        ax.set_xlabel('Time')
    else:
        ax.set_xlabel('Time Index')

    plt.tight_layout()
    plt.show()


class LoadArtemis():
    def __init__(self, 
                 start_time, 
                 end_time, 
                 instrument_list = ['Particles (space)', 'Electric Fields (space)', 'Magnetic Fields (space)'], 
                 mission_list = ['ARTEMIS'], 
                 satellite_list = ['THB'],
                 local_path = "cdaweb/", 
                 *args, **kwargs):
        
        """
        Load data from CDAWeb for the specified time range and instruments.

        Parameters:
            start_time (str): Start time in the format 'YYYY-MM-DDTHH:MM:SS'.
            end_time (str): End time in the format 'YYYY-MM-DDTHH:MM:SS'.
            instrument_list (list): List of instruments to load data from.
            mission_list (list): List of missions to load data from.
            satellite_list (list): List of satellites to load data from.
            local_path (str): Local path to save downloaded files.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
    
        """
        
        super().__init__(*args, **kwargs)
        
        self.start_time = start_time
        self.end_time = end_time
        self.instrument_list = instrument_list
        self.mission_list = mission_list
        self.data = {}
        self.cdaweb_obj = pyspedas.CDAWeb()
        self.local_path = local_path
        self.satellite_list = satellite_list
        self.name_map = {
            'thb_peef_en_eflux': 'B_electron_eflux',
            'thb_peif_en_eflux': 'B_ion_eflux',
            'thb_peef_vthermal': 'B_electron_vthermal',
            'thb_peif_vthermal': 'B_ion_vthermal',
            'thb_peef_velocity_gsm': 'B_electron_velocity_gsm',
            'thb_peif_velocity_gsm': 'B_ion_velocity_gsm',
            'thb_peef_avgtemp': 'B_electron_avgtemp',
            'thb_peif_avgtemp': 'B_ion_avgtemp',
            'thb_peef_density': 'B_electron_density',
            'thb_peif_density': 'B_ion_density',
            'thb_fgs_gsm': 'B_magnetic_field_gsm',
            'thb_efs_gsm': 'B_electric_field_gsm',
            'thc_peef_en_eflux': 'C_electron_eflux',
            'thc_peif_en_eflux': 'C_ion_eflux',
            'thc_peef_vthermal': 'C_electron_vthermal',
            'thc_peif_vthermal': 'C_ion_vthermal',
            'thc_peef_velocity_gsm': 'C_electron_velocity_gsm',
            'thc_peif_velocity_gsm': 'C_ion_velocity_gsm',
            'thc_peef_avgtemp': 'C_electron_avgtemp',
            'thc_peif_avgtemp': 'C_ion_avgtemp',
            'thc_peef_density': 'C_electron_density',
            'thc_peif_density': 'C_ion_density',
            'thc_fgs_gsm': 'C_magnetic_field_gsm',
            'thc_efs_gsm': 'C_electric_field_gsm',
        }

        dataset_list = self.cdaweb_obj.get_datasets(self.mission_list, self.instrument_list)


        # For all named variables in the name map, create a dictionary entry containg data and metadata
        # with new names

        # Loop through the dataset list and instrument list to find the relevant datasets
        for dataset in dataset_list:
            for instrument in instrument_list:

                # Check if the instrument is in the dataset and if the dataset is in the satellite list
                if (self.instrument_to_dataset(instrument) in dataset) and (dataset[0:3] in self.satellite_list):
                    urllist = self.cdaweb_obj.get_filenames([dataset], start_time, end_time)
                    variables = self.cdaweb_obj.cda_download(urllist, self.local_path)

                    # Loop through the variables in the dataset
                    for var in variables[2]:

                        # Check if the variable is in the name map
                        if var in self.name_map.keys():

                            # Update the variable name using the name map
                            var_new = self.update_naming(var, name_type='new')

                            # Store data and metadata in the 'data' class attribute
                            self.data[var_new] = {
                                'data': self.get_data(var),
                                'metadata': self.get_data(var, metadata=True)
                            }
        
        # Make a copy of the data to reset it later
        self._initial_data = deepcopy(self.data)

    def reset_data(self):
        """
        Reset the data to the initial state.
        """
        self.data = deepcopy(self._initial_data)
        
    def instrument_to_dataset(self, instrument_name):

        # Map instrument names to dataset names
        instrument_to_dataset_map = {
            'Particles (space)': 'ESA',
            'Electric Fields (space)' : 'FIT',
            'Magnetic Fields (space)' : 'FGM',
        }
        
        return instrument_to_dataset_map[instrument_name]

    def update_naming(self, old_name, name_type='new') -> str:
        # Map names from old to new format and vice versa
        if name_type == 'new':
            if old_name in self.name_map.values():
                # If the variable name is already in new format, return it as is
                return old_name
            elif old_name in self.name_map.keys():
                # If the variable name is in old format, find the corresponding new name
                return self.name_map[old_name]
            else:
                raise ValueError(f"Variable name '{old_name}' not found in the mapping.")

        elif name_type == 'old':
            if old_name in self.name_map.keys():
                # If the variable name is already in old format, return it as is
                return old_name
            elif old_name in self.name_map.values():
                # If the variable name is in new format, find the corresponding old name
                return self.legacy_naming(old_name)
            else:
                raise ValueError(f"Variable name '{old_name}' not found in the mapping.")
            
        else:
            raise ValueError(f"Invalid name_type '{name_type}'. Use 'new' or 'old'.")
            
    def legacy_naming(self, new_name):
        # Map new variable names to old one
        for key, value in self.name_map.items():
            if value == new_name:
                return key
        return None

    def get_data(self, var_name, metadata=False):

        legacy_name = self.update_naming(var_name, name_type='old')
        
        # Get the data or metadata for a specific variable
        data = pyspedas.get_data(legacy_name, metadata=metadata, dt=not metadata)

        data_dict = {}

        if metadata:
            data_dict = data
        else:
            for i, field in enumerate(data._fields):
                data_dict[field] = data.__getattribute__(data._fields[i])

        return data_dict

    def convert_min_delta(self, var_name, start_time=None, end_time=None, time_delta=None) -> dict:
        """
        Convert Spectrogram data to a uniform time delta, removing NaN rows and interpolating single NaN values.

        Parameters
        ----------
        data : dict
            Dictionary containing the spectrogram data with keys 'y', 'v', and 'times'.

        start_time : datetime, optional 
            Start time to which the spectrogram data will be cut. If None, the start time of the data will be used.

        end_time : datetime, optional   
            End time to which the spectrogram data will be cut. If None, the end time of the data will be used.
        """

        # Convert the variable name to new name if necessary
        var_name = self.update_naming(var_name, name_type='new')

        # Check what type of data it is
        var_type = self.data[var_name]['metadata']['CDF']['VATT']['PROPERTY']

        print(f"Preprocessing {var_name}")
        # Convert the data to a uniform time delta
        self.data[var_name]['data'] = data_cleaning.convert_min_delta(
            self.data[var_name]['data'],
            start_time=start_time,
            end_time=end_time,
            type=var_type,
            time_delta=time_delta
        )
        
    def convert_min_delta_all(self, start_time=None, end_time=None, lowest_time_delta_spect=True, dt_method='min', custom_time_delta=None) -> None:
        """
        Convert all data to a uniform time delta, removing NaN rows and interpolating single NaN values.

        Parameters
        ----------
        start_time : optional 
            Start time to which the spectrogram data will be cut. If None, the start time of the data will be used.

        end_time : optional   
            End time to which the spectrogram data will be cut. If None, the end time of the data will be used.

        
        """

        # Adjust start_time and end_time attributes of the class
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time

        # Delete variables that only contain NaN values
        self.delete_empty_variables()

        # Get all variable names in the data dictionary
        var_names = list(self.data.keys())

        # If a custom time delta is provided, use that
        if custom_time_delta is not None:
            min_time_delta = custom_time_delta
        elif lowest_time_delta_spect:
            # Find the minimum time delta for all spectrograms
            min_time_delta = self.set_time_delta_spectrograms(start_time, end_time, dt_method)
        else:
            # Set min time delta to None
            min_time_delta = None

        # Loop through each variable name
        for var_name in var_names:
              
            # Convert the variable to same time delta
            self.convert_min_delta(var_name, start_time=start_time, end_time=end_time, time_delta=min_time_delta)

    def delete_empty_variables(self) -> None:

        # Get all variable names in the data dictionary
        var_names = list(self.data.keys())

        # Loop through each variable name
        for var_name in var_names:
            
            # Check whether data only contains NaN values
            if np.all(np.isnan(self.data[var_name]['data']['y'])):
                
                # Remove the variable from the data dictionary
                del self.data[var_name]
                print(f"Variable '{var_name}' contains only NaN values and has been removed.")

    def set_time_delta_spectrograms(self, start_time, end_time, method='min') -> float:
        """
        Find the minimum time delta for all spectrograms in the data.

        Parameters
        ----------
        start : datetime
            Start time to which the spectrogram data will be cut.

        Returns
        -------
        float
            Minimum time delta for all spectrograms.
        """
        # Convert start_time and end_time to numpy.datetime64 if they are strings in 'YYYY-MM-DDTHH:MM:SS' format
        if start_time is not None and isinstance(start_time, str):
            start_time = np.datetime64(start_time)

        if end_time is not None and isinstance(end_time, str):
            end_time = np.datetime64(end_time)

        # Get all variable names in the data dictionary
        var_names = list(self.data.keys())

        min_time_delta = None

        # Loop through each variable name
        for var_name in var_names:
            # Check if the variable is a spectrogram
            if 'spectrogram' == self.data[var_name]['metadata']['CDF']['VATT']['PROPERTY']:
                if method == 'min':
                    # Find the minimum time delta for the current spectrogram
                    time_delta, _, _, _ = data_cleaning.find_min_time_diff(self.data[var_name]['data']['times'], start_time, end_time)
                elif method == 'max':
                    # Find the maximum time delta for the current spectrogram
                    time_delta, _, _, _ = data_cleaning.find_max_time_diff(self.data[var_name]['data']['times'], start_time, end_time)

                # Check if min_time_delta is None or if the current time delta is smaller
                if (min_time_delta is None or time_delta < min_time_delta) and method == 'min':
                    min_time_delta = time_delta
                elif (min_time_delta is None or time_delta > min_time_delta) and method == 'max':
                    min_time_delta = time_delta
                print(f"Variable '{var_name}' has a time delta of {time_delta} seconds.")

        return min_time_delta

    def extract_to_HDF5(self, filename, overwrite=False, session_id=None) -> None:
        """
        Save the loaded data to an HDF5 file.

        Parameters:
            filename (str): Name or path of the HDF5 file to save the data.
            overwrite (bool): Whether to overwrite the file if it already exists.
            session_id (str, optional): Specific session ID to write to. If provided, this session will be overwritten.
        """

        # Ensure the directory exists
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        # Determine file mode based on overwrite parameter
        mode = 'w' if overwrite else 'a'

        with h5py.File(filename, mode) as h5f:
            # Determine session ID
            if session_id is not None:
                # Use provided session ID and overwrite if it exists
                if session_id in h5f:
                    del h5f[session_id]
            else:
                # Determine the next session ID
                existing_sessions = [k for k in h5f.keys() if k.startswith('session_')]
                next_id = len(existing_sessions)
                session_id = f"session_{next_id:04d}"
            
            # Create a group for this time range
            grp = h5f.create_group(session_id)
            
            # Add metadata for the session
            grp.attrs['start_time'] = str(self.start_time)
            grp.attrs['end_time'] = str(self.end_time)
            grp.attrs['creation_date'] = datetime.now().isoformat()
 
            if var_name.startswith('B_'):
                grp.attrs['data_origin'] = 'Themis_B'
            elif var_name.startswith('C_'):
                grp.attrs['data_origin'] = 'Themis_C'
            else:
                grp.attrs['data_origin'] = 'Unknown'
            
            # Store times as a separate dataset (only once)
            times_stored = False
            
            # Store each variable's 'y' data in the session group
            for var_name, var_data in self.data.items():
                if var_name.startswith('B_') or var_name.startswith('C_'):
                    var_storage_name = var_name[2:]
                    
                if 'y' in var_data['data']:
                    # Store the 'y' data array
                    grp.create_dataset(var_storage_name, data=var_data['data']['y'])
                    
                    # Store additional metadata as attributes
                    var_dataset = grp[var_name]
                    
                    # Store times as a separate dataset for the first variable only
                    if not times_stored and 'times' in var_data['data']:
                        times_int64 = var_data['data']['times'].astype('int64')
                        grp.create_dataset('times', data=times_int64)
                        times_stored = True
                    
                    # Add variable type if available in metadata
                    if 'metadata' in var_data and 'CDF' in var_data['metadata']:
                        if 'VATT' in var_data['metadata']['CDF'] and 'PROPERTY' in var_data['metadata']['CDF']['VATT']:
                            var_dataset.attrs['type'] = var_data['metadata']['CDF']['VATT']['PROPERTY']
            
            print(f"Data saved to {filename} in group {session_id}")

    ### Plotting Methods ###

    # Does not update with preprocessing!!! ###
    def plot_pyspedas(self, variable_name):

        legacy_name = self.legacy_naming(variable_name)
        if legacy_name is None:
            raise ValueError(f"Variable name '{variable_name}' not found in the mapping.")

        pyspedas.tplot(legacy_name)

    # Updates with preprocessing, so preferably use this method
    def plot_electron_spectrogram(self, times=None, energy=None, figsize=(10, 4), vmin=None, vmax=None, norm='Log', title=None, variable_name='B_ion_eflux'):
        """
        Plot electron spectrogram using the loaded data.

        Parameters:
            times (np.ndarray): Time stamps for the spectrogram.
            energy (np.ndarray): Energy bins for the spectrogram.
            figsize (tuple): Size of the figure.
            vmin (float): Minimum value for color normalization.
            vmax (float): Maximum value for color normalization.
            norm (str): Normalization type ('Log' or 'Linear').
            title (str): Title of the plot.
        """
        electron_data = self.data[variable_name]['data']['y']
        plot_electron_spectrogram(electron_data, times=times, energy=energy, figsize=figsize, vmin=vmin, vmax=vmax, norm=norm, title=title)

    def plot_variables(self, figsize=(10,4), variable_names=None):

        fig, axs = plt.subplots(len(variable_names), 1, figsize=figsize)
        
        # Ensure axs is always iterable
        if len(variable_names) == 1:
            axs = [axs]
        
        for i in range(len(variable_names)):
            variable_name = variable_names[i]

            # Extract data
            variable_data = self.data[variable_name]['data']['y']
            variable_times = self.data[variable_name]['data']['times']

            # Convert to displayable format
            time_ns = variable_times.astype('int64')
            time_s = time_ns / 1e9
            time = mdates.date2num([datetime.fromtimestamp(ts, timezone.utc) for ts in time_s])

            # Check if the variable data is a vector
            if variable_data.ndim > 1:
                # If it's a vector, plot each component separately
                for j in range(variable_data.shape[1]):
                    axs[i].plot(time, variable_data[:, j], label=f'Component {j+1}')
                axs[i].legend()
            else:
                # If it's a scalar, plot it directly
                axs[i].plot(time, variable_data, label=variable_name)

            # Set time formatter
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))
            axs[i].xaxis_date()
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(variable_name)

            axs[i].plot(time, variable_data)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.grid()
        plt.show()   


