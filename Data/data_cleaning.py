import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from datetime import datetime, timezone
from types import SimpleNamespace
import scipy as sp

def convert_min_delta(data, start_time=None, end_time=None, type='scalar', time_delta=None) -> dict:
    """
    Convert data to a uniform time delta, removing NaN rows and interpolating single NaN values.

    Parameters
    ----------
    data : dict
        Dictionary containing the spectrogram data with keys 'y', 'v', and 'times'.

    start_time : datetime, optional 
        Start time to which the spectrogram data will be cut. If None, the start time of the data will be used.

    end_time : datetime, optional   
        End time to which the spectrogram data will be cut. If None, the end time of the data will be used.

    type : str
        Type of data to convert. Can be 'scalar', 'vector', or 'spectrogram'. Default is 'scalar'.
    """
    # Convert start_time and end_time to numpy.datetime64 if they are strings in 'YYYY-MM-DDTHH:MM:SS' format
    if start_time is not None and isinstance(start_time, str):
        start_time = np.datetime64(start_time)

    if end_time is not None and isinstance(end_time, str):
        end_time = np.datetime64(end_time)

    # Check the type of data and call the appropriate conversion function
    if type == 'scalar':
        data = convert_min_delta_scalar(data, start_time, end_time, time_delta)
    elif type == 'vector':
        data = convert_min_delta_vector(data, start_time, end_time, time_delta)
    elif type == 'spectrogram':
        data = convert_min_delta_spectrogram(data, start_time, end_time, time_delta)
    else:
        raise ValueError("Invalid type. Must be 'scalar', 'vector', or 'spectrogram'.")

    return data

def convert_min_delta_spectrogram(data, start_time=None, end_time=None, time_delta=None) -> dict:
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

    # Remove all NaN rows from the spectrogram
    data = remove_nan_columns(data)

    # Convert numpy.datetime64 objects to timestamps (seconds since epoch)
    times_dt = data['times']

    # Find the minimum time difference in the data
    min_time_diff, time_i, time_f, times_float = find_min_time_diff(times_dt, start_time, end_time)
    
    # Check whether the time difference is less than 1 second
    min_time_diff = max(min_time_diff, 1.0)      

    if time_delta is None:
        # Create a new time array with the minimum time difference
        new_time_float = np.arange(start_time.astype(float), end_time.astype(float), min_time_diff)

    else:
        # Create a new time array with the specified time delta
        new_time_float = np.arange(start_time.astype(float), end_time.astype(float), time_delta)

    # Interpolate the spectrogram data to match the new time array
    new_spectrogram = np.empty((len(new_time_float), data['y'].shape[1]))
    new_energy = np.empty((len(new_time_float), data['y'].shape[1]))
    for i in range(data['y'].shape[1]):

        # Extract the y and v columns for the current index
        y_col = data['y'][:, i]
        v_col = data['v'][:, i]

        # Make all zeroes NaN
        y_col[y_col == 0] = np.nan

        # Create mask for valid points, to avoid interpolating over NaNs
        valid_mask = ~np.isnan(y_col)

        # Interpolate only if there are valid points
        if np.sum(valid_mask) > 1:

            # Interpolates the spectrogram data (if there are NaNs in the first column, take the first valid point)
            # If the first value is NaN, fill it with the first valid value before interpolation
            if np.isnan(y_col[0]):
                first_valid_idx = np.where(valid_mask)[0][0]
                y_col[:first_valid_idx] = y_col[first_valid_idx]
                v_col[:first_valid_idx] = v_col[first_valid_idx]
            

            new_spectrogram[:, i] = np.interp(new_time_float, times_float[valid_mask], y_col[valid_mask])

            # Interpolates the energy bins, but they really are constant, just in case it needs to be done for some reason
            new_energy[:, i] = np.interp(new_time_float, times_float[valid_mask], v_col[valid_mask]) 
        else:
            raise ValueError(f"Not enough valid points to interpolate for index {i}")

    # Update the data dictionary with the new time and spectrogram
    # Convert new_time_float back to numpy.datetime64 objects
    new_time_dt = np.array(new_time_float * 1e9, dtype='datetime64[ns]')
    data['times'] = new_time_dt
    data['y'] = new_spectrogram
    data['v'] = new_energy

    return data

def convert_min_delta_vector(data, start_time=None, end_time=None, time_delta=None) -> dict:
    """
    Convert Vector data to a uniform time delta, removing NaN rows and interpolating single NaN values.

    Parameters
    ----------
    data : dict
        Dictionary containing the spectrogram data with keys 'y', 'v', and 'times'.

    start_time : datetime, optional 
        Start time to which the spectrogram data will be cut. If None, the start time of the data will be used.

    end_time : datetime, optional   
        End time to which the spectrogram data will be cut. If None, the end time of the data will be used.
    """

    # Convert numpy.datetime64 objects to timestamps (seconds since epoch)
    times_dt = data['times']
    
    # Find the minimum time difference in the data
    min_time_diff, time_i, time_f, times_float = find_min_time_diff(times_dt, start_time, end_time)
    
    # Check whether the time difference is less than 1 second
    min_time_diff  = max(min_time_diff, 1.0)
        
    if time_delta is None:
        # Create a new time array with the minimum time difference
        time_delta = min_time_diff

    # Create a new time array with the specified time delta
    new_time_float = np.arange(start_time.astype(float), end_time.astype(float), time_delta)

    # Create empty array for the new vector data
    new_vector = np.empty((len(new_time_float), data['y'].shape[1]))

    # Interpolate the vector data to match the new time array
    for i in range(data['y'].shape[1]):

        # Extract the y column for the current index
        y_col = data['y'][:, i]

        # Make all zeroes NaN
        y_col[y_col == 0] = np.nan

        # Create mask for valid points, to avoid interpolating over NaNs
        valid_mask = ~np.isnan(y_col)

        # Interpolate only if there are valid points
        if np.sum(valid_mask) > 1:

            # Interpolates the vector data (if there are NaNs in the first column, take the first valid point)
            # If the first value is NaN, fill it with the first valid value before interpolation
            if np.isnan(y_col[0]):
                first_valid_idx = np.where(valid_mask)[0][0]
                y_col[:first_valid_idx] = y_col[first_valid_idx]

            use_mean = True
            # Check how many data points are contained within one time delta. If more than two for every timestamp, use means
            for j in range(len(new_time_float)):
                # Find the indices of the valid points that are within the current time delta
                valid_indices = np.where((times_float[valid_mask] >= new_time_float[j] - time_delta / 2) & 
                                         (times_float[valid_mask] < new_time_float[j] + time_delta / 2))[0]
                
                if len(valid_indices) < 2:
                    use_mean = False
                
                else:
                    # If there are valid points, use the mean of the valid points
                    new_vector[j, i] = np.mean(y_col[valid_mask][valid_indices])


            if not use_mean:
                # If there are not enough valid points, use linear interpolation
                # Interpolates the vector data

                new_vector[:, i] = np.interp(new_time_float, times_float[valid_mask], y_col[valid_mask])

            
        else:
            raise ValueError(f"Not enough valid points to interpolate for index {i}")

    # Update the data dictionary with the new time and vector data
    # Convert new_time_float back to numpy.datetime64 objects
    new_time_dt = np.array(new_time_float * 1e9, dtype='datetime64[ns]')
    data['times'] = new_time_dt
    data['y'] = new_vector

    return data    

def convert_min_delta_scalar(data, start_time=None, end_time=None, time_delta=None) -> dict:
    """
    Convert Scalar data to a uniform time delta, removing NaN rows and interpolating single NaN values.

    Parameters
    ----------
    data : dict
        Dictionary containing the spectrogram data with keys 'y', 'v', and 'times'.

    start_time : datetime, optional !!!NOT IMPLEMENTED!!!
        Start time to which the spectrogram data will be cut. If None, the start time of the data will be used.

    end_time : datetime, optional   !!!NOT IMPLEMENTED!!!
        End time to which the spectrogram data will be cut. If None, the end time of the data will be used.
    """

    # Convert numpy.datetime64 objects to timestamps (seconds since epoch)
    times_dt = data['times']

    # Find the minimum time difference in the data
    min_time_diff, time_i, time_f, times_float = find_min_time_diff(times_dt, start_time, end_time)
    
    # Check whether the time difference is less than 1 second
    min_time_diff  = max(min_time_diff, 1.0)

    if time_delta is None:
        time_delta = min_time_diff
        
    # Create a new time array with the specified time delta
    new_time_float = np.arange(start_time.astype(float), end_time.astype(float), time_delta)

    # Create empty array for the new vector data
    new_scalar = np.empty(len(new_time_float))

    # Make all zeroes NaN
    data['y'][data['y'] == 0] = np.nan

    # Create mask for valid points, to avoid interpolating over NaNs
    valid_mask = ~np.isnan(data['y'])

    # Interpolate only if there are valid points
    if np.sum(valid_mask) > 1:

        # Initialize new_scalar with NaNs
        new_scalar = np.empty(len(new_time_float))
        new_scalar[:] = np.nan

        # Check how many data points are contained within one time delta. If more than two for every timestamp, use means
        for i in range(len(new_time_float)):

            # Find the indices of the valid points that are within the current time delta
            valid_indices = np.where((times_float[valid_mask] >= new_time_float[i] - time_delta / 2) & 
                                     (times_float[valid_mask] < new_time_float[i] + time_delta / 2))[0]
            
            # Check if there are at least two valid points to take the mean
            if len(valid_indices) >= 2:

                # If there are valid points, use the mean of the valid points
                new_scalar[i] = np.mean(data['y'][valid_mask][valid_indices])

        # Get all indices where new_scalar is still empty (i.e. there were not enough valid points to take the mean)
        empty_indices = np.where(np.isnan(new_scalar))[0]

        # Interpolate these indices using linear interpolation
        new_scalar[empty_indices] = np.interp(new_time_float[empty_indices], times_float[valid_mask], data['y'][valid_mask]) 
    else:
        raise ValueError(f"Not enough valid points to interpolate")

    # Update the data dictionary with the new time and vector data
    # Convert new_time_float back to numpy.datetime64 objects
    new_time_dt = np.array(new_time_float * 1e9, dtype='datetime64[ns]')
    data['times'] = new_time_dt
    data['y'] = new_scalar

    return data    

def find_min_time_diff(times_dt, start_time=None, end_time=None):
    """
    Find the minimum time difference in the data.

    Parameters
    ----------
    data : dict
        Dictionary containing the spectrogram data with keys 'y', 'v', and 'times'.

    Returns
    -------
    min_time_diff : float
        Minimum time difference in seconds.
    """

    if isinstance(times_dt[0], np.datetime64):
        # Convert to seconds since epoch
        times_float = (times_dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        times_float = np.array(times_float, dtype=float)
    else:
        times_float = np.array(times_dt)

    # Calculate the time differences in seconds
    time_diffs = np.diff(times_float)

    # From the start time to the end time, create a new time array with the minimum time difference
    if start_time is not None:
        start_time_float = (start_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        time_i = max(times_float[0], start_time_float)
    else:
        time_i = times_float[0]

    if end_time is not None:
        end_time_float = (end_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        time_f = min(times_float[-1], end_time_float)
    else:
        time_f = times_float[-1]

    # Find the index of the first timestamp that is greater than or equal to time_i
    start_idx = np.searchsorted(times_float, time_i)

    # Find the index of the last timestamp that is less than or equal to time_f
    end_idx = np.searchsorted(times_float, time_f)

    # Get the minimum time difference, but not less than 1 second, and within the start and end time
    min_time_diff = time_diffs[start_idx:end_idx].min()
    
    return min_time_diff, time_i, time_f, times_float

def find_max_time_diff(times_dt, start_time=None, end_time=None):
    """
    Find the maximum time difference in the data.

    Parameters
    ----------
    times_dt : array-like
        Array of datetime64 objects or timestamps.
    start_time : datetime or numpy.datetime64, optional
        Start time boundary for calculation. If None, the first timestamp is used.
    end_time : datetime or numpy.datetime64, optional
        End time boundary for calculation. If None, the last timestamp is used.

    Returns
    -------
    max_time_diff : float
        Maximum time difference in seconds.
    time_i : float
        Start time in seconds since epoch.
    time_f : float
        End time in seconds since epoch.
    times_float : ndarray
        Array of timestamps converted to seconds since epoch.
    """
    if isinstance(times_dt[0], np.datetime64):
        # Convert to seconds since epoch
        times_float = (times_dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        times_float = np.array(times_float, dtype=float)
    else:
        times_float = np.array(times_dt)

    # Calculate the time differences in seconds
    time_diffs = np.diff(times_float)

    # From the start time to the end time, create a new time array with the minimum time difference
    if start_time is not None:
        start_time_float = (start_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        time_i = max(times_float[0], start_time_float)
    else:
        time_i = times_float[0]

    if end_time is not None:
        end_time_float = (end_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        time_f = min(times_float[-1], end_time_float)
    else:
        time_f = times_float[-1]

    # Find the index of the first timestamp that is greater than or equal to time_i
    start_idx = np.searchsorted(times_float, time_i)

    # Find the index of the last timestamp that is less than or equal to time_f
    end_idx = np.searchsorted(times_float, time_f)

    # Get the maximum time difference within the start and end time
    max_time_diff = time_diffs[start_idx:end_idx].max()
    
    return max_time_diff, time_i, time_f, times_float

def remove_nan_rows(data):
    """
    Remove all columns (energy bins) from the spectrogram where:
    - All values are NaN, or
    - The column has more than 12 NaN values.
    """
    y = data['y']

    # Condition 1: Columns that are all NaN
    mask_not_all_nan = ~np.all(np.isnan(y), axis=1)
    # Condition 2: Columns with more than 12 NaNs
    mask_nan_count = np.sum(np.isnan(y), axis=1) <= 12

    # Combine both masks (keep columns that satisfy both conditions)
    mask = mask_not_all_nan & mask_nan_count

    # Apply the mask to the spectrogram data
    data['y'] = data['y'][mask, :]
    # Apply the mask to the energy bins
    data['v'] = data['v'][mask, :]
    data['times'] = data['times'][mask]

    return data

def interpolate_nan_columns(data):
    # Interpolate NaN values in the spectrogram
    for i in range(data['y'].shape[1]):
        data['y'][:, i] = np.interp(np.arange(data['y'].shape[0]), 
                                    np.arange(data['y'].shape[0])[~np.isnan(data['y'][:, i])], 
                                    data['y'][~np.isnan(data['y'][:, i]), i])
    return data

def remove_nan_columns(data):
    """
    Remove all columns (that contain only 0's, Inf's, or NaN's) from the spectrogram data.
    Also remove every column that has more than 10 NaNs (independent condition).
    """

    y = data['y']

    # Condition 1: Columns that are all NaN, zero, or Inf
    mask_valid = ~np.all(np.isnan(y) | (y == 0) | np.isinf(y), axis=0)

    # # Condition 2: Columns with more than 10 NaNs
    # mask_nan_count = np.sum(np.isnan(y), axis=0) <= 10

    # Combine both masks (keep columns that satisfy both conditions)
    mask = mask_valid 

    # Apply the mask to the spectrogram data
    data['y'] = data['y'][:, mask]

    # Apply the mask to the energy bins
    data['v'] = data['v'][:, mask]

    return data
