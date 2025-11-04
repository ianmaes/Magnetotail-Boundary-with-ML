import numpy as np
import pyspedas
from pyspedas.cotrans_tools.cotrans_lib import cdipdir, cdipdir_vect
from pyspedas.cotrans_tools.cotrans_lib import subcotrans

import warnings

def GSE_to_aSWGSM(position, dipole_tilt, aberration_y=0.0, aberration_z=0.0):
    """
    Convert positions from GSE to aberrated SWGSM coordinates.
    
    Parameters:
    - positions: 1x3 array of position in GSE coordinates (X, Y, Z).
    - aberration_z: Aberration angle in degrees around z axis.
    - aberration_y: Aberration angle in degrees around y axis.
    - dipole_tilt: 1x3 array of dipole direction vector (in GSE coordinates)
    
    Returns:
    - transformed_positions: 1x3 array of positions in aberrated SWGSM coordinates.
    """
    # Convert aberration angle from degrees to radians
    aberration_y_rad = np.radians(aberration_y)
    aberration_z_rad = np.radians(aberration_z)
    
    # Rotation matrix for aberration around Z-axis GSE to aSWGSE
    R_aberration = np.array([
        [np.cos(aberration_y_rad) * np.cos(aberration_z_rad), -np.sin(aberration_y_rad)* np.cos(aberration_z_rad), -np.sin(aberration_z_rad)],
        [np.sin(aberration_y_rad),  np.cos(aberration_y_rad), 0],
        [np.cos(aberration_y_rad) * np.sin(aberration_z_rad), -np.sin(aberration_y_rad)* np.sin(aberration_z_rad),  np.cos(aberration_z_rad)]
    ])
    
        # Degrees → radians
    az = np.radians(aberration_z)
    ay = np.radians(aberration_y)

    # Active rotations (right-handed): first about Z, then about Y.
    Rz = np.array([[ np.cos(ay), -np.sin(ay), 0.0],
                   [ np.sin(ay),  np.cos(ay), 0.0],
                   [ 0.0,         0.0,        1.0]])
    Ry = np.array([[ np.cos(az), 0.0,  -np.sin(az)],
                   [ 0.0,        1.0,  0.0],
                   [np.sin(az), 0.0,  np.cos(az)]])

    # Composite aberration: R_aberration = Ry · Rz (first Rz, then Ry)
    R_aberration = Ry @ Rz

    # Rotate dipole direction from GSE to aSWGSE
    dipole_tilt_aswgse = dipole_tilt @ R_aberration.T

    # Project dipole direction onto aSWGSE YZ-plane
    dipole_proj = dipole_tilt_aswgse.copy()
    dipole_proj[0] = 0  # Set X-component to zero
    
    # Normalize the projected dipole
    dipole_proj_norm = np.linalg.norm(dipole_proj)
    dipole_proj_unit = dipole_proj / dipole_proj_norm
    
    # Get values for rotation around x axis
    cos_theta = dipole_proj_unit[2]  # Z component
    sin_theta = dipole_proj_unit[1]  # Y component (for rotation about X-axis)
    
    # Create rotation matrices for each point (rotation about X-axis)
    R_dipole = np.array([
        [1,             0,              0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta,  cos_theta]
    ])

    # Apply aberration rotation first
    positions_aswgse = position @ R_aberration.T
    
    # Then apply dipole rotation
    transformed_position = positions_aswgse @ R_dipole.T

    return transformed_position

def GSE_to_aSWGSM_array(position_array, times_array):
    """
    Convert an array of positions from GSE to aberrated SWGSM coordinates.
    
    Parameters
    ----------
    - position_array: Nx3 array of positions in GSE coordinates (X, Y, Z).
    - dipole_tilt_array: Nx3 array of dipole direction vectors (in GSE coordinates).

    
    Returns
    ----------
    - transformed_positions: Nx3 array of positions in aberrated SWGSM coordinates.
    """

    # Convert times to list of float seconds since epoch
    times_array_list = times_array.astype('datetime64[s]').astype(float).tolist()

    # Get dipole tilt for each time
    dipole_pos_all_geo = np.vstack(cdipdir_vect(times_array_list)).T
    dipole_pos_all_gse = subcotrans(time_in=times_array_list, data_in=dipole_pos_all_geo, coord_in='geo', coord_out='gse')

    # Group times and positions by month to minimize OMNI calls
    times_grouped, positions_grouped = group_crossings_by_month(times_array, position_array)

    # For each group, get solar wind velocity and calculate aberration angles
    velocities = np.zeros((position_array.shape[0], 3))
    index = 0
    for times, positions in zip(times_grouped, positions_grouped):
        vel = get_solar_wind_velocity(times, positions)
        velocities[index:index+len(times)] = vel
        index += len(times)

    aberration_y_array, aberration_z_array = calculate_aberration_angles_array(velocities)

    transformed_positions = np.array([
        GSE_to_aSWGSM(position_array[i], dipole_pos_all_gse[i], aberration_y_array[i], aberration_z_array[i])
        for i in range(position_array.shape[0])
    ])
    
    return transformed_positions, aberration_y_array, aberration_z_array

def calculate_aberration_angles(velocity):
    """
    Calculate aberration angles based on solar wind velocity.
    
    Parameters
    ----------
    - velocity: Nx3 array of solar wind velocity components (Vx, Vy, Vz) in km/s.
    
    Returns
    ----------
    - aberration_y: N array of aberration angles in degrees around y axis.
    - aberration_z: N array of aberration angles in degrees around z axis.
    """
    Vx = velocity[:, 0]
    Vy = velocity[:, 1]
    Vz = velocity[:, 2]
    
    # Calculate aberration angles in radians
    aberration_y_rad = np.arctan2(Vy + 30, np.abs(Vx))
    aberration_z_rad = np.arctan2(Vz, np.sqrt(Vx**2 + (Vy + 30)**2))
    
    # Convert to degrees
    aberration_y = np.degrees(aberration_y_rad)
    aberration_z = np.degrees(aberration_z_rad)
    
    return aberration_y, aberration_z

def calculate_aberration_angles_array(velocity_array):
    """
    Calculate aberration angles for an array of solar wind velocities.
    
    Parameters
    ----------
    - velocity_array: Nx3 array of solar wind velocity components (Vx, Vy, Vz) in km/s.
    
    Returns
    ----------
    - aberration_y_array: N array of aberration angles in degrees around y axis.
    - aberration_z_array: N array of aberration angles in degrees around z axis.
    """
    aberration_y_array, aberration_z_array = calculate_aberration_angles(velocity_array)
    
    return aberration_y_array, aberration_z_array

def get_solar_wind_pressure_array(timestamps):
    """
    Retrieve solar wind dynamic pressure data from OMNI for an array of timestamps.
    
    Parameters
    ----------
    - timestamps: Nx1 array of np.datetime64 or similar, the times for which to retrieve the data.
    
    Returns
    ----------
    - pressures: N array of solar wind dynamic pressure in nPa.
    """
    pressures = np.zeros(len(timestamps))
    month_groups, position_groups = group_crossings_by_month(timestamps, np.zeros((len(timestamps), 3)))
    
    index = 0
    for times in month_groups:
        month_pressures = get_solar_wind_pressure(times)
        pressures[index:index+len(times)] = month_pressures
        index += len(times)
    
    return pressures

def get_solar_wind_pressure(timestamps):
    """
    Retrieve solar wind dynamic pressure data from OMNI for a given timestamp.
    
    Parameters
    ----------
    - timestamp: np.datetime64 or similar, the time for which to retrieve the data.
    - position: 1x3 array of position in GSE coordinates (X, Y, Z).
    
    Returns
    ----------
    - pressure: array of solar wind dynamic pressure in nPa.
    """

    # Get month from first timestamp
    month = np.datetime64(timestamps[0], 'M')
    month_str = str(month.astype('datetime64[s]'))
    month_end_str = str((month + np.timedelta64(1, 'M')).astype('datetime64[s]'))

    # Load OMNI data using pyspedas
    omni_vars = ['Pressure']
    omni_data = pyspedas.projects.omni.data(trange=[month_str, month_end_str], datatype='5min', varnames=omni_vars, time_clip=True, notplot=True)


    # For each timestamp and position, get the solar wind dynamic pressure
    pressures = np.zeros(len(timestamps))
    i = 0
    for time in timestamps:
        if time > np.datetime64('2017-09-09T00:00:00') and time < np.datetime64('2017-09-10T00:00:00'):
            print("Debug breakpoint")

        use_daily_avg = False

        # Start 1 hour before to ensure coverage
        time_init = time - np.timedelta64(60, 'm')  

        mask = (omni_data['Pressure']['x'] >= time_init) & (omni_data['Pressure']['x'] <= time)

        # Get pressure of last hour and average
        if 'Pressure' in omni_data and len(omni_data['Pressure']['y'][mask]) > 0:
            Pdyn = np.nanmean(omni_data['Pressure']['y'][mask])
            if np.isnan(Pdyn):
                use_daily_avg = True
        else:
            use_daily_avg = True

        if use_daily_avg:
            day_mask = (omni_data['Pressure']['x'] >= time - np.timedelta64(24, 'h')) & (omni_data['Pressure']['x'] <= time)
            Pdyn = np.nanmean(omni_data['Pressure']['y'][day_mask])
            warnings.warn(f"No OMNI Pdyn data available in the last hour for timestamp {time}, using daily average.")

        pressures[i] = Pdyn

        # Index time
        i += 1
    return pressures

def get_solar_wind_velocity(timestamps, positions):
    """
    Retrieve solar wind velocity data from OMNI for a given timestamp.
    
    Parameters
    ----------
    - timestamp: np.datetime64 or similar, the time for which to retrieve the data.
    - position: 1x3 array of position in GSE coordinates (X, Y, Z).
    
    Returns
    ----------
    - velocity: 1x3 array of solar wind velocity components (Vx, Vy, Vz) in km/s.
    """

    # Get month from first timestamp
    month = np.datetime64(timestamps[0], 'M')
    month_str = str(month.astype('datetime64[s]'))
    month_end_str = str((month + np.timedelta64(1, 'M')).astype('datetime64[s]'))

    # Load OMNI data using pyspedas
    omni_vars = ['Vx', 'Vy', 'Vz']
    omni_data = pyspedas.projects.omni.data(trange=[month_str, month_end_str], datatype='5min', varnames=omni_vars, time_clip=True, notplot=True)


    # For each timestamp and position, get the solar wind velocity
    velocities = np.zeros((len(timestamps), 3))
    i = 0
    for time, pos in zip(timestamps, positions):

        use_daily_avg = False

        # Start 1 hour before to ensure coverage
        time_init = time - np.timedelta64(60, 'm')  

        mask = (omni_data['Vx']['x'] >= time_init) & (omni_data['Vx']['x'] <= time)

        # Get velocity_x of last hour and average
        if 'Vx' in omni_data and len(omni_data['Vx']['y'][mask]) > 0:
            Vx = np.nanmean(omni_data['Vx']['y'][mask])

        # Calculate time delay based on position (in Earth radii) and average solar wind speed from omni data
        x_gse = pos[0]

        if 'Vx' in omni_data and len(omni_data['Vx']['y'][mask]) > 0:

            avg_vx = np.nanmean(omni_data['Vx']['y'][mask])
            day_mask = (omni_data['Vx']['x'] >= time - np.timedelta64(24, 'h')) & (omni_data['Vx']['x'] <= time)
            if len(omni_data['Vx']['y'][mask]) == 0 or np.isnan(avg_vx) or avg_vx == 0:
                
                avg_vx = np.nanmean(omni_data['Vx']['y'][day_mask])
                use_daily_avg = True
                warnings.warn(f"No OMNI Vx data available in the last hour for timestamp {time}, using daily average.")
            
            try:
                time_delay = (np.abs(x_gse) * 6371) / np.abs(avg_vx)  # in seconds

            except:
                print(f"Error calculating time delay for timestamp {time} with position {pos} and avg_vx {avg_vx}. Using default time delay of 3600s.")


        else:
            raise ValueError("OMNI data does not contain Vx information.")


        if use_daily_avg or len(omni_data['Vx']['y'][mask]) == 0:
            
            Vx = np.nanmean(omni_data['Vx']['y'][day_mask])
            Vy = np.nanmean(omni_data['Vy']['y'][day_mask])
            Vz = np.nanmean(omni_data['Vz']['y'][day_mask])

            warnings.warn("No OMNI data available in the last hour after time delay, using daily average.")
        else:


            # Adjust timestamp for time delay
            adjusted_time = time - np.timedelta64(int(time_delay), 's')
            adjusted_time_init = adjusted_time - np.timedelta64(60, 'm')


            mask = (omni_data['Vx']['x'] >= adjusted_time_init) & (omni_data['Vx']['x'] <= adjusted_time)
            Vx = np.nanmean(omni_data['Vx']['y'][mask])
            Vy = np.nanmean(omni_data['Vy']['y'][mask])
            Vz = np.nanmean(omni_data['Vz']['y'][mask])

            if len(omni_data['Vx']['y'][mask]) == 0 or np.isnan(Vx):
                Vx = np.nanmean(omni_data['Vx']['y'][day_mask])
                Vy = np.nanmean(omni_data['Vy']['y'][day_mask])
                Vz = np.nanmean(omni_data['Vz']['y'][day_mask])
                warnings.warn(f"No OMNI Vx data available after time delay for timestamp {time}, using daily average.")

        velocities[i] = np.array([Vx, Vy, Vz])

        # Index time
        i += 1
    return velocities

def group_crossings_by_month(crossing_times, crossing_positions):
    """
    Group crossing times and positions by month.
    
    Parameters
    ----------
    - crossing_times: array-like of np.datetime64, crossing timestamps.
    - crossing_positions: Nx3 array of crossing positions in GSE coordinates.
    
    Returns
    ----------
    - grouped_times: list of arrays, each containing crossing times for a specific month.
    - grouped_positions: list of arrays, each containing crossing positions for a specific month.
    """
    from collections import defaultdict

    grouped_times_dict = defaultdict(list)
    grouped_positions_dict = defaultdict(list)

    for time, position in zip(crossing_times, crossing_positions):
        month = np.datetime64(time, 'M')
        grouped_times_dict[month].append(time)
        grouped_positions_dict[month].append(position)

    grouped_times = [np.array(times) for times in grouped_times_dict.values()]
    grouped_positions = [np.array(positions) for positions in grouped_positions_dict.values()]

    return grouped_times, grouped_positions