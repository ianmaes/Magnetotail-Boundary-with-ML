import numpy as np
import torch
import datetime
import pyspedas 
import pyspedas.projects.omni as omni

from Helper.silence import silence_all
from Helper.shue_model import shue_radius_at_x
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter, HourLocator, date2num, MonthLocator, DayLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from Helper.shue_model import shue_magnetopause_surface

def plot_3d_magnetopause_crossings(predicted_positions, true_positions, true_data_pos=None, 
                                 Bz_nT=0.0, Pdyn_nPa=2.0, earth_radius=3.0, 
                                 xlim=[-60, 20], ylim=[-40, 40], zlim=[-40, 40],
                                 figsize=(13, 10), **kwargs):
    """
    Create a 3D plot showing predicted vs true magnetopause crossing positions.
    
    Parameters:
    - predicted_positions: Array of predicted crossing positions (N x 3).
    - true_positions: Array of true crossing positions (N x 3).
    - true_data_pos: Optional spacecraft trajectory positions.
    - Bz_nT: IMF Bz component in nT for Shue model (default 0.0).
    - Pdyn_nPa: Solar wind dynamic pressure in nPa for Shue model (default 2.0).
    - earth_radius: Earth radius in Re (default 3.0).
    - xlim, ylim, zlim: Axis limits as [min, max] lists.
    - figsize: Figure size tuple.
    - kwargs: Additional keyword arguments for figure creation.
    """
    import matplotlib.lines as mlines
    
    fig = plt.figure(figsize=figsize, **kwargs)
    ax = fig.add_subplot(111, projection='3d')

    # Create Earth sphere with radius of 1 Re
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    # Plot predicted positions
    ax.scatter(predicted_positions[:, 0], predicted_positions[:, 1], predicted_positions[:, 2], 
               c='red', s=50, alpha=0.8, label='Predicted Crossings')

    # Plot true positions
    ax.scatter(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], 
               c='blue', s=50, alpha=0.3, label='True Crossings')

    # Plot spacecraft trajectory if provided
    if true_data_pos is not None:
        ax.plot(true_data_pos[:, 0], true_data_pos[:, 1], true_data_pos[:, 2], 
                c='black', alpha=0.1, label='Spacecraft Trajectory')

    # Set labels
    ax.set_xlabel('X (Re)')
    ax.set_ylabel('Y (Re)')
    ax.set_zlabel('Z (Re)')

    # Set equal aspect ratio for all axes
    box_aspect = [3/2, 1, 1]  # Different aspect ratio
    ax.set_box_aspect(box_aspect)

    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Build the magnetopause surface
    mp_x, mp_y, mp_z = shue_magnetopause_surface(Bz_nT=Bz_nT, Pdyn_nPa=Pdyn_nPa,
                                                 Xmin=-80, Xmax=xlim[1])

    # Apply the same visual x-stretch
    x_stretch = (box_aspect[1] / box_aspect[0]) * ((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))
    mp_x_plot = mp_x * x_stretch

    # Draw the surface
    ax.plot_surface(mp_x_plot, mp_y, mp_z, alpha=0.15, linewidth=0, color='gray')

    # Add a legend entry for the surface
    mp_proxy = mlines.Line2D([], [], color='gray', lw=6, alpha=0.6,
                             label=f'Magnetopause (Shue model: Bz={Bz_nT} nT, Pdyn={Pdyn_nPa} nPa)')

    stretch = box_aspect[1] / box_aspect[0]  # Stretch factor for x-axis

    # Earth sphere
    earth_x = earth_radius * np.outer(np.cos(u), np.sin(v)) * stretch * (np.diff(xlim) / np.diff(ylim))
    earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
    earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot Earth as a green sphere
    ax.plot_surface(earth_x, earth_y, earth_z, color='green', alpha=0.6)
    
    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mp_proxy)
    labels.append(mp_proxy.get_label())
    ax.legend(handles, labels)
    
    ax.set_title('Predicted vs True Magnetopause Crossing Positions')
    plt.tight_layout()
    plt.show()

def plot_variables_from_sections(section, variables, start_time=None, end_time=None, crossing_times=None, crossing_times_true=None, xlabel=None, ylabel=None, title=None, logscale=True, show_time_tick_labels=False, hour_interval=12, **kwargs):
    """
    Plots a specified variable from a list of sections over time.

    Parameters:
    - sections: List of dictionaries, each containing 'time' and the specified variable.
    - variable: The key in the section dictionaries to plot.
    - title: Optional title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis. If None, uses the variable name.
    
    Returns:
    - dict: Dictionary containing the plotted data with original data type names
    """

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Number of variables to plot
    n_variables = len(variables)

    # Create subplots with minimal spacing
    fig, axs = plt.subplots(n_variables, 1, **kwargs)

    # Handle case where there's only one subplot (axs is not a list)
    if n_variables == 1:
        axs = [axs]

    # Gather times
    times = section['times']

    # Convert to matplotlib times
    if isinstance(times, torch.Tensor):
        times = times.cpu().numpy()
        times_ns = times.astype('datetime64[ns]')
    else:
        times = np.array(times)
        times_ns = times.astype('datetime64[ns]')
    
    # Filter data based on start_time and end_time if provided
    time_mask = np.ones(len(times_ns), dtype=bool)
    if start_time is not None:
        start_time_ns = np.datetime64(start_time)
        time_mask &= (times_ns >= start_time_ns)
    if end_time is not None:
        end_time_ns = np.datetime64(end_time)
        time_mask &= (times_ns <= end_time_ns)
    
    # Apply time mask to filter data
    times_ns = times_ns[time_mask]
    
    # Initialize output dictionary
    output_data = {
        'times': times_ns,
        'times_original_type': type(section['times']).__name__
    }
    
    if crossing_times is not None:
        crossing_times = np.array(crossing_times)

        # Check what crossing times lie within the time range
        crossing_times = crossing_times[(crossing_times >= times_ns[0]) & (crossing_times <= times_ns[-1])]

        # Plot vertical lines at crossing times
        for i, ct in enumerate(crossing_times):
            ct_dt = datetime.datetime.fromtimestamp(ct.astype('O') / 1e9, datetime.timezone.utc)
            ct_num = date2num(ct_dt)
            for j, ax in enumerate(axs):
                if i == 0 and j == 0:
                    ax.axvline(ct_num, color='k', linestyle='--', alpha=1, lw=0.7, label='Detected Crossings')
                else:
                    ax.axvline(ct_num, color='k', linestyle='--', alpha=1, lw=0.7)

    if crossing_times_true is not None:
        crossing_times_true = np.array(crossing_times_true)

        # Check what crossing times lie within the time range
        crossing_times_true = crossing_times_true[(crossing_times_true >= times_ns[0]) & (crossing_times_true <= times_ns[-1])]

        # Plot vertical lines at true crossing times
        for i, ct in enumerate(crossing_times_true):
            ct_dt = datetime.datetime.fromtimestamp(ct.astype('O') / 1e9, datetime.timezone.utc)
            ct_num = date2num(ct_dt)
            for j, ax in enumerate(axs):
                if i == 0 and j == 0:
                    ax.axvline(ct_num, color='r', linestyle='dashdot', alpha=1, lw=0.7, label='True Crossings')
                else:
                    ax.axvline(ct_num, color='r', linestyle='dashdot', alpha=1, lw=0.7)
    
    # Convert datetime64[ns] to datetime objects properly
    times_dt = [datetime.datetime.fromtimestamp(t.astype('datetime64[s]').astype('int'), datetime.timezone.utc) for t in times_ns]
    times = date2num(times_dt)

    # Loop through each variable and plot
    for i, var in enumerate(variables):

        # Check whether variable is in the section keys
        if var not in section.keys():
            print(f"Variable '{var}' not found in section keys.")
            continue

        # Check if the variable is a torch tensor
        istorch = isinstance(section[var], torch.Tensor)

        # Convert variable to numpy array if it's a torch tensor
        if istorch:
            values = section[var].cpu().numpy()
        else:
            values = section[var]
        
        # Store original data type
        output_data[f'{var}_original_type'] = type(section[var]).__name__

        # Apply time mask to filter values
        if values.ndim == 1:
            values = values[time_mask]
        elif values.ndim == 2:
            values = values[time_mask]
        
        # Store filtered values in output dictionary
        output_data[var] = values

        # Scalar variable
        if values.ndim == 1:

            # Plot the variable
            axs[i].plot(times, values, color=colors[i % len(colors)])

            if var == 'ion_density':
                axs[i].set_ylabel('n$_i$ (cm$^{-3}$)')
            elif var == 'ion_avgtemp':
                axs[i].set_ylabel('T$_i$ (eV)')
                axs[i].set_ylim(0, 2900)
            elif var == 'ion_vthermal':
                axs[i].set_ylabel('v$_{th,i}$ (km/s)')
            elif var == 'ion_velocity_magnitude':
                axs[i].set_ylabel('|v$_i$| (km/s)')
            
            
            # Set log scale if requested
            if logscale:
                axs[i].set_yscale('log')
            
            # Remove whitespace on left and right
            axs[i].set_xlim(times[0], times[-1])

            # Only show x-axis labels and ticks if this is the last plot or only plot
            if i == n_variables - 1:
                axs[i].set_xlabel(xlabel)
            else:
                axs[i].set_xticklabels([])
                axs[i].set_xticks([])

        # Vector variable with 2 or 3 components
        elif values.ndim == 2 and values.shape[1] in [2, 3]:

            if var == 'magnetic_field_gsm':
                var_name_list = ['B$_x$', 'B$_y$', 'B$_z$']
                vector_colors = ['red', 'green', 'blue']
                axs[i].set_ylabel('B (nT)')


            # Plot each component of the variable in the same plot
            for j in range(values.shape[1]):
                axs[i].plot(times, values[:, j], label=f"{var_name_list[j]}", color=vector_colors[j])
            
            axs[i].legend(loc='right')
            
            # Set log scale if requested
            if logscale:
                axs[i].set_yscale('log')
            
            # Remove whitespace on left and right
            axs[i].set_xlim(times[0], times[-1])

            # Only show x-axis labels and ticks if this is the last plot or only plot
            if i == n_variables - 1:
                axs[i].set_xlabel(xlabel)
            else:
                axs[i].set_xticklabels([])
                axs[i].set_xticks([])


        # Spectrogram-like variable with more than 3 components
        elif values.ndim == 2 and values.shape[1] > 3:

            # Create a meshgrid for the spectrogram
            T, F = np.meshgrid(times, np.arange(values.shape[1]))
            pcm = axs[i].pcolormesh(T, F, values.T, shading='auto', cmap='jet', 
            norm=LogNorm(vmin=1e2, vmax=1e8))
            
            # Add colorbar with controlled width using constrained layout approach
            pos = axs[i].get_position()
            cbar_ax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
            fig.colorbar(pcm, cax=cbar_ax, label='eV/(cm^2-s-sr-eV)')
            
            # Only show x-axis labels and ticks if this is the last plot or only plot
            if i == n_variables - 1:
                axs[i].set_xlabel(xlabel)
            else:
                axs[i].set_xticklabels([])
                axs[i].set_xticks([])
            
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

    axs[0].set_title(title)
    axs[0].legend()

    axs[-1].xaxis_date()
    axs[-1].xaxis.set_major_locator(HourLocator(interval=hour_interval))
    axs[-1].xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')

    # Format dates and apply tight layout with minimal spacing
    fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.05)
    plt.show()

    return output_data
 
def load_position_data(sections, satellite_name='THB'):
    """
    Load position data for given sections from CDAWeb.
    
    Args:
      sections: List of dictionaries, each containing 'times' key with numpy datetime64 array.
      satellite_name: Name of the satellite (default 'THB').
    """
    # CDAWeb setup
    dataset = satellite_name + '_OR_SSC (2007-02-24 to 2025-08-31)'
    cdaweb_obj = pyspedas.CDAWeb()

    start_time = None
    end_time = None

    for i, section_key in tqdm(enumerate(sections)):

        # Get the section
        section = sections[section_key]

        # Get start and end times of the section
        section_start = section['times'][0].numpy().astype('datetime64[ns]')
        section_end = section['times'][-1].numpy().astype('datetime64[ns]')
        if start_time is None or section_start < start_time:
            start_time = section_start
        if end_time is None or section_end > end_time:
            end_time = section_end

    # Convert to string format for CDAWeb
    start_time_str = str(start_time)[:19].replace('T', ' ')
    end_time_str = str(end_time)[:19].replace('T', ' ')

    with silence_all():
        # Get the list of files for the specified time range
        urllist = cdaweb_obj.get_filenames([dataset],start_time_str, end_time_str)
        cdaweb_obj.cda_download(urllist, "cdaweb/")

def plot_crossing_radius(crossing_times, sections, satellite_name='THB', loaded=True, **kwargs):
    """
    Plot average crossing radii per section.
    Parameters:
    - crossing_times: List of numpy datetime64 objects representing crossing times.
    - sections: List of dictionaries, each containing 'times' key with numpy datetime64 array.
    - satellite_name: Name of the satellite (default 'THB').
    - kwargs: Additional keyword arguments for plt.subplots().
    """
    # CDAWeb setup
    dataset = satellite_name + '_OR_SSC (2007-02-24 to 2025-08-31)'
    cdaweb_obj = pyspedas.CDAWeb()

    # Initialize arrays to store average and standard deviation of crossing radii and average times
    avg_radii = []
    std_radii = []
    avg_times = []
    n_crossings = []
    max_radii = []
    min_radii = []




    for i, section_key in tqdm(enumerate(sections)):

        section = sections[section_key]

        # Get start and end times of the section
        section_start = section['times'][0].numpy().astype('datetime64[ns]')
        section_end = section['times'][-1].numpy().astype('datetime64[ns]')

        # Convert to string format for CDAWeb
        section_start_str = str(section_start)[:19].replace('T', ' ')
        section_end_str = str(section_end)[:19].replace('T', ' ')

        # Filter crossing times that fall within the section time range
        relevant_crossings = [ct for ct in crossing_times if section_start <= ct <= section_end]

        

        if not relevant_crossings:
            print("No crossing times found within section time range.")
            continue
        
        if not loaded:
            with silence_all():
                # Get the list of files for the specified time range
                urllist = cdaweb_obj.get_filenames([dataset],section_start_str, section_end_str)
                cdaweb_obj.cda_download(urllist, "cdaweb/")

        # Load the position data
        true_data = pyspedas.get_data('XYZ_GSM')
        pos_data_times = true_data.times.astype('datetime64[s]')
        pos_data = true_data.y

        # Calculate crossing radii
        crossing_radii = np.zeros(len(relevant_crossings))

        # Loop through each relevant crossing time
        for j, ct in enumerate(relevant_crossings):

            # Find the index of the closest time in pos_data_times to the crossing time
            idx = np.argmin(np.abs(pos_data_times - ct))
            crossing_pos = pos_data[idx]

            # Calculate the radius
            crossing_radius = np.linalg.norm(crossing_pos[1:3])
            crossing_radii[j] = crossing_radius


        # Calculate average time of the section
        avg_time = section_start + (section_end - section_start) / 2

        # Calculate average and standard deviation of crossing radii
        avg_radius = np.mean(crossing_radii)
        std_radius = np.std(crossing_radii)

        # Store results
        if len(crossing_radii) > 0:
            avg_radii.append(avg_radius)
            std_radii.append(std_radius)
            avg_times.append(avg_time)
            n_crossings.append(len(crossing_radii))
            max_radii.append(np.max(crossing_radii))
            min_radii.append(np.min(crossing_radii))

        

    
    # Convert to numpy arrays
    avg_radii = np.array(avg_radii)
    avg_times = np.array(avg_times)
    std_radii = np.array(std_radii)
    n_crossings = np.array(n_crossings)
    max_radii = np.array(max_radii)
    min_radii = np.array(min_radii)
    
    # Group and average data points that are within 10 days of each other
    if len(avg_times) > 1:
        grouped_radii = []
        grouped_times = []
        grouped_stds = []
        grouped_max = []
        grouped_min = []
        used_indices = set()
        
        for i in range(len(avg_times)):
            if i in used_indices:
                continue
                
            # Find all points within 10 days of current point
            time_diff = np.abs(avg_times - avg_times[i]).astype('timedelta64[D]').astype(int)
            close_indices = np.where(time_diff <= 10)[0]
            
            # Calculate weighted averages based on n_crossings
            weights = n_crossings[close_indices]
            weighted_radius = np.average(avg_radii[close_indices], weights=weights)
            weighted_time = avg_times[close_indices][np.argmax(weights)]  # Use time of highest weight
            weighted_std = np.sqrt(np.average(std_radii[close_indices]**2, weights=weights))
            
            grouped_radii.append(weighted_radius)
            grouped_times.append(weighted_time)
            grouped_stds.append(weighted_std)
            grouped_max.append(np.max(max_radii[close_indices]))
            grouped_min.append(np.min(min_radii[close_indices]))
            used_indices.update(close_indices)
        
        avg_radii = np.array(grouped_radii)
        avg_times = np.array(grouped_times)
        max_radii = np.array(grouped_max)
        min_radii = np.array(grouped_min)
        std_radii = np.array(grouped_stds)


    # Plotting
    fig, ax = plt.subplots(**kwargs)
    ax.errorbar(avg_times, avg_radii, yerr=std_radii, marker='o', linestyle='-', color='b', capsize=5, label='Avg Radius ± Std Dev')
    ax.plot(avg_times, max_radii, marker='^', linestyle='--', color='r', label='Max Radius')
    ax.plot(avg_times, min_radii, marker='v', linestyle='--', color='g', label='Min Radius')
    ax.set_ylabel('Average Crossing Radius (Re)')
    ax.set_title('Average Crossing Radius per Section')
    ax.legend()

    # Date formatting
    ax.xaxis_date()
    
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y'))

    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return avg_times, avg_radii, std_radii

def plot_shue_radius(X, sections=None, loaded=True, **kwargs):
    """
    Plot Shue magnetopause radius at given X positions over time for varying Bz and Pdyn values.
    
    Parameters:
    - X: X position (in Re) where the radius is calculated (negative value).
    - Bz_values: Array of Bz values (in nT).
    - Pdyn_values: Array of Pdyn values (in nPa).
    - times: Array of times corresponding to the Bz and Pdyn values.
    - kwargs: Additional keyword arguments for plt.subplots().
    """

    bz_data = pyspedas.get_data('BZ_GSM')
    pdyn_data = pyspedas.get_data('Pressure')

    # Interpolate nan values in bz_data.y and pdyn_data.y
    nans_bz, x_bz= np.isnan(bz_data.y), lambda z: z.nonzero()[0]
    bz_data.y[nans_bz]= np.interp(x_bz(nans_bz), x_bz(~nans_bz), bz_data.y[~nans_bz])

    nans_pdyn, x_pdyn= np.isnan(pdyn_data.y), lambda z: z.nonzero()[0]
    pdyn_data.y[nans_pdyn]= np.interp(x_pdyn(nans_pdyn), x_pdyn(~nans_pdyn), pdyn_data.y[~nans_pdyn])

    # Average out over 1 week (168 hours)
    Bz_values = np.convolve(bz_data.y, np.ones(168)/168, mode='valid')
    Pdyn_values = np.convolve(pdyn_data.y, np.ones(168)/168, mode='valid')

    # Adjust times accordingly
    times = bz_data.times[83:-84].astype("datetime64[s]")

    if sections is not None:
        # Calculate average values per section
        section_times = []
        section_bz_avg = []
        section_pdyn_avg = []
        
        for section_key, section in sections.items():
            section_start = section['times'][0].numpy().astype('datetime64[ns]')
            section_end = section['times'][-1].numpy().astype('datetime64[ns]')
            
            # Find data points within this section
            section_mask = (times >= section_start) & (times <= section_end)
            
            if np.any(section_mask):
                # Calculate average time for this section
                section_avg_time = section_start + (section_end - section_start) / 2
                
                # Calculate average Bz and Pdyn for this section
                section_bz = np.mean(Bz_values[section_mask])
                section_pdyn = np.mean(Pdyn_values[section_mask])
                
                section_times.append(section_avg_time)
                section_bz_avg.append(section_bz)
                section_pdyn_avg.append(section_pdyn)
        
        # Replace original arrays with section averages
        times = np.array(section_times)
        Bz_values = np.array(section_bz_avg)
        Pdyn_values = np.array(section_pdyn_avg)
        


    # Calculate Shue radii
    radii = shue_radius_at_x(X, Bz_values, Pdyn_values)

    # Create figure and axis
    fig, ax = plt.subplots(**kwargs, figsize=(10, 5))

    # Scatter plot with color mapping based on radius values
    sc = ax.plot(times, radii, marker='o', linestyle='-', color='b')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Shue Magnetopause Radius (Re)')
    ax.set_title(f'Shue Magnetopause Radius at X={X} Re Over Time')

    # Date formatting
    ax.xaxis_date()
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y'))
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_omni_data(start_time, end_time, variables=['Pressure', 'BZ_GSM', 'BY_GSM', 'BX_GSE', 'ABS_B', 'V', 'SIGMA-V', ], sections=None, avg_times=None, **kwargs):
    """
    Fetch and plot OMNI solar wind data for specified variables over a given time range.
    Parameters:
    - start_time: Start time as a string in 'YYYY-MM-DD HH:MM:SS' format.
    - end_time: End time as a string in 'YYYY-MM-DD HH:MM:SS' format.
    - variables: List of variable names to plot (e.g., ['BX_GSE', 'BY_GSE', 'BZ_GSE']).
    - kwargs: Additional keyword arguments for plt.subplots().
    """
    with silence_all():
        omni.data(trange=[start_time, end_time],  datatype='hourly')

    # Check if all magnetic field components are present
    mag_components = ['BX_GSE', 'BY_GSM', 'BZ_GSM']
    has_all_components = all(comp in variables for comp in mag_components)
    
    if has_all_components:
        print("All magnetic field components (BX_GSE, BY_GSM, BZ_GSM) detected. Calculating magnetic field magnitude...")
        # Add magnitude to variables if not already present
        if 'B_MAG_CALC' not in variables:
            variables = ['B_MAG_CALC'] + variables

    # Dictionary to map variable names to their data and colors
    var_config = {
        'V': {'color': 'orange', 'label': 'Velocity'},
        'SIGMA-V': {'color': 'blue', 'label': 'Sigma-Velocity'},
        'Pressure': {'color': 'green', 'label': 'Pressure'},
        'BZ_GSM': {'color': 'red', 'label': 'Bz_GSM'},
        'BY_GSM': {'color': 'cyan', 'label': 'By_GSM'},
        'BX_GSE': {'color': 'magenta', 'label': 'Bx_GSM'},
        'ABS_B': {'color': 'purple', 'label': 'IMF'},
        'B_MAG_CALC': {'color': 'black', 'label': 'B Magnitude (Calculated)'}
    }
    
    # Calculate magnetic field magnitude if all components are available
    if has_all_components:
        bx_data = pyspedas.get_data('BX_GSE')
        by_data = pyspedas.get_data('BY_GSM')
        bz_data = pyspedas.get_data('BZ_GSM')
        
        # Calculate magnitude
        b_magnitude = np.sqrt(bx_data.y**2 + by_data.y**2 + bz_data.y**2)
    
    # Create figure with subplots
    fig, axs = plt.subplots(len(variables), 1, figsize=(10, 2.4 * len(variables)), **kwargs)
    
    # Handle case where there's only one subplot
    if len(variables) == 1:
        axs = [axs]
    
    # Loop through each variable
    for i, var in enumerate(variables):
        if var == 'B_MAG_CALC' and has_all_components:
            # Plot calculated magnetic field magnitude
            data_averaged = np.convolve(b_magnitude, np.ones(720)/720, mode='valid')
            axs[i].plot(bx_data.times[359:-360].astype("datetime64[s]"), data_averaged, 
                       label=f'24-hour Averaged {var_config[var]["label"]}', 
                       color=var_config[var]['color'])
            axs[i].set_ylabel(var_config[var]['label'])
            axs[i].legend()
        elif var in var_config and var != 'B_MAG_CALC':
            # Get data
            data = pyspedas.get_data(var)
            
            # Interpolate nan values in data.y
            nans, x= np.isnan(data.y), lambda z: z.nonzero()[0]
            data.y[nans]= np.interp(x(nans), x(~nans), data.y[~nans])

            # Apply 24-hour averaging (720 points for hourly data)
            data_averaged = np.convolve(data.y, np.ones(720)/720, mode='valid')
            
            # Plot the data
            axs[i].plot(data.times[359:-360].astype("datetime64[s]"), data_averaged, 
                       label=f'24-hour Averaged {var_config[var]["label"]}', 
                       color=var_config[var]['color'])
            axs[i].set_ylabel(var_config[var]['label'])
            axs[i].legend()
        else:
            print(f"Variable '{var}' not found in configuration.")

        if sections is not None:
            # Plot vertical lines for section boundaries, and plot average variables within each section
            for section_key, section in sections.items():
                # Get section start and end times
                section_start = section['times'][0].numpy().astype('datetime64[ns]')
                section_end = section['times'][-1].numpy().astype('datetime64[ns]')
                
                # Plot vertical lines for section boundaries
                for ax in axs:
                    ax.axvline(section_start, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                    ax.axvline(section_end, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    
    # Set common properties
    axs[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()