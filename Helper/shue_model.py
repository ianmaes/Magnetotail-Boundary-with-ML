import numpy as np
import pyspedas
from scipy.optimize import fsolve

def shue_magnetopause_surface(Bz_nT=0.0, Pdyn_nPa=2.0, n_theta=180, n_phi=144,
                Xmin=-80.0, Xmax=10.0):
  """
  Returns X,Y,Z for a 3D magnetopause surface using the Shue-form profile:
    r(θ) = r0 * (2 / (1 + cosθ))**α
  with r0 and α parameterized by Bz and dynamic pressure.
  """
  # General-assumption parameters (Bz in nT, Pdyn in nPa)
  r0 = (10.22 + 1.29 * np.exp(0.184 * Bz_nT)) * (Pdyn_nPa ** (-1.0 / 6.6))
  alpha = 0.58 - 0.007 * Bz_nT

  theta = np.linspace(0.0, np.pi, n_theta)        # 0 = subsolar, pi = tail
  phi   = np.linspace(0.0, 2.0*np.pi, n_phi)      # azimuthal rotation
  TH, PH = np.meshgrid(theta, phi, indexing='ij')

  R = r0 * (2.0 / (1.0 + np.cos(TH)))**alpha
  X = R * np.cos(TH)
  Y = R * np.sin(TH) * np.cos(PH)
  Z = R * np.sin(TH) * np.sin(PH)

  # Keep only the chunk within your plotting window
  mask = (X >= Xmin) & (X <= Xmax)
  X = np.where(mask, X, np.nan)
  Y = np.where(mask, Y, np.nan)
  Z = np.where(mask, Z, np.nan)
  return X, Y, Z

def shue_radius_at_x(x_target, Bz_nT=0.0, Pdyn_nPa=2.0):
  """
  Returns the magnetopause Y-Z cross-section radius at a specific X position (e.g., X = -60 Re).
  For the Shue model: r(θ) = r0 * (2 / (1 + cosθ))**α
  where X = r*cosθ, so θ = arccos(X/r).
  
  This requires solving: x_target = r*cosθ where r = r0*(2/(1+cosθ))**α
  Returns the radius in the Y-Z plane at the given X position.
  
  Args:
    x_target: Target X position (scalar)
    Bz_nT: IMF Bz in nT (scalar or array)
    Pdyn_nPa: Dynamic pressure in nPa (scalar or array)
  
  Returns:
    yz_radius: Y-Z plane radius (scalar if inputs are scalars, array if inputs are arrays)
  """
  Bz_nT = np.atleast_1d(Bz_nT)
  Pdyn_nPa = np.atleast_1d(Pdyn_nPa)
  
  # Broadcast to common shape
  Bz_nT, Pdyn_nPa = np.broadcast_arrays(Bz_nT, Pdyn_nPa)
  
  # Initialize output array
  yz_radius = np.zeros_like(Bz_nT, dtype=float)
  
  # Process each element
  for i in range(Bz_nT.size):
    r0_i, alpha_i = _shue_params(Bz_nT.flat[i], Pdyn_nPa.flat[i])
    
    def equation(theta):
      r = r0_i * (2.0 / (1.0 + np.cos(theta)))**alpha_i
      return r * np.cos(theta) - x_target
    
    # Initial guess: for negative X, theta > π/2
    theta_guess = np.pi - 0.1
    theta_solution = fsolve(equation, theta_guess)[0]
    
    # Calculate the Y-Z plane radius at this X position
    r_total = r0_i * (2.0 / (1.0 + np.cos(theta_solution)))**alpha_i
    yz_radius.flat[i] = r_total * np.sin(theta_solution)
  
  # Return scalar if input was scalar
  if yz_radius.size == 1:
    return yz_radius.item()
  return yz_radius

# --- Magnetopause curve in the X–Z plane (Shue et al.-style) ---
def shue_magnetopause_xz(Bz_nT=0.0, Pdyn_nPa=2.0, n_theta=720):
    """
    2D X–Z cut (Y=0) of the Shue magnetopause model:
      r(θ) = r0 * (2 / (1 + cosθ))**α
    with r0 and α parameterized by Bz and dynamic pressure.
    Returns arrays X(θ), Z(θ) for θ ∈ [0, π].
    """
    r0 = (10.22 + 1.29 * np.exp(0.184 * Bz_nT)) * (Pdyn_nPa ** (-1.0 / 6.6))
    alpha = 0.58 - 0.007 * Bz_nT
    theta = np.linspace(0.0, np.pi, n_theta)  # 0=subsolar (+X), π=tail (−X)
    R = r0 * (2.0 / (1.0 + np.cos(theta)))**alpha
    X = R * np.cos(theta)
    Z = R * np.sin(theta)  # φ = 90° → Y = 0 plane
    return X, Z

# Shared parameterization
def _shue_params(Bz_nT=0.0, Pdyn_nPa=2.0):
    r0 = (10.22 + 1.29 * np.exp(0.184 * Bz_nT)) * (Pdyn_nPa ** (-1.0 / 6.6))
    alpha = 0.58 - 0.007 * Bz_nT
    return r0, alpha

def shue_magnetopause_xy(Bz_nT=0.0, Pdyn_nPa=2.0, n_theta=720):
    """
    2D X–Y cut (Z=0 plane) of the Shue magnetopause model.
    Returns X(θ), Y(θ) for θ ∈ [0, π]. (Upper half; mirror Y for lower.)
    """
    r0, alpha = _shue_params(Bz_nT, Pdyn_nPa)
    theta = np.linspace(0.0, np.pi, n_theta)   # 0=subsolar (+X), π=tail (−X)
    R = r0 * (2.0 / (1.0 + np.cos(theta)))**alpha
    X = R * np.cos(theta)
    Y = R * np.sin(theta)                      # φ = 0 → Z = 0 plane
    return X, Y

def shue_magnetopause_yz(Bz_nT=0.0, Pdyn_nPa=2.0, n_phi=720):
    """
    2D Y–Z cut (X=0 plane) of the Shue magnetopause model.
    At θ = π/2 the surface intersects the Y–Z plane as a circle
    of radius R_eq = r0 * 2**α. Returns Y(φ), Z(φ) for φ ∈ [0, 2π].
    """
    r0, alpha = _shue_params(Bz_nT, Pdyn_nPa)
    R_eq = r0 * (2.0 / (1.0 + np.cos(np.pi/2)))**alpha  # = r0 * 2**α
    phi = np.linspace(0.0, 2.0*np.pi, n_phi)
    Y = R_eq * np.cos(phi)
    Z = R_eq * np.sin(phi)
    return Y, Z

def filter_train_samples_by_time(train_samples, dataset, boundary_time, method='less_than', data_origin_keep_always='Themis_C', data_origin_exclude_always=None):
    """
    Filter train samples based on boundary time and data origin.
    
    Args:
      train_samples: Dictionary of training samples
      dataset: Dataset object with get_session_info method
      boundary_time: numpy datetime64 boundary time
      method: Method to filter the data ('less_than' or 'more_than')
    
    Returns:
      Filtered train_samples dictionary
    """
    train_samples_keys = list(train_samples.keys())
    train_samples_values = list(train_samples.values())
    
    filtered_samples = {}
    for i in range(len(train_samples_keys)):
      sample_time = train_samples_values[i]['times'][0].numpy().astype('datetime64[ns]')
      data_origin = dataset.get_session_info()[i]['data_origin']
      
      if method == 'less_than':
        time_condition = sample_time < boundary_time
      elif method == 'more_than':
        time_condition = sample_time > boundary_time
      else:
        raise ValueError("method must be 'less_than' or 'more_than'")
      
      if (time_condition or data_origin == data_origin_keep_always) and (data_origin_exclude_always is None or data_origin != data_origin_exclude_always):
        filtered_samples[train_samples_keys[i]] = train_samples_values[i]
    
    return filtered_samples

def get_solar_wind_conditions(start_time, end_time):
    """
    Fetch solar wind conditions from OMNI data using pyspedas.
    
    Args:
      start_time: Start time as a string (e.g., '2020-01-01T00:00:00')
      end_time: End time as a string (e.g., '2020-01-02T00:00:00')
    """

    # CONTINUE HERE
    
    