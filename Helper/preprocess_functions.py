import torch
import numpy as np
from copy import deepcopy

def exponential_moving_average(data, alpha=0.333, padding_mode='edge'):
    """
    Apply exponential moving average to 1D data with optional padding.
    
    Args:
        data (array-like): Input data (numpy array or torch tensor)
        alpha (float): Smoothing factor between 0 and 1. Higher values give more weight to recent observations.
        padding_mode (str): Padding mode for the start of the data. Options:
            - 'edge': Replicate edge values (default)
            - 'reflect': Reflect values around the edge
            - 'constant': Use constant value (first value)
            - 'none': No padding (original behavior)
    
    Returns:
        numpy.ndarray: Exponentially smoothed data
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    data = np.asarray(data, dtype=np.float64)
    
    if len(data) == 0:
        return data
    
    result = np.zeros_like(data)
    
    # Calculate effective window size for padding (typical EMA effective window)
    effective_window = int(3 / alpha) if alpha > 0 else 10
    pad_size = min(effective_window, len(data) - 1)
    
    if padding_mode == 'none' or pad_size == 0:
        # Original behavior - no padding
        result[0] = data[0]
    elif padding_mode == 'edge':
        # Replicate first value
        result[0] = data[0]
        for i in range(1, min(pad_size + 1, len(data))):
            result[i] = alpha * data[i] + (1 - alpha) * data[0]
    elif padding_mode == 'reflect':
        # Reflect around the first value
        result[0] = data[0]
        for i in range(1, min(pad_size + 1, len(data))):
            if i < len(data):
                reflected_val = 2 * data[0] - data[min(i, len(data) - 1)]
                result[i] = alpha * data[i] + (1 - alpha) * reflected_val
    elif padding_mode == 'constant':
        # Use first value as constant
        result[0] = data[0]
        for i in range(1, min(pad_size + 1, len(data))):
            result[i] = alpha * data[i] + (1 - alpha) * data[0]
    
    # Continue with normal EMA for the rest
    start_idx = min(pad_size + 1, len(data)) if padding_mode != 'none' else 1
    for i in range(start_idx, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result

def average_window(data, window_size=3):
    """
    Window averages tensor data across the time dimension (first dimension).
    
    Args:
        data (torch.Tensor): Input tensor with shape [L, ...] where L is the time dimension.
        window_size (int): Size of the averaging window. Default is 3.
    
    Returns:
        torch.Tensor: Window-averaged tensor with time dimension reduced to L - window_size + 1.
    """
    
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)

    
    if data is None:
        return None
        
    # Get tensor dimensions
    shape = data.shape
    L = shape[0]  # Time dimension
    
    # Calculate output time dimension
    L_out = L - window_size + 1
    
    if L_out <= 0:
        raise ValueError(f"Window size {window_size} is too large for time dimension {L}")
    
    # Initialize output tensor based on data dimensionality
    if len(shape) == 1:  # Scalar data [L]
        averaged_tensor = torch.zeros(L_out, device=data.device, dtype=data.dtype)
    elif len(shape) == 2:  # Spectrogram data [L, D] or Vector data [L, vector_dim]
        averaged_tensor = torch.zeros(L_out, shape[1], device=data.device, dtype=data.dtype)
    else:
        raise ValueError(f"Unsupported tensor shape: {shape}")
    
    # Perform window averaging
    for t in range(L_out):
        window_data = data[t:t+window_size]  # Extract window [window_size, ...]
        averaged_tensor[t] = window_data.mean(dim=0)  # Average across time dimension
    
    return averaged_tensor

def remove_outliers_with_interpolation(data, n_std=3):
    """
    Remove outliers from tensor data and replace with interpolated values.
    
    Args:
        data (torch.Tensor): Input tensor data (1D)
        n_std (float): Number of standard deviations to use as threshold. Default is 3.
    
    Returns:
        torch.Tensor: Cleaned data with outliers replaced by interpolated values
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)

    
    # Calculate the mean and standard deviation
    mean_data = data.mean().item()
    std_data = data.std().item()

    # Remove all values above n standard deviations from the mean and replace with interpolated values
    threshold = mean_data + n_std * std_data
    outlier_mask = data > threshold

    if outlier_mask.any():
        # Create a copy to modify
        data_cleaned = data.clone()
        data_cleaned[outlier_mask] = float('nan')
        
        # Get indices of valid (non-NaN) and outlier values
        valid_indices = torch.where(~torch.isnan(data_cleaned))[0]
        outlier_indices = torch.where(torch.isnan(data_cleaned))[0]
        
        if len(valid_indices) > 1 and len(outlier_indices) > 0:
            # Interpolate outlier values using linear interpolation
            data_cleaned[outlier_indices] = torch.tensor(
                np.interp(
                    outlier_indices.cpu().numpy(),
                    valid_indices.cpu().numpy(),
                    data[valid_indices].cpu().numpy()
                ), 
                dtype=data.dtype
            ).to(data.device)
        
        return data_cleaned
    
    return data

def remove_outliers_with_local_interpolation(data, n_std=3, window_size=20):
    """
    Remove outliers from tensor data using local mean and standard deviation 
    within a surrounding window, and replace with interpolated values.
    
    Args:
        data (torch.Tensor): Input tensor data (1D)
        n_std (float): Number of standard deviations to use as threshold. Default is 3.
        window_size (int): Number of timestamps to use for local statistics. Default is 20.
    
    Returns:
        torch.Tensor: Cleaned data with outliers replaced by interpolated values
    """
    if len(data) < window_size:
        # Fallback to global statistics if data is too short
        return remove_outliers_with_interpolation(data, n_std)
    
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)

    data_cleaned = data.clone()
    outlier_mask = torch.zeros_like(data, dtype=torch.bool)
    
    # Calculate local statistics for each point
    for i in range(len(data)):
        # Define window bounds
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        
        # Extract local window
        local_window = data[start_idx:end_idx]
        
        # Calculate local mean and std
        local_mean = local_window.mean().item()
        local_std = local_window.std().item()
        
        # Check if current point is an outlier
        threshold = local_mean + n_std * local_std
        if data[i] > threshold:
            outlier_mask[i] = True
    
    if outlier_mask.any():
        # Set outliers to NaN
        data_cleaned[outlier_mask] = float('nan')
        
        # Get indices of valid (non-NaN) and outlier values
        valid_indices = torch.where(~torch.isnan(data_cleaned))[0]
        outlier_indices = torch.where(torch.isnan(data_cleaned))[0]
        
        if len(valid_indices) > 1 and len(outlier_indices) > 0:
            # Interpolate outlier values using linear interpolation
            data_cleaned[outlier_indices] = torch.tensor(
                np.interp(
                    outlier_indices.cpu().numpy(),
                    valid_indices.cpu().numpy(),
                    data[valid_indices].cpu().numpy()
                ), 
                dtype=data.dtype
            ).to(data.device)
        
        return data_cleaned
    
    return data

def downweight_variable(
    variable: torch.Tensor,
    kappa=None,
    p: float = 2.0,
    win: int = 5,
    smoother_win: int = 15,
    *,
    mode: str = "rational",     # "rational", "exp", or "logistic"
    wmin: float = 0.05,         # clamp weights to [wmin, <1)  (lower = harsher)
    kappa_pct: float = None,    # if set, use this percentile of |dvar/dt| for kappa (e.g., 95 for softer, 80 for harsher)
    logistic_a: float = 8.0,    # steepness for logistic mode
    blend_alpha: float = 1.0,   # scales how strongly w trusts the raw variable (alpha<1 = harsher)
    use_forward_diff: bool = False  # if True, use forward difference instead of centered
):
    """
    Downweight a variable based on its temporal gradient to reduce noise.

    Args:
        variable (torch.Tensor): 1D tensor.
        kappa (float, optional): Scale where downweighting "kicks in".
            If None and kappa_pct is None: uses median + 2*std (mid).
            If kappa_pct is set: uses that percentile of |grad|.
        p (float): Sharpness of decay. Higher = harsher (faster drop).
        win (int): Gradient window (smaller -> larger gradients -> harsher).
        smoother_win (int): Smoother window (larger -> more smoothing overall).
        mode (str): "rational" -> w = 1/(1+(x/kappa)^p)  (softer tails)
                    "exp"      -> w = exp(-(x/kappa)^p)   (harsher tails)
                    "logistic" -> w = 1/(1+exp(a*(x-kappa))) (very sharp step)
        wmin (float): Minimum weight clamp. Lower -> harsher.
        kappa_pct (float): Percentile for kappa (e.g., 95 softer, 80 harsher).
        logistic_a (float): Logistic steepness (larger -> harsher transition).
        blend_alpha (float): Multiplies w in the blend. alpha<1 -> harsher.
                             Use 1.0 to match the original template.
        use_forward_diff (bool): If True, use forward difference y(t+1)-y(t) and assign
                               weight to point at t+1. If False, use centered difference.

    Returns:
        tuple: (downweighted_variable, weights, gradients)
               All are torch tensors on the same device/dtype as input.
    """
    # ---- to numpy for processing ----
    var_np = variable.detach().cpu().numpy()
    target_len = len(var_np)

    # ---- gradient calculation ----
    if use_forward_diff:
        # Forward difference: y(t+1) - y(t)
        dvar_dt = np.abs(np.diff(var_np))  # This gives us n-1 differences
        # Pad with the last difference to match original length
        dvar_dt = np.append(dvar_dt, dvar_dt[-1] if len(dvar_dt) > 0 else 0.0)
        
        # Weight assignment: dvar_dt[i] corresponds to the difference from point i to i+1
        # So the weight for point i+1 should be based on dvar_dt[i]
        # Shift weights: w[i+1] = weight_based_on(dvar_dt[i])
        weights_shifted = True
    else:
        # Centered finite difference (original approach)
        if win % 2 == 0:
            win += 1
            print(f"Warning: win must be odd, increased to {win}")

        k = win // 2  # Half window for centered difference
        var_pad = np.pad(var_np, (k, k), mode='edge')
        dvar_dt = np.abs((var_pad[2*k:] - var_pad[:-2*k]) / (2*k))
        
        # Ensure exact length match
        if len(dvar_dt) != target_len:
            if len(dvar_dt) > target_len:
                dvar_dt = dvar_dt[:target_len]
            else:
                dvar_dt = np.pad(dvar_dt, (0, target_len - len(dvar_dt)), mode='edge')
        
        weights_shifted = False

    # ---- kappa calculation ----
    if kappa is None:
        if kappa_pct is not None:
            kappa = np.percentile(dvar_dt, kappa_pct)
        else:
            kappa = np.median(dvar_dt) + 2.0 * np.std(dvar_dt)
        
        # Normalize kappa to be window-invariant
        kappa = float(max(kappa, 1e-12))

    # ---- weights calculation ----
    x = dvar_dt / kappa
    if mode == "exp":
        w = np.exp(-(np.power(x, p)))
    elif mode == "logistic":
        w = 1.0 / (1.0 + np.exp(logistic_a * (x - 1.0)))
    else:  # "rational" (default)
        w = 1.0 / (1.0 + np.power(x, p))

    # For forward difference, shift weights so w[i+1] is based on dvar_dt[i]
    if weights_shifted:
        w_shifted = np.ones_like(w)
        w_shifted[1:] = w[:-1]  # w[i+1] = weight based on difference from i to i+1
        w_shifted[0] = w[0]     # Keep first weight unchanged
        w = w_shifted

    # clamp weights (avoid 0 or 1 exactly)
    w = np.clip(w, wmin, 0.999)

    # ---- simple box smoother ----
    sw = max(1, smoother_win)
    pad = sw // 2
    var_pad_smooth = np.pad(var_np, (pad, pad), mode='edge')
    csum = np.cumsum(np.insert(var_pad_smooth, 0, 0.0))
    var_smooth = (csum[sw:] - csum[:-sw]) / sw

    # Ensure all arrays have the same length as the original variable
    if len(w) > target_len:
        w = w[:target_len]
    elif len(w) < target_len:
        w = np.pad(w, (0, target_len - len(w)), mode='edge')
    
    if len(var_smooth) > target_len:
        var_smooth = var_smooth[:target_len]
    elif len(var_smooth) < target_len:
        var_smooth = np.pad(var_smooth, (0, target_len - len(var_smooth)), mode='edge')
    
    if len(dvar_dt) > target_len:
        dvar_dt = dvar_dt[:target_len]

    # ---- blend control via blend_alpha ----
    a_w = np.clip(blend_alpha * w, 0.0, 0.999)
    var_downweighted = a_w * var_np + (1.0 - a_w) * var_smooth

    # ---- back to torch on original device/dtype ----
    dev = variable.device
    dtp = variable.dtype
    return (
        torch.tensor(var_downweighted, dtype=dtp, device=dev),
        torch.tensor(w, dtype=dtp, device=dev),
        torch.tensor(dvar_dt, dtype=dtp, device=dev),
    )
