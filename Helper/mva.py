import numpy as np
from dataclasses import dataclass

@dataclass
class MVAResult:
    # Columns of R are the unit basis vectors L, M, N in original coords
    R: np.ndarray             # shape (3,3)  [L | M | N]
    L: np.ndarray             # shape (3,)
    M: np.ndarray             # shape (3,)
    N: np.ndarray             # shape (3,)
    eigvals: np.ndarray       # [λ_L, λ_M, λ_N] (descending)
    lambda_ratios: tuple      # (λ_L/λ_M, λ_M/λ_N)
    B_mean: np.ndarray        # mean(B) used for demeaning
    B_lmn: np.ndarray | None  # B projected into LMN (demeaned), shape (N,3)

def mva(B: np.ndarray,
        weights: np.ndarray | None = None,
        demean: bool = True,
        return_projected: bool = True,
        ignore_nan: bool = True) -> MVAResult:
    """
    Minimum Variance Analysis on magnetic field vectors.

    Parameters
    ----------
    B : (N,3) array_like
        Magnetic field time series [Bx, By, Bz] in any consistent units.
    weights : (N,) array_like, optional
        Optional sample weights (e.g., equal spacing -> None; or dt).
    demean : bool
        If True, subtract the mean field before building the covariance.
    return_projected : bool
        If True, also return the demeaned data projected into LMN.
    ignore_nan : bool
        If True, drop rows with any NaNs before analysis.

    Returns
    -------
    MVAResult
        Contains rotation matrix R=[L M N], eigenvalues, ratios, mean(B),
        and B_lmn (if return_projected=True).
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[1] != 3:
        raise ValueError("B must be an array of shape (N,3).")

    if ignore_nan:
        mask = np.all(np.isfinite(B), axis=1)
        if weights is not None:
            mask &= np.isfinite(weights)
        B = B[mask]
        if weights is not None:
            weights = np.asarray(weights, dtype=float)[mask]

    if B.shape[0] < 5:
        raise ValueError("Need at least 5 samples for a stable MVA.")

    # Mean and demean
    if weights is None:
        B_mean = B.mean(axis=0)
    else:
        w = weights / np.sum(weights)
        B_mean = (w[:, None] * B).sum(axis=0)

    X = B - B_mean if demean else B.copy()

    # Weighted covariance (3x3)
    if weights is None:
        C = (X.T @ X) / (X.shape[0] - 1)
    else:
        w = weights / np.sum(weights)
        # “biased” weighted covariance; for large N this is fine for MVA
        C = (X * w[:, None]).T @ X

    # Symmetrize for numerical cleanliness
    C = 0.5 * (C + C.T)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(C)  # returns ascending
    # Sort to descending: λ_L >= λ_M >= λ_N
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    L, M, N = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]

    # Ensure right-handed LMN (optional but common convention)
    if np.dot(np.cross(L, M), N) < 0:
        N = -N
        eigvecs[:, 2] = N

    R = eigvecs  # columns are L, M, N

    B_lmn = X @ R if return_projected else None
    ratios = (eigvals[0] / eigvals[1], eigvals[1] / eigvals[2])

    return MVAResult(R=R, L=L, M=M, N=N,
                     eigvals=eigvals,
                     lambda_ratios=ratios,
                     B_mean=B_mean,
                     B_lmn=B_lmn)

# --- convenience helpers ---

def project_to_lmn(B: np.ndarray, R: np.ndarray, subtract_mean: np.ndarray | None = None) -> np.ndarray:
    """
    Project any B (N,3) into a previously obtained LMN basis (columns of R).
    If subtract_mean is given, subtract it first (useful to be consistent with MVA).
    """
    Bp = B - subtract_mean if subtract_mean is not None else B
    return Bp @ R

def quality_flags(mva: MVAResult, min_ratio_LM: float = 3.0, min_ratio_MN: float = 3.0) -> dict:
    """
    Simple quality assessment: large λ_L/λ_M and λ_M/λ_N imply a well-defined normal.
    """
    rLM, rMN = mva.lambda_ratios
    return {
        "good_normal": rMN >= min_ratio_MN,
        "good_L": rLM >= min_ratio_LM,
        "ratios": (rLM, rMN)
    }