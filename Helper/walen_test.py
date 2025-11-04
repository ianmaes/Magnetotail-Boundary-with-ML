import numpy as np

MU0 = 4e-7 * np.pi
MP  = 1.67262192369e-27  # kg

def _ensure_array(x, N):
    return np.full(N, float(x)) if np.isscalar(x) else np.asarray(x, float).reshape(-1)

def walen_simple(B_nT, V_kms, n_cm3, ion_mass_amu=1.0, demean=True):
    """
    Minimal Walén (Alfvén) test on arrays you provide.
    Units: B in nT, V in km/s, n in cm^-3.
    Returns dict with slope alpha, correlation r, per-component stats, etc.
    """
    B_nT = np.asarray(B_nT, float)
    V_kms = np.asarray(V_kms, float)

    if np.ndim(n_cm3) == 0:
        n_arr = np.full(B_nT.shape[0], float(n_cm3))
    else:
        n_arr = np.asarray(n_cm3, float).reshape(-1)
        assert n_arr.shape[0] == B_nT.shape[0], "n must be scalar or length N"

    assert B_nT.shape == V_kms.shape and B_nT.shape[1] == 3, "B and V must be (N,3)"

    # --- SI units ---
    B_T  = B_nT * 1e-9                # T
    V_ms = V_kms * 1e3                # m/s
    rho  = n_arr * 1e6 * ion_mass_amu * MP   # kg/m^3

    # --- fluctuations (global mean by default) ---
    if demean:
        dB = B_T  - np.nanmean(B_T,  axis=0, keepdims=True)
        dV = V_ms - np.nanmean(V_ms, axis=0, keepdims=True)
    else:
        dB, dV = B_T, V_ms

    # --- convert magnetic fluctuations to "Alfvén units" (velocity) ---
    VAfac = 1.0 / np.sqrt(MU0 * rho)           # (m^3/kg)^0.5
    dV_B  = dB * VAfac[:, None]                 # m/s

    dv   = dV   / 1e3                           # km/s
    dv_b = dV_B / 1e3                           # km/s

    # --- vector fit through origin: δv ≈ α δv_B ---
    x = dv_b.reshape(-1)
    y = dv.reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    alpha = (x @ y) / (x @ x) if (x @ x) > 0 else np.nan
    r = np.corrcoef(x, y)[0, 1] if x.size > 2 else np.nan

    # per-component slopes & correlations
    slopes = []; corrs = []
    for k in range(3):
        xk = dv_b[:, k]; yk = dv[:, k]
        mk = np.isfinite(xk) & np.isfinite(yk)
        xk = xk[mk]; yk = yk[mk]
        if xk.size < 2:
            slopes.append(np.nan); corrs.append(np.nan)
        else:
            slopes.append((xk @ yk) / (xk @ xk))
            corrs.append(np.corrcoef(xk, yk)[0, 1])

    # diagnostics
    zplus  = dv + dv_b
    zminus = dv - dv_b
    e_zp = np.nanmean(np.sum(zplus**2,  axis=1))
    e_zm = np.nanmean(np.sum(zminus**2, axis=1))
    sigma_c = (e_zp - e_zm) / (e_zp + e_zm)          # cross-helicity in [-1,1]
    ev = np.nanmean(np.sum(dv**2,   axis=1))
    eb = np.nanmean(np.sum(dv_b**2, axis=1))
    sigma_r = (ev - eb) / (ev + eb)                  # residual energy ~0 for Alfvénic

    Bmag = np.linalg.norm(B_nT, axis=1)              # nT
    Bmag_CV = float(np.nanstd(Bmag) / np.nanmean(Bmag)) if np.isfinite(Bmag).all() else np.nan

    return {
        "alpha": float(alpha),                       # expect ≈ +1 or −1
        "r": float(r),                               # |r| → 1 if Alfvénic
        "slopes_xyz": tuple(float(s) for s in slopes),
        "corrs_xyz": tuple(float(c) for c in corrs),
        "sigma_c": float(sigma_c),                   # → ±1 if strongly Alfvénic
        "sigma_r": float(sigma_r),                   # → 0 if Alfvénic
        "Bmag_CV": Bmag_CV,                          # small if |B| ~ const
        "note": "Ideal Alfvénic: alpha≈±1, |r|≳0.7–0.8, |sigma_c|→1, sigma_r≈0, small Bmag_CV."
    }

def _to_si(B_nT, V_kms, n, density_units='cm^-3', ion_mass_amu=1.0):
    B_T = np.asarray(B_nT, float) * 1e-9
    V_mps = np.asarray(V_kms, float) * 1e3
    n_arr = np.asarray(n, float)
    if density_units == 'cm^-3':
        n_m3 = n_arr * 1e6
    elif density_units == 'm^-3':
        n_m3 = n_arr
    else:
        raise ValueError("density_units must be 'cm^-3' or 'm^-3'")
    rho = n_m3 * (ion_mass_amu * MP)
    return B_T, V_mps, rho

def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a).all(axis=1) if a.ndim == 2 else np.isfinite(a)
    return m

def estimate_ht_velocity(B_T, V_mps):
    """Least-squares de Hoffmann–Teller velocity."""
    A = np.zeros((3,3)); c = np.zeros(3)
    for b, v in zip(B_T, V_mps):
        b2 = np.dot(b,b)
        M = b2*np.eye(3) - np.outer(b,b)
        A += M; c += M @ v
    V_HT = np.linalg.lstsq(A, c, rcond=None)[0]
    Eprime = np.cross(V_mps - V_HT, B_T)
    qual = np.sqrt((Eprime**2).sum(axis=1)).mean()
    return V_HT, qual

def walen_test_once(B_nT, V_kms, n, *, density_units='cm^-3', ion_mass_amu=1.0,
                    use_ht=True, mean_mode='global'):
    """Return Walén/Alfvén diagnostics over one interval."""
    B_nT = np.asarray(B_nT, float); V_kms = np.asarray(V_kms, float)
    N = len(B_nT); n = _ensure_array(n, N)
    B_T, V_mps, rho = _to_si(B_nT, V_kms, n, density_units, ion_mass_amu)

    # optional HT frame
    V_HT = np.zeros(3); ht_q = np.nan
    if use_ht:
        V_HT, ht_q = estimate_ht_velocity(B_T, V_mps)
        V_mps = V_mps - V_HT

    # fluctuations
    if mean_mode == 'global':
        B0 = np.nanmean(B_T, axis=0, keepdims=True)
        V0 = np.nanmean(V_mps, axis=0, keepdims=True)
    else:
        raise ValueError("mean_mode supports only 'global' here for simplicity.")
    dB = B_T - B0
    dV = V_mps - V0

    # convert magnetic fluctuations to velocity units
    VAfac = 1.0 / np.sqrt(MU0 * rho)
    dV_B = dB * VAfac[:,None]  # m/s
    dV_B_kms = dV_B / 1e3
    dV_kms = dV / 1e3

    # vector slope & correlation
    num = np.sum(np.einsum('ij,ij->i', dV_kms, dV_B_kms))
    den = np.sum(np.einsum('ij,ij->i', dV_B_kms, dV_B_kms))
    alpha = num/den if den>0 else np.nan
    r = num / np.sqrt(den * np.sum(np.einsum('ij,ij->i', dV_kms, dV_kms))) if den>0 else np.nan

    # component slopes/correlations
    slopes = []; corrs = []
    for k in range(3):
        x = dV_B_kms[:,k]; y = dV_kms[:,k]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        den_k = np.dot(x,x); num_k = np.dot(x,y)
        slope_k = num_k/den_k if den_k>0 else np.nan
        xm = x - x.mean(); ym = y - y.mean()
        denom = np.sqrt(np.dot(xm,xm)*np.dot(ym,ym))
        corr_k = (np.dot(xm,ym)/denom) if denom>0 else np.nan
        slopes.append(slope_k); corrs.append(corr_k)

    # Elsässer diagnostics
    zplus  = dV_kms + dV_B_kms
    zminus = dV_kms - dV_B_kms
    e_zp = np.mean(np.sum(zplus**2,  axis=1))
    e_zm = np.mean(np.sum(zminus**2, axis=1))
    sigma_c = (e_zp - e_zm) / (e_zp + e_zm)
    ev = np.mean(np.sum(dV_kms**2, axis=1))
    eb = np.mean(np.sum(dV_B_kms**2, axis=1))
    sigma_r = (ev - eb) / (ev + eb)
    cos_theta = np.einsum('ij,ij->i', dV_kms, dV_B_kms) / (
        np.linalg.norm(dV_kms,axis=1)*np.linalg.norm(dV_B_kms,axis=1) + 1e-30
    )
    cos_theta_mean = np.nanmean(cos_theta)

    Bmag = np.linalg.norm(B_T, axis=1)
    Bmag_CV = np.std(Bmag)/np.mean(Bmag)

    return {
        "alpha_vector_slope": float(alpha),
        "vector_correlation_r": float(r),
        "component_slopes_xyz": tuple(slopes),
        "component_correlations_xyz": tuple(corrs),
        "cross_helicity_sigma_c": float(sigma_c),
        "residual_energy_sigma_r": float(sigma_r),
        "mean_alignment_cos_theta": float(cos_theta_mean),
        "Bmag_coefficient_of_variation": float(Bmag_CV),
        "HT_velocity_mps": V_HT if use_ht else None,
        "HT_residual_metric": float(ht_q) if use_ht else None,
        "note": "Ideal Alfvénic: slope≈±1, |r|→1, |sigma_c|→1, sigma_r≈0, |B| ~ const (low CV)."
    }

def sliding_walen(times_dt64, B_nT, V_kms, n, *,
                  window_seconds=180, step_seconds=30,
                  density_units='cm^-3', ion_mass_amu=1.0,
                  use_ht=True, min_samples=50):
    """Scan for best Walén windows."""
    t_ns = times_dt64.astype('datetime64[ns]').astype(np.int64)
    t0 = t_ns.min()
    t = (t_ns - t0) * 1e-9  # seconds from start, float

    out = []
    t_min = t.min(); t_max = t.max()
    w = window_seconds; s = step_seconds

    i0 = 0
    while True:
        start = t_min + i0*s
        end = start + w
        if end > t_max + 1e-6:
            break
        idx = (t >= start) & (t < end)
        if idx.sum() >= min_samples:
            res = walen_test_once(B_nT[idx], V_kms[idx], _ensure_array(n, len(B_nT))[idx],
                                  density_units=density_units, ion_mass_amu=ion_mass_amu,
                                  use_ht=use_ht, mean_mode='global')
            out.append({
                "start_s": float(start), "end_s": float(end),
                "alpha": res["alpha_vector_slope"],
                "r": res["vector_correlation_r"],
                "sigma_c": res["cross_helicity_sigma_c"],
                "sigma_r": res["residual_energy_sigma_r"],
                "Bmag_CV": res["Bmag_coefficient_of_variation"]
            })
        i0 += 1
        if start + w >= t_max:
            break
    return out