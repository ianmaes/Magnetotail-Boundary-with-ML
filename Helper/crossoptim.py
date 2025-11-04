


import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Optional
from scipy.optimize import linear_sum_assignment

@dataclass
class MatchingResult:
    # Pairs of original indices (true_idx, pred_idx)
    matches: List[Tuple[int, int]]

    # For each match (same order as `matches`)
    tp_time_diffs_minutes: List[float]          # absolute minutes
    tp_time_diffs_minutes_signed: List[float]   # signed (pred - true) in minutes

    # Bookkeeping
    true_matched: Set[int]
    pred_matched: Set[int]
    unmatched_true: Set[int]
    unmatched_pred: Set[int]

    # Diagnostics (based on ±window reachability)
    in_range_true: List[int]   # true idx that have >=1 prediction within window
    out_range_pred: List[int]  # pred idx that have 0 true within window

    # Newly added field to track removed true crossings
    removed_true_crossings: Set[int] = None

def _prep_arrays(
    true_crossings: Iterable[np.datetime64],
    predicted_crossings: Iterable[np.datetime64],
    unmatched_true: Optional[Iterable[int]],
    unmatched_pred: Optional[Iterable[int]],
):
    ta = np.asarray(true_crossings)
    pa = np.asarray(predicted_crossings)
    if unmatched_true is None:
        unmatched_true = set(range(len(ta)))
    else:
        unmatched_true = set(unmatched_true)
    if unmatched_pred is None:
        unmatched_pred = set(range(len(pa)))
    else:
        unmatched_pred = set(unmatched_pred)

    t_idx = np.array(sorted(unmatched_true), dtype=int)
    p_idx = np.array(sorted(unmatched_pred), dtype=int)
    t_times = ta[t_idx]
    p_times = pa[p_idx]
    return ta, pa, unmatched_true, unmatched_pred, t_idx, p_idx, t_times, p_times

def _pairwise_minutes(t_times: np.ndarray, p_times: np.ndarray) -> np.ndarray:
    """
    Signed pairwise (pred - true) differences in minutes (float).
    shape = (n_true, n_pred)
    """
    # Broadcast subtract yields timedelta64; division gives float minutes.
    return (p_times[None, :] - t_times[:, None]) / np.timedelta64(1, 'm')

def _in_out_range(mask_in_window: np.ndarray, t_idx: np.ndarray, p_idx: np.ndarray):
    in_range_true = sorted([int(t_idx[i]) for i in np.where(mask_in_window.any(axis=1))[0]])
    out_range_pred = sorted([int(p_idx[j]) for j in np.where(~mask_in_window.any(axis=0))[0]])
    return in_range_true, out_range_pred

def match_crossings_greedy(
    true_crossings: Iterable[np.datetime64],
    predicted_crossings: Iterable[np.datetime64],
    time_window: np.timedelta64,                          # e.g. np.timedelta64(30, 'm')
    unmatched_true: Optional[Iterable[int]] = None,
    unmatched_pred: Optional[Iterable[int]] = None,
    removal_window_minutes: float = 10.0,                 # window to remove near-TP true crossings from FNs
) -> MatchingResult:
    """
    Closest-first greedy one-to-one matching within a time window.
    
    True crossings within removal_window_minutes of any matched true crossing are moved to 
    removed_true_crossings instead of being counted as false negatives.
    """
    # Prep
    (ta, pa, u_true, u_pred, t_idx, p_idx, t_times, p_times) = _prep_arrays(
        true_crossings, predicted_crossings, unmatched_true, unmatched_pred
    )
    if len(t_idx) == 0 or len(p_idx) == 0:
        return MatchingResult([], [], [], set(), set(), set(u_true), set(u_pred), [], [], set())

    # Pairwise differences (minutes)
    d_signed = _pairwise_minutes(t_times, p_times)
    d_abs = np.abs(d_signed)

    # Window mask
    win_minutes = time_window / np.timedelta64(1, 'm')
    mask = d_abs <= win_minutes

    # Diagnostics
    in_range_true, out_range_pred = _in_out_range(mask, t_idx, p_idx)

    # Build all valid pairs (i_true_global, j_pred_global, abs, signed)
    valid_pairs = []
    ti, tj = np.where(mask)
    for i, j in zip(ti.tolist(), tj.tolist()):
        valid_pairs.append((
            int(t_idx[i]), int(p_idx[j]),
            float(d_abs[i, j]), float(d_signed[i, j])
        ))
    # Sort by absolute time difference (ascending)
    valid_pairs.sort(key=lambda x: x[2])

    # Greedy selection
    true_matched: Set[int] = set()
    pred_matched: Set[int] = set()
    matches: List[Tuple[int, int]] = []
    tp_abs: List[float] = []
    tp_signed: List[float] = []

    u_true_work = set(u_true)
    u_pred_work = set(u_pred)

    for ti_global, pj_global, abs_min, signed_min in valid_pairs:
        if ti_global in u_true_work and pj_global in u_pred_work:
            matches.append((ti_global, pj_global))
            tp_abs.append(abs_min)
            tp_signed.append(signed_min)
            true_matched.add(ti_global)
            pred_matched.add(pj_global)
            u_true_work.remove(ti_global)
            u_pred_work.remove(pj_global)

    # Find true crossings to remove from false negatives (near true positives)
    removed_true_crossings: Set[int] = set()
    unmatched_true_initial = set(u_true) - true_matched
    
    if unmatched_true_initial and true_matched:
        # Get times of matched true crossings
        matched_true_times = ta[list(true_matched)]
        
        for unmatched_idx in list(unmatched_true_initial):
            unmatched_time = ta[unmatched_idx]
            # Check if within removal window of any matched true crossing
            time_diffs_minutes = np.abs((matched_true_times - unmatched_time) / np.timedelta64(1, 'm'))
            if np.any(time_diffs_minutes <= removal_window_minutes):
                removed_true_crossings.add(unmatched_idx)

    unmatched_true_final = unmatched_true_initial - removed_true_crossings
    unmatched_pred_final = set(u_pred) - pred_matched

    return MatchingResult(
        matches=matches,
        tp_time_diffs_minutes=tp_abs,
        tp_time_diffs_minutes_signed=tp_signed,
        true_matched=true_matched,
        pred_matched=pred_matched,
        unmatched_true=unmatched_true_final,
        unmatched_pred=unmatched_pred_final,
        in_range_true=in_range_true,
        out_range_pred=out_range_pred,
        removed_true_crossings=removed_true_crossings,
    )

def match_crossings_hungarian(
    true_crossings: Iterable[np.datetime64],
    predicted_crossings: Iterable[np.datetime64],
    time_window: np.timedelta64,                          # e.g. np.timedelta64(30, 'm')
    unmatched_true: Optional[Iterable[int]] = None,
    unmatched_pred: Optional[Iterable[int]] = None,
    fn_penalty_minutes: Optional[float] = None,           # penalty to leave a true crossing unmatched (FN)
    fp_penalty_minutes: Optional[float] = None,           # penalty to leave a predicted crossing unmatched (FP)
    big_M: float = 1e9,                                   # very large cost for forbidden/out-of-window pairs
    removal_window_minutes: float = 10.0,                 # window to remove near-TP true crossings from FNs
) -> MatchingResult:
    """
    Global-optimal one-to-one matching with explicit penalties for unmatched items.
    Uses a square augmented cost matrix with dummy rows/cols.

    - Real pair cost = |pred - true| (minutes) if within the time window, else big_M.
    - Each true has its own dummy *column* (cost = fn_penalty_minutes) to allow 'unmatched true' (FN).
    - Each pred has its own dummy *row* (cost = fp_penalty_minutes) to allow 'unmatched pred' (FP).
    - Bottom-right dummy-vs-dummy block is set to 0 so extra dummies can pair without affecting cost.

    True crossings within removal_window_minutes of any matched true crossing are moved to 
    removed_true_crossings instead of being counted as false negatives.

    Choose fn_penalty_minutes / fp_penalty_minutes to bias recall vs precision:
      * Larger FN penalty -> match more trues (higher recall).
      * Larger FP penalty -> avoid leaving predictions unmatched (useful if over-detection is rare).
    Sensible default: time_window in minutes.
    """
    if linear_sum_assignment is None:
        raise ImportError("scipy is required: pip install scipy")

    # Prep
    (ta, pa, u_true, u_pred, t_idx, p_idx, t_times, p_times) = _prep_arrays(
        true_crossings, predicted_crossings, unmatched_true, unmatched_pred
    )
    if len(t_idx) == 0 or len(p_idx) == 0:
        # No real pairs possible → nothing to match
        return MatchingResult([], [], [], set(), set(), set(u_true), set(u_pred), [], [])

    # Pairwise differences (minutes)
    d_signed = _pairwise_minutes(t_times, p_times)
    d_abs = np.abs(d_signed)

    # Window mask
    win_minutes = float(time_window / np.timedelta64(1, 'm'))
    mask = d_abs <= win_minutes

    # Diagnostics
    in_range_true, out_range_pred = _in_out_range(mask, t_idx, p_idx)

    # Default penalties if not given
    if fn_penalty_minutes is None:
        fn_penalty_minutes = win_minutes
    if fp_penalty_minutes is None:
        fp_penalty_minutes = win_minutes

    nT, nP = d_abs.shape
    N = nT + nP  # size of augmented square matrix

    # Build augmented cost matrix
    C = np.full((N, N), big_M, dtype=float)

    # Top-left: real pair costs (within window) else big_M
    real_costs = np.where(mask, d_abs, big_M)
    C[:nT, :nP] = real_costs

    # Top-right: FN dummies (each true i has its own dummy column nP + i)
    for i in range(nT):
        C[i, nP + i] = fn_penalty_minutes  # diagonal: allow true i to go unmatched at this penalty
    # Off-diagonals in this block remain big_M, preventing a true i from using dummy of another true.

    # Bottom-left: FP dummies (each pred j has its own dummy row nT + j)
    for j in range(nP):
        C[nT + j, j] = fp_penalty_minutes  # diagonal: allow pred j to go unmatched at this penalty
    # Off-diagonals in this block remain big_M.

    # Bottom-right: dummy-vs-dummy can pair at 0 cost to complete the square assignment
    C[nT:, nP:] = 0.0

    # Solve
    row_ind, col_ind = linear_sum_assignment(C)

    # Decode assignments
    true_matched: Set[int] = set()
    pred_matched: Set[int] = set()
    matches: List[Tuple[int, int]] = []
    tp_abs: List[float] = []
    tp_signed: List[float] = []

    # Map back to original indices
    for r, c in zip(row_ind, col_ind):
        if r < nT and c < nP:
            # Real ↔ real assignment
            if mask[r, c]:
                ti_global = int(t_idx[r])
                pj_global = int(p_idx[c])
                matches.append((ti_global, pj_global))
                true_matched.add(ti_global)
                pred_matched.add(pj_global)
                tp_abs.append(float(d_abs[r, c]))
                tp_signed.append(float(d_signed[r, c]))
            # If not in window, the solver *might* still assign (shouldn't, due to big_M),
            # but we defensively ignore such pairs.

        # r < nT and c >= nP  : true r matched to its dummy → FN (unmatched true)
        # r >= nT and c < nP  : pred c matched to its dummy row → FP (unmatched pred)
        # r >= nT and c >= nP : dummy-dummy, ignore

    # Find true crossings to remove from false negatives (near true positives)
    removed_true_crossings: Set[int] = set()
    unmatched_true_initial = set(u_true) - true_matched
    
    if unmatched_true_initial and true_matched:
        # Get times of matched true crossings
        matched_true_times = ta[list(true_matched)]
        
        for unmatched_idx in list(unmatched_true_initial):
            unmatched_time = ta[unmatched_idx]
            # Check if within removal window of any matched true crossing
            time_diffs_minutes = np.abs((matched_true_times - unmatched_time) / np.timedelta64(1, 'm'))
            if np.any(time_diffs_minutes <= removal_window_minutes):
                removed_true_crossings.add(unmatched_idx)

    unmatched_true_final = unmatched_true_initial - removed_true_crossings
    unmatched_pred_final = set(u_pred) - pred_matched

    # Create result with added removed_true_crossings field
    result = MatchingResult(
        matches=matches,
        tp_time_diffs_minutes=tp_abs,
        tp_time_diffs_minutes_signed=tp_signed,
        true_matched=true_matched,
        pred_matched=pred_matched,
        unmatched_true=unmatched_true_final,
        unmatched_pred=unmatched_pred_final,
        in_range_true=in_range_true,
        out_range_pred=out_range_pred,
        removed_true_crossings=removed_true_crossings
    )
        
    return result
