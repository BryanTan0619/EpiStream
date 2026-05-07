#!/usr/bin/env python3
"""
evaluate_v3.py — Improved metrics keeping the original SemC / Sep / MBU
philosophy but adding two orthogonal length-bias corrections.

Changes vs. evaluate.py (v2 run-based):
  1. evidence_separability: each evidence dimension is normalized by its
     session-level 90th-percentile per-bin rate, removing the
     damage_dealt vs. combat_count scale mismatch (~100× difference)
     that previously dominated Sep scores.
  2. inter_window_separability_v3: uses the normalized evidence sep;
     clips combined Sep to [0, 1].
  3. boundary_utility_v3 / compute_mbu_v3: propagate normalized Sep
     into MBU computation so that MBU also benefits from the fix.
  4. wMBU / wMBU+: changed from duration-weighted to equal-weight
     per-boundary average.  Duration-weighting caused boundaries between
     long windows (DisPider) to dominate the aggregate, making the
     metric sensitive to segmentation granularity rather than quality.
     Equal-weight gives a clean interpretation:
       "what fraction / avg utility of a randomly chosen boundary?"
  5. [NEW 2026-04-27] window_consistency_length_aware(): SemC short-window
     bias correction.  Root cause: R=1 windows (single-run, no observed
     state transition) are algebraically forced to SemC=1.0 regardless of
     duration.  Short windows have much higher P(R=1) than long windows
     (78% at 5s vs 16% at 120s), biasing the aggregate by Δ≈+0.44.
     Fix: for R=1 windows only, multiply SemC by conf(T)=1−exp(−T/τ)
     with τ=9 (calibrated from average per-bin transition rate 0.11).
     Effect:  bias Δ(5s−120s) drops from +0.437 → −0.015.
             GT windows (N=2118): mean score change −0.019, median 0.0
             (coherent long GT windows barely affected: T≥30 → conf≈1).
     This correction is applied in compute_window_metrics_v3() (via the
     ``tau`` argument, default 9.0) and in boundary_utility_v3() /
     compute_mbu_v3() for the consistency terms.
  6. TagCl retired: window_consistency() now absorbs MarginNorm (the only
     independent TagCl signal), making TagCl a redundant metric.

Column mapping for print_side_by_side (unchanged header names):
  wSemC  ← wavg_intra["consistency"]    (length-aware SemC, dur-weighted)
  SemC   ← avg_intra["consistency"]     (length-aware SemC, equal-weight)
  Sep    ← avg_inter["combined_separability"]   (normed evidence)
  wSep   ← wavg_inter["combined_separability"]  (normed evidence, dur-weighted)
  mMBU   ← mbu["median_mbu"]            (equal-weight median, normed evidence)
  MBU+   ← mbu["positive_ratio"]        (equal-weight fraction, normed evidence)
  wMBU   ← mbu["wavg_mbu"]             (equal-weight avg, normed evidence)
  wMBU+  ← mbu["wpositive_ratio"]      (equal-weight fraction, normed evidence)
"""

import os
import sys
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from map import parse_tlog, map_tag_to_states, update_evidence, init_evidence_vector
from metrics import (
    ALL_STATES,
    per_bin_state_distribution,
    window_state_distribution,
    run_state_distribution,
    state_entropy,
    state_purity,
    state_transition_rate,
    longest_state_run_ratio,
    window_consistency,
    window_consistency_length_aware,
    aggregate_evidence,
    state_divergence,          # bin-based TV — kept for state channel (fine)
    compute_tag_clarity_stats,
    margin_norm, conf_ratio, ent_clarity, local_margin_norm,
    support_rate, tag_clarity,
    short_window_penalty, long_window_penalty, overlap_penalty,
)
from segment_gt import segment_match_log, build_time_bins, format_time

# Re-export helpers used by batch runner (so it only needs to import this module)
from evaluate import (
    build_bins_for_llm_windows,
    build_fixed_window_bins,
    print_side_by_side,
)


# ============================================================
# Improvement 1: evidence_separability with session-level normalization
# ============================================================

def compute_evidence_scale(all_bins: List[Dict]) -> Dict[str, float]:
    """
    Compute per-dimension scale for evidence_separability normalization.

    For each evidence dimension, collects per-bin raw values across all_bins,
    then takes the 90th-percentile as the scale.  Clipped to a minimum of
    1e-3 so dimensions with near-zero activity are treated as inactive but
    don't cause division-by-zero.

    This makes every dimension contribute on a comparable [0, ~1] scale
    regardless of the absolute magnitude of that event type in this session.

    Args:
        all_bins: flat list of time bins for the whole session.

    Returns:
        dict {dimension_name: scale_value}
    """
    ev_lists: Dict[str, List[float]] = defaultdict(list)
    for b in all_bins:
        for k, v in b["evidence"].items():
            ev_lists[k].append(float(v))

    scale = {}
    for k, vals in ev_lists.items():
        if not vals:
            scale[k] = 1e-3
            continue
        vals_sorted = sorted(vals)
        p90_idx = min(int(0.90 * len(vals_sorted)), len(vals_sorted) - 1)
        p90 = vals_sorted[p90_idx]
        scale[k] = max(p90, 1e-3)
    return scale


def evidence_sep_normed(W1_bins: List[Dict],
                        W2_bins: List[Dict],
                        ev_scale: Optional[Dict[str, float]] = None) -> float:
    """
    Normalized evidence separability.

    Computes per-bin rate for each dimension, divides by the session-level
    90th-percentile scale, then takes the mean absolute difference across
    active dimensions.  Result is in [0, ∞) but typically in [0, 1] when
    the scale is calibrated.  We clip to [0, 1] for stability.

    Without ev_scale (None), falls back to the original L1 on per-bin rates
    (backward compatible).
    """
    if not W1_bins or not W2_bins:
        return 0.0

    e1 = aggregate_evidence(W1_bins)
    e2 = aggregate_evidence(W2_bins)
    len1 = max(len(W1_bins), 1)
    len2 = max(len(W2_bins), 1)

    dims = list(e1.keys())
    dist = 0.0
    n_active = 0

    for k in dims:
        r1 = e1[k] / len1
        r2 = e2[k] / len2
        if ev_scale is not None:
            s = ev_scale.get(k, 1e-3)
            if s < 1e-9:
                continue
            dist += abs(r1 - r2) / s
        else:
            dist += abs(r1 - r2)
        n_active += 1

    if n_active == 0:
        return 0.0
    return min(dist / n_active, 1.0)


def inter_window_sep_v3(W1: List[Dict],
                        W2: List[Dict],
                        ev_scale: Optional[Dict[str, float]] = None,
                        l1: float = 0.5,
                        l2: float = 0.5) -> float:
    """
    Inter-window separability (v3).

    Identical to evaluate.py's inter_window_separability() except the
    evidence channel uses normalized evidence_sep_normed().
    """
    d_state = state_divergence(W1, W2)              # bin-based TV, kept
    d_evidence = evidence_sep_normed(W1, W2, ev_scale)
    combined = l1 * d_state + l2 * d_evidence
    return min(combined, 1.0)


# ============================================================
# Improvement 2: boundary utility using normalized Sep
# ============================================================

def boundary_utility_v3(
    W_left: List[Dict],
    W_right: List[Dict],
    ev_scale: Optional[Dict[str, float]] = None,
    bin_size: float = 1.0,
    *,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    delta: float = 0.1,
    w1: float = 1/3,
    w2: float = 1/6,
    w3: float = 1/6,
    w4: float = 1/3,
    tau: float = 9.0,
    l1: float = 0.5,
    l2: float = 0.5,
    min_window_secs: float = 5.0,
    max_window_secs: float = 180.0,
) -> float:
    """
    Marginal Boundary Utility (v3).

    Uses two improvements over boundary_utility_single() in metrics.py:
      1. Inter-window Sep uses normalized evidence (inter_window_sep_v3).
      2. Intra-window consistency uses length-aware SemC
         (window_consistency_length_aware) to remove short-window R=1 bias.

    MBU(b_i) = U(W_i, W_{i+1}) - U(W_i ⊕ W_{i+1})

    Positive MBU → boundary is justified.

    Args:
        tau: confidence half-life for length-aware SemC (seconds/bins).
             Default 9.0 calibrated from empirical transition rate ≈ 0.11/bin.
    """
    if not W_left or not W_right:
        return 0.0

    left_c = window_consistency_length_aware(W_left, tau, w1, w2, w3, w4)
    right_c = window_consistency_length_aware(W_right, tau, w1, w2, w3, w4)
    sep = inter_window_sep_v3(W_left, W_right, ev_scale, l1, l2)

    penalty_cut = 0.0
    penalty_cut += short_window_penalty(W_left, bin_size, min_window_secs)
    penalty_cut += short_window_penalty(W_right, bin_size, min_window_secs)
    penalty_cut += long_window_penalty(W_left, bin_size, max_window_secs)
    penalty_cut += long_window_penalty(W_right, bin_size, max_window_secs)
    penalty_cut += 0.3 * overlap_penalty(W_left)
    penalty_cut += 0.3 * overlap_penalty(W_right)

    U_cut = alpha * left_c + beta * right_c + gamma * sep - delta * penalty_cut

    W_merged = W_left + W_right
    merged_c = window_consistency_length_aware(W_merged, tau, w1, w2, w3, w4)

    penalty_merged = 0.0
    penalty_merged += short_window_penalty(W_merged, bin_size, min_window_secs)
    penalty_merged += long_window_penalty(W_merged, bin_size, max_window_secs)
    penalty_merged += 0.3 * overlap_penalty(W_merged)

    U_merged = merged_c - delta * penalty_merged

    return U_cut - U_merged


def compute_mbu_v3(
    window_bins_list: List[List[Dict]],
    ev_scale: Optional[Dict[str, float]] = None,
    bin_size: float = 1.0,
    dominant_states: Optional[List[str]] = None,
    **kwargs,
) -> Dict:
    """
    Compute MBU for all boundaries using normalized evidence (v3).

    Key difference from compute_mbu() in metrics.py:
      - Uses boundary_utility_v3 (normalized evidence Sep).
      - wMBU / wMBU+ use EQUAL-WEIGHT boundary averaging instead of
        duration-weighted averaging.  This removes the length bias where
        boundaries between long windows (DisPider) contribute
        disproportionately to the aggregate score.
      - Approach B: when dominant_states is provided, only boundaries between
        windows with *different* dominant states are included in MBU computation.
        Same-state boundaries are excluded, consistent with the inter-window
        separability cross-state filter.
    """
    n = len(window_bins_list)
    empty = {
        "per_boundary": [],
        "avg_mbu": 0.0,
        "positive_ratio": 0.0,
        "median_mbu": 0.0,
        "min_mbu": 0.0,
        "max_mbu": 0.0,
        "wavg_mbu": 0.0,
        "wpositive_ratio": 0.0,
    }
    if n <= 1:
        return empty

    per_boundary = []
    for i in range(n - 1):
        # Approach B: skip same-state boundaries when states are available
        if dominant_states is not None:
            if dominant_states[i] == dominant_states[i + 1]:
                continue

        W_left = window_bins_list[i]
        W_right = window_bins_list[i + 1]

        mbu = boundary_utility_v3(W_left, W_right, ev_scale, bin_size, **kwargs)

        left_dur = (W_left[-1]["end"] - W_left[0]["start"]) if W_left else 0.0
        right_dur = (W_right[-1]["end"] - W_right[0]["start"]) if W_right else 0.0

        per_boundary.append({
            "boundary_idx": i,
            "boundary_time": W_left[-1]["end"] if W_left else 0.0,
            "mbu": mbu,
            "left_duration": round(left_dur, 2),
            "right_duration": round(right_dur, 2),
        })

    if not per_boundary:
        return empty

    mbu_values = [pb["mbu"] for pb in per_boundary]
    n_bnd = len(mbu_values)

    sorted_mbu = sorted(mbu_values)
    mid = n_bnd // 2
    median_mbu = (sorted_mbu[mid] if n_bnd % 2 == 1
                  else (sorted_mbu[mid - 1] + sorted_mbu[mid]) / 2.0)

    avg_mbu = sum(mbu_values) / n_bnd
    positive_ratio = sum(1 for v in mbu_values if v > 0) / n_bnd

    # wMBU / wMBU+: equal-weight per boundary (no duration weighting).
    # Interpretation: "what is the average marginal utility of a randomly
    # chosen boundary?" — directly comparable across methods with different
    # numbers of boundaries.
    wavg_mbu = avg_mbu                  # equal-weight = plain average
    wpositive_ratio = positive_ratio    # equal-weight = plain fraction

    return {
        "per_boundary": per_boundary,
        "avg_mbu": avg_mbu,
        "positive_ratio": positive_ratio,
        "median_mbu": median_mbu,
        "min_mbu": min(mbu_values),
        "max_mbu": max(mbu_values),
        "wavg_mbu": wavg_mbu,       # → printed as wMBU↑
        "wpositive_ratio": wpositive_ratio,  # → printed as wMBU+↑
    }


# ============================================================
# Main entry point: compute_window_metrics_v3
# ============================================================

def compute_window_metrics_v3(
    window_bins_list: List[List[Dict]],
    all_bins: Optional[List[Dict]] = None,
    tau: float = 9.0,
) -> Dict[str, Any]:
    """
    Drop-in replacement for compute_window_metrics() with length-bias-free
    normalization.

    Args:
        window_bins_list: list of bin-lists, one per window (required).
        all_bins: flat list of every time bin for the full session, used to
            compute the evidence normalization scale.  If None, the bins
            from window_bins_list are concatenated (slightly less accurate
            when windows don't cover the full session).
        tau: confidence half-life (seconds/bins) for the length-aware SemC
            correction applied to R=1 windows.  Default 9.0 (calibrated from
            empirical per-bin transition rate ≈ 0.11).  Set tau=None to
            disable the correction and use raw window_consistency().

    Returns:
        Same dict structure as compute_window_metrics() so that
        print_side_by_side() and aggregate_results() work unchanged.
        The ``consistency`` field in per_window / avg_intra / wavg_intra now
        contains length-aware SemC when tau is not None.
    """
    if not window_bins_list:
        return {
            "num_windows": 0,
            "total_duration": 0.0,
            "per_window": [],
            "inter_window": [],
            "avg_intra": {k: 0.0 for k in [
                "entropy", "purity", "transition_rate", "lsr", "consistency",
                "margin_norm", "conf_ratio", "ent_clarity", "local_margin",
                "support_rate", "tag_clarity"]},
            "wavg_intra": {k: 0.0 for k in [
                "entropy", "purity", "transition_rate", "lsr", "consistency",
                "margin_norm", "conf_ratio", "ent_clarity", "local_margin",
                "support_rate", "tag_clarity"]},
            "avg_inter": {"state_divergence": 0.0,
                          "evidence_separability": 0.0,
                          "combined_separability": 0.0},
            "wavg_inter": {"state_divergence": 0.0,
                           "evidence_separability": 0.0,
                           "combined_separability": 0.0},
            "mbu": {
                "per_boundary": [], "avg_mbu": 0.0, "positive_ratio": 0.0,
                "median_mbu": 0.0, "min_mbu": 0.0, "max_mbu": 0.0,
                "wavg_mbu": 0.0, "wpositive_ratio": 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Compute evidence normalization scale for this session
    # ------------------------------------------------------------------
    if all_bins is None:
        ref_bins = [b for w in window_bins_list for b in w]
    else:
        ref_bins = all_bins
    ev_scale = compute_evidence_scale(ref_bins)

    # Pick the consistency function (length-aware by default, raw if tau=None)
    _consistency = (
        (lambda wb: window_consistency_length_aware(wb, tau))
        if tau is not None
        else window_consistency
    )

    n = len(window_bins_list)

    # ------------------------------------------------------------------
    # Intra-window metrics (identical to evaluate.py — already run-based)
    # ------------------------------------------------------------------
    intra_keys = [
        "entropy", "purity", "transition_rate", "lsr", "consistency",
        "margin_norm", "conf_ratio", "ent_clarity", "local_margin",
        "support_rate", "tag_clarity",
    ]

    per_window = []
    for i, wb in enumerate(window_bins_list):
        duration = wb[-1]["end"] - wb[0]["start"] if wb else 0.0
        m = {
            "window_id": i,
            "start": wb[0]["start"] if wb else 0.0,
            "end": wb[-1]["end"] if wb else 0.0,
            "duration": round(duration, 2),
            "num_bins": len(wb),
            "entropy": state_entropy(wb),
            "purity": state_purity(wb),
            "transition_rate": state_transition_rate(wb),
            "lsr": longest_state_run_ratio(wb),
            "consistency": _consistency(wb),
            "margin_norm": margin_norm(wb),
            "conf_ratio": conf_ratio(wb),
            "ent_clarity": ent_clarity(wb),
            "local_margin": local_margin_norm(wb),
            "support_rate": support_rate(wb),
            "tag_clarity": tag_clarity(wb),
        }
        pw = window_state_distribution(wb)
        m["dominant_state"] = max(pw, key=pw.get)
        per_window.append(m)

    # Equal-weight intra averages
    avg_intra = {}
    for key in intra_keys:
        vals = [m[key] for m in per_window]
        avg_intra[key] = sum(vals) / len(vals) if vals else 0.0

    # Duration-weighted intra averages
    durations = [m["duration"] for m in per_window]
    total_duration = sum(durations)
    wavg_intra = {}
    for key in intra_keys:
        if total_duration > 0:
            wavg_intra[key] = sum(
                m[key] * m["duration"] for m in per_window
            ) / total_duration
        else:
            wavg_intra[key] = 0.0

    # ------------------------------------------------------------------
    # Inter-window metrics (v3: normalized evidence Sep)
    # Approach B: only evaluate cross-state adjacent pairs (dominant_state differs).
    # Same-state adjacent pairs are excluded so that intra-window purity and
    # inter-window separability can be optimized simultaneously without conflict.
    # ------------------------------------------------------------------
    inter_metrics = []
    inter_pair_weights = []   # for duration-weighted avg (parallel to inter_metrics)
    for i in range(n - 1):
        # Cross-state filter (Approach B)
        dom1 = per_window[i]["dominant_state"]
        dom2 = per_window[i + 1]["dominant_state"]
        if dom1 == dom2:
            continue  # skip same-state pairs

        w1 = window_bins_list[i]
        w2 = window_bins_list[i + 1]
        d_state = state_divergence(w1, w2)
        d_evidence = evidence_sep_normed(w1, w2, ev_scale)
        d_combined = inter_window_sep_v3(w1, w2, ev_scale)
        inter_metrics.append({
            "pair": f"{i}-{i+1}",
            "state_divergence": d_state,
            "evidence_separability": d_evidence,
            "combined_separability": d_combined,
        })
        dur1 = per_window[i]["duration"]
        dur2 = per_window[i + 1]["duration"]
        inter_pair_weights.append((dur1 + dur2) / 2.0)

    # Equal-weight inter averages
    avg_inter = {}
    for key in ["state_divergence", "evidence_separability", "combined_separability"]:
        vals = [m[key] for m in inter_metrics]
        avg_inter[key] = sum(vals) / len(vals) if vals else 0.0

    # Duration-weighted inter averages (pair weight = mean of adjacent durations)
    total_pair_weight = sum(inter_pair_weights)
    wavg_inter = {}
    for key in ["state_divergence", "evidence_separability", "combined_separability"]:
        if total_pair_weight > 0 and inter_metrics:
            wavg_inter[key] = sum(
                inter_metrics[k][key] * inter_pair_weights[k]
                for k in range(len(inter_metrics))
            ) / total_pair_weight
        else:
            wavg_inter[key] = 0.0

    # ------------------------------------------------------------------
    # MBU (v3: normalized evidence, length-aware SemC, equal-weight wMBU)
    # Approach B: pass dominant_states so same-state boundaries are excluded.
    # ------------------------------------------------------------------
    dominant_states = [pw["dominant_state"] for pw in per_window]
    mbu_kwargs = {} if tau is None else {"tau": tau}
    mbu_result = compute_mbu_v3(window_bins_list, ev_scale, bin_size=1.0,
                                dominant_states=dominant_states,
                                **mbu_kwargs)

    return {
        "num_windows": n,
        "total_duration": round(total_duration, 2),
        "per_window": per_window,
        "inter_window": inter_metrics,
        "avg_intra": avg_intra,
        "wavg_intra": wavg_intra,
        "avg_inter": avg_inter,
        "wavg_inter": wavg_inter,
        "mbu": mbu_result,
    }
