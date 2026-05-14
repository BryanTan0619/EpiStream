
import os
import sys
import json
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from map import parse_tlog
from segment_gt import build_time_bins
from metrics import (
    window_consistency,
    inter_window_separability,
    short_window_penalty,
    long_window_penalty,
    overlap_penalty,
)

PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TLOG_BASE_DIR = os.environ.get("TLOG_BASE_DIR", "/path/to/tlog_files")
MEMORY_CACHE_PATH = os.environ.get("MEMORY_CACHE_PATH") or os.path.join(
    PROJECT_DIR, "cache", "memory", "GT-Original.json"
)

BIN_SIZE = 1.0
MIN_WINDOW_SECS = 5.0
MAX_WINDOW_SECS = 180.0

ALPHA = 0.3
BETA = 0.3
GAMMA = 0.3
DELTA = 0.1

W1, W2, W3, W4 = 0.25, 0.25, 0.25, 0.25

L1, L2 = 0.7, 0.3

LOOKAHEAD_DELTA = 1.0
LOOKAHEAD_BEYOND_BOUNDARY = 6.0

def normalize_gt_original_window(window: Dict) -> Dict:
    return {
        "start_time": float(window["start"]),
        "end_time": float(window["end"]),
        "duration_secs": float(window.get("duration", window["end"] - window["start"])),
        "dominant_state": window.get("original_label") or window.get("label") or window.get("state") or "Unknown",
    }

def load_gt_original_windows_cache(cache_path: str = MEMORY_CACHE_PATH) -> Dict[str, List[Dict]]:
    if not os.path.isfile(cache_path):
        return {}

    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    matches = raw.get("matches", {}) if isinstance(raw, dict) else {}
    gt_map: Dict[str, List[Dict]] = {}
    for match_id, match_data in matches.items():
        if not isinstance(match_data, dict):
            continue
        windows = match_data.get("windows", [])
        norm_windows = [
            normalize_gt_original_window(w)
            for w in windows
            if isinstance(w, dict) and "start" in w and "end" in w
        ]
        if norm_windows:
            gt_map[match_id] = norm_windows
    return gt_map

def compute_u_close_at_time(
    all_bins: List[Dict],
    match_start_time: float,
    cut_time: float,
    window_start_time: float,
    right_window_end_time: float,
    bin_size: float = BIN_SIZE,
) -> float:
    if not all_bins:
        return 0.0
    
    ws_idx = max(0, int((window_start_time - match_start_time) / bin_size))
    t_idx = min(len(all_bins), int((cut_time - match_start_time) / bin_size))
    right_end_idx = min(len(all_bins), int((right_window_end_time - match_start_time) / bin_size))
    
    W_left = all_bins[ws_idx:t_idx]
    W_right = all_bins[t_idx:right_end_idx]
    
    if not W_left or len(W_left) < 1:
        return 0.0
    
    if not W_right or len(W_right) < 1:
        return 0.0
    
    left_c = window_consistency(W_left, W1, W2, W3, W4)
    right_c = window_consistency(W_right, W1, W2, W3, W4)
    sep = inter_window_separability(W_left, W_right, L1, L2)
    
    penalty_cut = (
        short_window_penalty(W_left, bin_size, MIN_WINDOW_SECS)
        + short_window_penalty(W_right, bin_size, MIN_WINDOW_SECS)
        + long_window_penalty(W_left, bin_size, MAX_WINDOW_SECS)
        + long_window_penalty(W_right, bin_size, MAX_WINDOW_SECS)
        + 0.3 * overlap_penalty(W_left)
        + 0.3 * overlap_penalty(W_right)
    )
    
    u_close = ALPHA * left_c + BETA * right_c + GAMMA * sep - DELTA * penalty_cut
    return float(u_close)

def _get_right_window_end(
    gt_windows: List[Dict],
    current_window_idx: int,
    all_bins: List[Dict],
) -> float:
    next_win_idx = current_window_idx + 1
    if next_win_idx < len(gt_windows):
        return gt_windows[next_win_idx]["end_time"]
    else:
        return all_bins[-1]["end"] if all_bins else 0.0

def compute_utility_forward_looking(
    all_bins: List[Dict],
    match_start_time: float,
    sample_time: float,
    window_start_time: float,
    gt_windows: List[Dict],
    current_window_idx: int,
    bin_size: float = BIN_SIZE,
) -> Tuple[float, float]:
    if not all_bins:
        return 0.0, 0.0
    
    current_win = gt_windows[current_window_idx] if current_window_idx < len(gt_windows) else None
    if not current_win:
        current_boundary_time = all_bins[-1]["end"]
    else:
        current_boundary_time = current_win["end_time"]
    
    match_end_time = all_bins[-1]["end"] if all_bins else current_boundary_time
    effective_boundary_time = min(
        current_boundary_time + LOOKAHEAD_BEYOND_BOUNDARY,
        match_end_time
    )
    
    right_window_end_unified = effective_boundary_time
    
    u_close = compute_u_close_at_time(
        all_bins, match_start_time, sample_time, window_start_time,
        right_window_end_unified, bin_size
    )
    
    best_future_u = None
    first_c = math.ceil((sample_time + 1e-9) / LOOKAHEAD_DELTA) * LOOKAHEAD_DELTA
    c = first_c
    while c <= effective_boundary_time:
        if c <= current_boundary_time:
            u_future = compute_u_close_at_time(
                all_bins, match_start_time, c, window_start_time,
                right_window_end_unified, bin_size
            )
        else:
            u_future = compute_u_close_at_time(
                all_bins, match_start_time, c, current_boundary_time,
                right_window_end_unified, bin_size
            )
        if best_future_u is None or u_future > best_future_u:
            best_future_u = u_future
        c += LOOKAHEAD_DELTA
    
    if best_future_u is not None:
        u_continue = best_future_u
    else:
        u_continue = u_close
    
    return float(u_close), float(u_continue)

def precompute_utilities_v2(jsonl_path: str, output_path: str, verbose: bool = True):
    if os.path.exists(output_path):
        if verbose:
            print(f"  Already exists: {output_path}")
        return output_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if verbose:
        print(f"  Loaded {len(samples)} samples from {jsonl_path}")

    gt_original_cache = load_gt_original_windows_cache()
    if verbose:
        print(f"  Loaded GT-Original windows for {len(gt_original_cache)} matches "
              f"from {MEMORY_CACHE_PATH}")
    
    match_groups = defaultdict(list)
    for i, s in enumerate(samples):
        match_groups[s["match_id"]].append((i, s))
    
    bins_cache = {}
    augmented = [None] * len(samples)
    
    match_items = list(match_groups.items())
    for match_id, group in tqdm(match_items, desc="  Matches", disable=not verbose):
        
        tlog_path = os.path.join(TLOG_BASE_DIR, f"{match_id}.txt")
        if not os.path.isfile(tlog_path):
            if verbose:
                print(f"    Warning: tlog not found, skipping")
            for idx, s in group:
                s["u_close"], s["u_continue"] = 0.0, 1.0
                augmented[idx] = s
            continue
        
        events = parse_tlog(tlog_path)
        match_start_time = events[0]["time"] if events else 0.0
        all_bins = build_time_bins(events, bin_size=BIN_SIZE)
        bins_cache[match_id] = (all_bins, match_start_time)
        
        gt_windows = gt_original_cache.get(match_id)
        if not gt_windows:
            if verbose:
                print(f"    Warning: GT-Original labels not found in memory cache, skipping")
            for idx, s in group:
                s["u_close"], s["u_continue"] = 0.0, 1.0
                augmented[idx] = s
            continue
        
        for idx, s in tqdm(group, desc=f"    {match_id[:20]}", leave=False, disable=not verbose):
            u_close, u_continue = compute_utility_forward_looking(
                all_bins,
                match_start_time,
                s["sample_time"],
                s["window_start_time"],
                gt_windows,
                s["current_window_idx"],
                BIN_SIZE,
            )
            s["u_close"] = u_close
            s["u_continue"] = u_continue
            augmented[idx] = s
    
    NORM_EPS = 1e-8
    window_groups = defaultdict(list)
    for i, s in enumerate(augmented):
        if s is not None:
            key = (s["match_id"], s["current_window_idx"])
            window_groups[key].append(i)

    for key, indices in window_groups.items():
        u_close_vals = [augmented[i]["u_close"] for i in indices]
        max_u_close = max(u_close_vals)
        min_u_close = min(u_close_vals)
        u_range = max_u_close - min_u_close

        if u_range > NORM_EPS:
            for i in indices:
                raw_uc = augmented[i]["u_close"]
                raw_ucont = augmented[i]["u_continue"]
                augmented[i]["u_close_raw"] = raw_uc
                augmented[i]["u_continue_raw"] = raw_ucont
                augmented[i]["u_close"] = (raw_uc - min_u_close) / u_range
                augmented[i]["u_continue"] = 1.0
                augmented[i]["u_norm_max"] = max_u_close
                augmented[i]["u_norm_min"] = min_u_close
                augmented[i]["u_norm_range"] = u_range
        else:
            for i in indices:
                augmented[i]["u_close_raw"] = augmented[i]["u_close"]
                augmented[i]["u_continue_raw"] = augmented[i]["u_continue"]
                augmented[i]["u_close"] = 0.0
                augmented[i]["u_continue"] = 1.0
                augmented[i]["u_norm_max"] = max_u_close
                augmented[i]["u_norm_min"] = min_u_close
                augmented[i]["u_norm_range"] = 0.0

    if verbose:
        n_windows = len(window_groups)
        ranges = [augmented[indices[0]]["u_norm_range"]
                  for indices in window_groups.values()
                  if augmented[indices[0]]["u_norm_range"] > NORM_EPS]
        if ranges:
            print(f"  Per-window min-max normalization: {n_windows} windows, "
                  f"u_range [{min(ranges):.4f}, {max(ranges):.4f}]")

    with open(output_path, 'w', encoding='utf-8') as f:
        for s in augmented:
            if s:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    if verbose:
        print(f"Saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute forward-looking utilities")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Training JSONL file")
    parser.add_argument("--val_jsonl", type=str, required=True, help="Validation JSONL file")
    parser.add_argument("--test_jsonl", type=str, default=None, help="Test JSONL file (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--delta", type=float, default=LOOKAHEAD_DELTA, help="Step size between candidate cut points (seconds)")
    
    args = parser.parse_args()
    
    LOOKAHEAD_DELTA = args.delta
    
    print("="*80)
    print("Forward-Looking Utility Computation (V2)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Strategy: max_{{c>t}} U(CLOSE at c)")
    print(f"  Candidate step delta: {LOOKAHEAD_DELTA}s")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_output = os.path.join(args.output_dir, "train_with_utility.jsonl")
    precompute_utilities_v2(args.train_jsonl, train_output, verbose=True)
    
    val_output = os.path.join(args.output_dir, "val_with_utility.jsonl")
    precompute_utilities_v2(args.val_jsonl, val_output, verbose=True)
    
    test_output = None
    if args.test_jsonl:
        test_output = os.path.join(args.output_dir, "test_with_utility.jsonl")
        precompute_utilities_v2(args.test_jsonl, test_output, verbose=True)
    
    print("\n Utility computation complete!")
    print(f"\nOutput files:")
    print(f"  {train_output}")
    print(f"  {val_output}")
    if test_output:
        print(f"  {test_output}")
