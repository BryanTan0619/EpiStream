
import os
import sys
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any

from map import (
    parse_tlog,
    map_tag_to_states,
    map_tag_to_states_with_phase,
    update_evidence,
    compute_dominant_state,
    init_evidence_vector,
    VisibilityTracker,
)
from metrics import (
    ALL_STATES,
    window_state_distribution,
    run_state_distribution,
    state_entropy,
    state_purity,
    state_transition_rate,
    longest_state_run_ratio,
    window_consistency,
    aggregate_evidence,
    inter_window_separability,
    evaluate_cut_gain,
    boundary_utility_single,
    state_entropy_run_based,
    margin_norm,
    conf_ratio,
    ent_clarity,
    support_rate,
    tag_clarity,
)

ALPHA = 0.38
BETA = 0.38
GAMMA = 0.28
DELTA = 0.10

W1 = 0.25
W2 = 0.25
W3 = 0.25
W4 = 0.25

L1 = 0.5
L2 = 0.5

B1 = 0.25
B2 = 0.25
B3 = 0.25
B4 = 0.25

TAG_CLARITY_PROTECT_THRESHOLD = 0.8

TAG_CLARITY_SPLIT_THRESHOLD = 0.5

CUT_THRESHOLD = 0.006
LOOKAHEAD_BINS = 20
MIN_WINDOW_SECS = 5.0
MAX_WINDOW_SECS = 60.0

EVIDENCE_SHIFT_THRESHOLD = 1.5

MBU_MERGE_THRESHOLD = 0.06
MBU_MAX_PASSES = 50

MIN_SEPARABILITY_FOR_BOUNDARY = 0.14

def build_time_bins(events: List[Dict], bin_size: float) -> List[Dict]:
    if not events:
        return []

    t_start = events[0]["time"]
    t_end = events[-1]["time"]
    num_bins = int((t_end - t_start) / bin_size) + 1

    bins = []
    for i in range(num_bins):
        bins.append({
            "index": i,
            "start": t_start + i * bin_size,
            "end": t_start + (i + 1) * bin_size,
            "tags": [],
            "events": [],
            "states": set(),
            "evidence": init_evidence_vector(),
            "dominant_state": "Unknown",
        })

    in_parachute_phase = True

    vis_tracker = VisibilityTracker()

    for e in events:
        tag = e["tag"]

        if tag == "落地":
            pass

        idx = int((e["time"] - t_start) / bin_size)
        idx = min(idx, num_bins - 1)
        bins[idx]["tags"].append(tag)
        bins[idx]["events"].append(e)

        mapped = map_tag_to_states_with_phase(e, in_parachute_phase)
        bins[idx]["states"].update(mapped)

        update_evidence(bins[idx]["evidence"], e, tracker=vis_tracker)

        if tag == "落地":
            in_parachute_phase = False

    for b in bins:
        b["dominant_state"] = compute_dominant_state(b)

    _propagate_empty_bins(bins)

    return bins

def _propagate_empty_bins(bins: List[Dict]) -> None:
    for i, b in enumerate(bins):
        if b["states"] and b["dominant_state"] != "Unknown":
            continue

        nearest = None
        for offset in range(1, 6):
            if i - offset >= 0 and bins[i - offset]["dominant_state"] != "Unknown":
                nearest = bins[i - offset]["dominant_state"]
                break
            if i + offset < len(bins) and bins[i + offset]["dominant_state"] != "Unknown":
                nearest = bins[i + offset]["dominant_state"]
                break

        if nearest:
            b["states"] = {nearest}
            b["dominant_state"] = nearest

STRONG_BOUNDARY_TAGS = {
    "敌人进入视野",
    "射击武器",
    "受到伤害",
    "被玩家攻击",
    "淘汰玩家",
    "击倒玩家",
    "上载具",
    "下载具",
    "落地",
}

def get_dominant_of_range(bins: List[Dict], start: int, end: int) -> str:
    if start >= end:
        return "Unknown"
    counts = defaultdict(int)
    for i in range(start, min(end, len(bins))):
        ds = bins[i]["dominant_state"]
        if ds != "Unknown":
            counts[ds] += 1
    if not counts:
        return "Unknown"
    return max(counts, key=counts.get)

def evidence_shift_large(bins: List[Dict], t: int, lookback: int = 3) -> bool:
    if t < lookback or t >= len(bins):
        return False

    ev_before = init_evidence_vector()
    ev_after = init_evidence_vector()

    for i in range(max(0, t - lookback), t):
        for k in ev_before:
            ev_before[k] += bins[i]["evidence"][k]

    for i in range(t, min(len(bins), t + lookback)):
        for k in ev_after:
            ev_after[k] += bins[i]["evidence"][k]

    dist = sum(abs(ev_before[k] - ev_after[k]) for k in ev_before)
    return dist > EVIDENCE_SHIFT_THRESHOLD

def _local_state_changed(bins: List[Dict], t: int, lookback: int = 3,
                         lookahead: int = 3) -> bool:
    if t < 1 or t >= len(bins):
        return False
    lb = max(0, t - lookback)
    la = min(len(bins), t + lookahead)
    prev_dom = get_dominant_of_range(bins, lb, t)
    next_dom = get_dominant_of_range(bins, t, la)
    return (prev_dom != next_dom
            and prev_dom != "Unknown"
            and next_dom != "Unknown")

def _activity_burst(bins: List[Dict], t: int, lookback: int = 5) -> bool:
    if t < lookback or t >= len(bins):
        return False
    prev_event_count = sum(len(bins[i]["events"]) for i in range(t - lookback, t))
    cur_event_count = len(bins[t]["events"])
    if prev_event_count <= 1 and cur_event_count >= 3:
        return True
    if prev_event_count >= lookback * 2 and cur_event_count == 0:
        return True
    return False

def is_candidate_boundary(bins: List[Dict], t: int) -> bool:
    if t <= 0 or t >= len(bins):
        return False

    prev_dom = get_dominant_of_range(bins, max(0, t - 2), t)
    next_dom = get_dominant_of_range(bins, t, min(len(bins), t + 2))
    if prev_dom != next_dom and prev_dom != "Unknown" and next_dom != "Unknown":
        return True

    tags_at_t = set(bins[t]["tags"])
    if tags_at_t.intersection(STRONG_BOUNDARY_TAGS):
        if t > 0 and bins[t]["dominant_state"] != bins[t - 1]["dominant_state"]:
            return True
        if any(tag in ("射击武器", "受到伤害", "被玩家攻击", "敌人进入视野") for tag in tags_at_t):
            if t > 0 and "Combat" not in bins[t - 1]["states"] and "Observe" not in bins[t - 1]["states"]:
                return True
        if any(tag in ("上载具", "下载具") for tag in tags_at_t):
            return True
        if "落地" in tags_at_t:
            return True

    if evidence_shift_large(bins, t):
        return True

    if _local_state_changed(bins, t, lookback=6, lookahead=6):
        return True

    if _activity_burst(bins, t, lookback=5):
        return True

    return False

def _soften_evidence(ev: Dict[str, float]) -> Dict[str, Any]:
    def _level(count, thresholds=(1, 3, 6)):
        if count <= 0:
            return "none"
        elif count < thresholds[1]:
            return "low"
        elif count < thresholds[2]:
            return "moderate"
        else:
            return "high"

    combat_cnt = ev.get("combat_count", 0) + ev.get("hit_count", 0)
    loot_cnt = ev.get("loot_count", 0) + ev.get("equip_count", 0)
    enemy_cnt = int(ev.get("peak_enemy_visible", 0) or ev.get("enemy_visible_count", 0))
    teammate_cnt = int(ev.get("peak_teammate_visible", 0) or ev.get("teammate_visible_count", 0))

    return {
        "combat": combat_cnt > 0,
        "combat_level": _level(combat_cnt, (1, 4, 10)),
        "loot": loot_cnt > 0,
        "loot_level": _level(loot_cnt, (1, 4, 8)),
        "enemy_visible": enemy_cnt > 0,
        "enemy_count": min(enemy_cnt, 9),
        "enemy_count_level": _level(enemy_cnt, (1, 3, 5)),
        "teammate_visible": teammate_cnt > 0,
        "teammate_count": min(teammate_cnt, 9),
        "hp_drop": ev.get("damage_taken", 0) > 0,
        "knockdown": ev.get("knockdown_count", 0) > 0,
        "eliminate": ev.get("eliminate_count", 0) > 0,
        "vehicle_used": ev.get("vehicle_count", 0) > 0,
    }

def finalize_window(window_bins: List[Dict], window_id: int) -> Dict[str, Any]:
    if not window_bins:
        return {}

    start_time = window_bins[0]["start"]
    end_time = window_bins[-1]["end"]

    pw = window_state_distribution(window_bins)
    q = run_state_distribution(window_bins)
    dominant = max(q, key=q.get)
    ev = aggregate_evidence(window_bins)

    mn = margin_norm(window_bins)
    cr = conf_ratio(window_bins)
    ec = ent_clarity(window_bins)
    sr = support_rate(window_bins)
    tc = tag_clarity(window_bins, B1, B2, B3, B4)

    return {
        "window_id": window_id,
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "duration_secs": round(end_time - start_time, 2),
        "num_bins": len(window_bins),
        "dominant_state": dominant,
        "state_distribution": {s: round(v, 4) for s, v in pw.items()},
        "run_state_distribution": {s: round(v, 4) for s, v in q.items()},
        "evidence": _soften_evidence(ev),
        "evidence_raw": {k: round(v, 2) for k, v in ev.items()},
        "metrics": {
            "entropy": round(state_entropy(window_bins), 4),
            "purity": round(state_purity(window_bins), 4),
            "transition_rate": round(state_transition_rate(window_bins), 4),
            "longest_run_ratio": round(longest_state_run_ratio(window_bins), 4),
            "consistency": round(window_consistency(window_bins, W1, W2, W3, W4), 4),
        },
        "tag_clarity": {
            "margin_norm": round(mn, 4),
            "conf_ratio": round(cr, 4),
            "ent_clarity": round(ec, 4),
            "support_rate": round(sr, 4),
            "tag_clarity": round(tc, 4),
        },
    }

def _mbu_merge_pass(all_bins: List[Dict], boundary_indices: List[int],
                    bin_size: float) -> List[int]:
    if len(boundary_indices) <= 1:
        return boundary_indices

    starts = [0] + boundary_indices
    ends = boundary_indices + [len(all_bins)]

    mbu_scores = []
    for i, b_idx in enumerate(boundary_indices):
        W_left = all_bins[starts[i]:ends[i]]
        W_right = all_bins[starts[i + 1]:ends[i + 1]]

        mbu = boundary_utility_single(
            W_left, W_right, bin_size,
            alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
            w1=W1, w2=W2, w3=W3, w4=W4,
            l1=L1, l2=L2,
            min_window_secs=5.0,
            max_window_secs=180.0,
        )

        tc_left = tag_clarity(W_left, B1, B2, B3, B4)
        tc_right = tag_clarity(W_right, B1, B2, B3, B4)
        tc_merged = tag_clarity(W_left + W_right, B1, B2, B3, B4)

        tc_protected = False
        avg_tc_sides = (tc_left + tc_right) / 2.0
        tc_drop = avg_tc_sides - tc_merged
        if (avg_tc_sides >= TAG_CLARITY_PROTECT_THRESHOLD
                and tc_drop > 0.15):
            tc_protected = True

        sep = inter_window_separability(W_left, W_right, L1, L2)
        weak_sep = sep < MIN_SEPARABILITY_FOR_BOUNDARY

        mbu_scores.append((i, b_idx, mbu, tc_protected, weak_sep))

    to_remove = []
    for i, b_idx, mbu, tc_prot, weak_sep in mbu_scores:
        if weak_sep and mbu <= MBU_MERGE_THRESHOLD * 3:
            to_remove.append((i, b_idx, mbu))
        elif mbu <= MBU_MERGE_THRESHOLD and not tc_prot:
            to_remove.append((i, b_idx, mbu))

    if not to_remove:
        return boundary_indices

    to_remove.sort(key=lambda x: x[2])
    worst_i, worst_b_idx, worst_mbu = to_remove[0]

    new_boundaries = [b for b in boundary_indices if b != worst_b_idx]
    return new_boundaries

def mbu_post_optimize(all_bins: List[Dict], boundary_indices: List[int],
                      bin_size: float, max_passes: int = MBU_MAX_PASSES,
                      verbose: bool = False) -> List[int]:
    current = list(boundary_indices)
    removed_total = 0

    for pass_num in range(max_passes):
        prev_count = len(current)
        prev_state = list(current)
        current = _mbu_merge_pass(all_bins, current, bin_size)
        removed = prev_count - len(current)
        removed_total += removed

        if verbose and removed > 0:
            print(f"    MBU pass {pass_num + 1}: removed {removed} boundary "
                  f"({len(current)} remaining)")

        if removed == 0:
            break

        starts = [0] + current
        ends = current + [len(all_bins)]
        valid = True
        for i in range(len(starts)):
            dur = (ends[i] - starts[i]) * bin_size
            if dur > MAX_WINDOW_SECS * 2.0:
                valid = False
                break
        if not valid:
            current = prev_state
            removed_total -= removed
            break

    if verbose:
        print(f"    MBU optimization: removed {removed_total} redundant boundaries "
              f"({len(boundary_indices)} -> {len(current)})")

    return current

def _merge_same_state_boundaries(all_bins: List[Dict],
                                  boundary_indices: List[int],
                                  bin_size: float,
                                  verbose: bool = False) -> List[int]:
    if not boundary_indices:
        return boundary_indices

    current = sorted(boundary_indices)
    removed_total = 0

    for _ in range(50):
        starts = [0] + current
        ends = current + [len(all_bins)]
        to_remove_idx = None
        best_score = -1e9

        for i, b_idx in enumerate(current):
            W_left = all_bins[starts[i]:ends[i]]
            W_right = all_bins[starts[i + 1]:ends[i + 1]]

            if not W_left or not W_right:
                continue

            q_left = run_state_distribution(W_left)
            q_right = run_state_distribution(W_right)
            dom_left = max(q_left, key=q_left.get)
            dom_right = max(q_right, key=q_right.get)

            if dom_left != dom_right:
                continue

            merged_dur = (ends[i + 1] - starts[i]) * bin_size
            if merged_dur > MAX_WINDOW_SECS:
                continue

            mbu = boundary_utility_single(
                W_left, W_right, bin_size,
                alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                w1=W1, w2=W2, w3=W3, w4=W4,
                l1=L1, l2=L2,
                min_window_secs=5.0,
                max_window_secs=MAX_WINDOW_SECS,
            )

            if mbu > MBU_MERGE_THRESHOLD * 3:
                continue

            score = -mbu
            if score > best_score:
                best_score = score
                to_remove_idx = i

        if to_remove_idx is None:
            break

        removed_b = current[to_remove_idx]
        current = [b for b in current if b != removed_b]
        removed_total += 1

    if verbose and removed_total > 0:
        print(f"    Phase 2.5 (same-state merge): removed {removed_total} "
              f"same-state boundaries")

    return current

def _find_best_split(all_bins: List[Dict], w_start: int, w_end: int,
                     bin_size: float) -> int:
    best_gain = -1e9
    best_t = -1
    min_bins = max(int(MIN_WINDOW_SECS / bin_size), 1)

    for t in range(w_start + min_bins, w_end - min_bins + 1):
        if not is_candidate_boundary(all_bins, t):
            continue
        gain = evaluate_cut_gain(
            all_bins, w_start, t, bin_size,
            alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
            w1=W1, w2=W2, w3=W3, w4=W4,
            l1=L1, l2=L2,
            lookahead_bins=LOOKAHEAD_BINS,
            min_window_secs=MIN_WINDOW_SECS,
            max_window_secs=MAX_WINDOW_SECS,
        )
        if gain > best_gain:
            best_gain = gain
            best_t = t

    if best_gain > CUT_THRESHOLD * 3 and best_t > 0:
        return best_t
    return -1

def _split_long_windows(all_bins: List[Dict], boundary_indices: List[int],
                        bin_size: float, max_window_bins: int = 60,
                        verbose: bool = False) -> List[int]:
    starts = [0] + sorted(boundary_indices)
    ends = sorted(boundary_indices) + [len(all_bins)]
    new_boundaries = list(boundary_indices)
    added = 0

    for i in range(len(starts)):
        w_start = starts[i]
        w_end = ends[i]
        w_len = w_end - w_start

        if w_len <= max_window_bins:
            continue

        wb = all_bins[w_start:w_end]
        rent = state_entropy_run_based(wb)
        tc = tag_clarity(wb, B1, B2, B3, B4)

        needs_split = False
        if rent >= 0.20:
            needs_split = True
        elif tc < TAG_CLARITY_SPLIT_THRESHOLD:
            needs_split = True
        elif w_len > max_window_bins * 2:
            needs_split = True

        if not needs_split:
            continue

        split_t = _find_best_split(all_bins, w_start, w_end, bin_size)
        if split_t > 0 and split_t not in new_boundaries:
            wb_left = all_bins[w_start:split_t]
            wb_right = all_bins[split_t:w_end]
            tc_left = tag_clarity(wb_left, B1, B2, B3, B4)
            tc_right = tag_clarity(wb_right, B1, B2, B3, B4)
            avg_tc_after = (tc_left + tc_right) / 2.0

            if avg_tc_after >= tc - 0.05:
                new_boundaries.append(split_t)
                added += 1

    if verbose and added > 0:
        print(f"    Phase 3 (split): added {added} splits for long windows")

    return sorted(new_boundaries)

def segment_match_log(events: List[Dict], bin_size: float = 1.0,
                      use_mbu_optimization: bool = True,
                      verbose: bool = True) -> List[Dict]:
    bins = build_time_bins(events, bin_size)
    if not bins:
        return []

    boundary_indices = []
    cur_start = 0
    t = 1

    while t < len(bins):
        current_duration = (t - cur_start) * bin_size

        if is_candidate_boundary(bins, t):
            gain = evaluate_cut_gain(
                bins, cur_start, t, bin_size,
                alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                w1=W1, w2=W2, w3=W3, w4=W4,
                l1=L1, l2=L2,
                lookahead_bins=LOOKAHEAD_BINS,
                min_window_secs=MIN_WINDOW_SECS,
                max_window_secs=MAX_WINDOW_SECS,
            )

            adaptive_threshold = CUT_THRESHOLD
            if current_duration > 60:
                adaptive_threshold = CUT_THRESHOLD * 0.5
            elif current_duration > 40:
                adaptive_threshold = CUT_THRESHOLD * 0.65
            elif current_duration > 25:
                adaptive_threshold = CUT_THRESHOLD * 0.8
            elif current_duration < 20:
                adaptive_threshold = CUT_THRESHOLD * 1.5

            if gain > adaptive_threshold and current_duration >= MIN_WINDOW_SECS:
                boundary_indices.append(t)
                cur_start = t

        if current_duration >= MAX_WINDOW_SECS:
            boundary_indices.append(t)
            cur_start = t

        t += 1

    initial_count = len(boundary_indices)

    if use_mbu_optimization and boundary_indices:
        if verbose:
            print(f"  Phase 1 (sweep): {initial_count} boundaries found")
            print(f"  Phase 2 (MBU optimization):")

        boundary_indices = mbu_post_optimize(
            bins, boundary_indices, bin_size,
            max_passes=MBU_MAX_PASSES, verbose=verbose,
        )

    boundary_indices = _merge_same_state_boundaries(
        bins, boundary_indices, bin_size, verbose=verbose,
    )

    max_window_bins = int(MAX_WINDOW_SECS * 0.5 / bin_size)
    boundary_indices = _split_long_windows(
        bins, boundary_indices, bin_size,
        max_window_bins=max_window_bins, verbose=verbose,
    )

    if use_mbu_optimization:
        boundary_indices = mbu_post_optimize(
            bins, boundary_indices, bin_size,
            max_passes=MBU_MAX_PASSES, verbose=False,
        )

    boundary_indices = _merge_same_state_boundaries(
        bins, boundary_indices, bin_size, verbose=False,
    )

    windows = []
    starts = [0] + boundary_indices
    ends = boundary_indices + [len(bins)]

    for window_id, (s, e) in enumerate(zip(starts, ends)):
        window = finalize_window(bins[s:e], window_id)
        if window:
            windows.append(window)

    return windows

def format_time(secs: float) -> str:
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m:02d}:{s:02d}"

def print_summary(windows: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("RULE-BASED UTILITY-AWARE SEGMENTATION RESULTS")
    print("=" * 80)
    print(f"Total windows: {len(windows)}")
    print()

    for w in windows:
        start_fmt = format_time(w["start_time"])
        end_fmt = format_time(w["end_time"])
        tc_info = w.get("tag_clarity", {})
        tc_val = tc_info.get("tag_clarity", 0.0)
        sr_val = tc_info.get("support_rate", 0.0)
        print(f"  Window {w['window_id']:3d}  |  {start_fmt} - {end_fmt}  "
              f"({w['duration_secs']:6.1f}s)  |  {w['dominant_state']:<10s}  |  "
              f"purity={w['metrics']['purity']:.2f}  entropy={w['metrics']['entropy']:.2f}  "
              f"TR={w['metrics']['transition_rate']:.2f}  LSR={w['metrics']['longest_run_ratio']:.2f}  "
              f"C={w['metrics']['consistency']:.3f}  "
              f"TC={tc_val:.2f}  SR={sr_val:.2f}")

    print()

    tc_values = [w.get("tag_clarity", {}).get("tag_clarity", 0.0) for w in windows]
    if tc_values:
        avg_tc = sum(tc_values) / len(tc_values)
        min_tc = min(tc_values)
        max_tc = max(tc_values)
        low_tc_count = sum(1 for v in tc_values if v < TAG_CLARITY_SPLIT_THRESHOLD)
        print(f"TagClarity summary: avg={avg_tc:.3f}  min={min_tc:.3f}  "
              f"max={max_tc:.3f}  low_clarity_windows={low_tc_count}/{len(windows)}")
        print()

    total_dur = sum(w["duration_secs"] for w in windows)
    if total_dur > 0 and windows:
        w_consistency = sum(
            w["metrics"]["consistency"] * w["duration_secs"] for w in windows
        ) / total_dur
        w_tc = sum(
            w.get("tag_clarity", {}).get("tag_clarity", 0.0) * w["duration_secs"]
            for w in windows
        ) / total_dur
        eq_consistency = sum(
            w["metrics"]["consistency"] for w in windows
        ) / len(windows)
        eq_tc = sum(tc_values) / len(tc_values) if tc_values else 0.0
        print(f"Duration-weighted averages:  wSemCons={w_consistency:.4f}  wTagClarity={w_tc:.4f}")
        print(f"Equal-weight averages:       SemCons={eq_consistency:.4f}   TagClarity={eq_tc:.4f}")
        print()

    state_durations = defaultdict(float)
    total_duration = 0.0
    for w in windows:
        dur = w["duration_secs"]
        state_durations[w["dominant_state"]] += dur
        total_duration += dur

    print("State breakdown:")
    for s in sorted(state_durations.keys()):
        pct = state_durations[s] / total_duration * 100 if total_duration > 0 else 0
        print(f"  {s:<12s}: {state_durations[s]:7.1f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<12s}: {total_duration:7.1f}s")
    print("=" * 80)

def save_results(windows: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "num_windows": len(windows),
        "total_duration": sum(w["duration_secs"] for w in windows),
        "windows": windows,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Rule-based Utility-aware Segmentation for Game TLog"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to tlog text file")
    parser.add_argument("--bin_size", "-b", type=float, default=1.0,
                        help="Time bin size in seconds (default: 1.0)")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory (default: ../output relative to script)")
    parser.add_argument("--min_window", type=float, default=15.0,
                        help="Minimum window duration in seconds (default: 15.0)")
    parser.add_argument("--max_window", type=float, default=180.0,
                        help="Maximum window duration in seconds (default: 180.0)")
    parser.add_argument("--cut_threshold", type=float, default=0.05,
                        help="Minimum utility gain to accept a cut (default: 0.05)")
    args = parser.parse_args()

    global MIN_WINDOW_SECS, MAX_WINDOW_SECS, CUT_THRESHOLD
    MIN_WINDOW_SECS = args.min_window
    MAX_WINDOW_SECS = args.max_window
    CUT_THRESHOLD = args.cut_threshold

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), "output")
    else:
        output_dir = args.output_dir

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Parsing tlog: {input_path}")
    events = parse_tlog(input_path)
    print(f"Parsed {len(events)} events, time range: "
          f"{events[0]['time']:.1f}s - {events[-1]['time']:.1f}s")

    print(f"Running segmentation (bin_size={args.bin_size}s, "
          f"min_window={MIN_WINDOW_SECS}s, max_window={MAX_WINDOW_SECS}s)...")
    windows = segment_match_log(events, bin_size=args.bin_size)

    print_summary(windows)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_gt_segments.json")
    save_results(windows, output_path)

if __name__ == "__main__":
    main()
