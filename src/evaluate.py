
import os
import sys
import json
import argparse
import random
from collections import defaultdict
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from map import parse_tlog, map_tag_to_states, update_evidence, init_evidence_vector
from metrics import (
    ALL_STATES,
    per_bin_state_distribution,
    window_state_distribution,
    state_entropy,
    state_purity,
    state_entropy_bin_based,
    state_purity_bin_based,
    state_transition_rate,
    longest_state_run_ratio,
    window_consistency,
    aggregate_evidence,
    state_divergence,
    evidence_separability,
    inter_window_separability,
    compute_mbu,
    margin_norm,
    conf_ratio,
    ent_clarity,
    local_margin_norm,
    support_rate,
    tag_clarity,
    compute_tag_clarity_stats,
)
from segment_gt import segment_match_log, build_time_bins, format_time

segment_match_log_refined = None
TLogParser = None
try:
    from segment_llm import segment_with_llm, create_client, DEFAULT_MODEL, DEFAULT_BASE_URL
except ImportError:
    segment_with_llm = None
    create_client = None
    DEFAULT_MODEL = "unavailable"
    DEFAULT_BASE_URL = ""

def build_fixed_window_bins(all_bins: List[Dict], window_seconds: float,
                            bin_size: float = 1.0) -> List[List[Dict]]:
    if not all_bins:
        return []

    bins_per_window = max(1, int(window_seconds / bin_size))
    window_bins_list = []

    for i in range(0, len(all_bins), bins_per_window):
        chunk = all_bins[i:i + bins_per_window]
        if chunk:
            window_bins_list.append(chunk)

    return window_bins_list

def build_random_window_bins(
    all_bins: List[Dict],
    rng: random.Random,
    *,
    target_num_windows: int = None,
    target_mean_duration: float = None,
    min_window_seconds: float = 8.0,
    bin_size: float = 1.0,
) -> List[List[Dict]]:
    if not all_bins:
        return []

    n_bins = len(all_bins)
    min_bins = max(1, int(round(min_window_seconds / bin_size)))
    boundaries = set()

    if target_num_windows and target_num_windows > 1:
        needed = max(0, target_num_windows - 1)
        valid = list(range(min_bins, n_bins - min_bins + 1))
        rng.shuffle(valid)
        for idx in valid:
            if all(abs(idx - b) >= min_bins for b in boundaries):
                boundaries.add(idx)
            if len(boundaries) >= needed:
                break
    else:
        target_mean_duration = target_mean_duration or 30.0
        target_mean_bins = max(min_bins, int(round(target_mean_duration / bin_size)))
        cur = 0
        while cur + min_bins < n_bins:
            jitter = rng.randint(-max(1, target_mean_bins // 3), max(1, target_mean_bins // 3))
            step = max(min_bins, target_mean_bins + jitter)
            nxt = cur + step
            if nxt >= n_bins - min_bins:
                break
            boundaries.add(nxt)
            cur = nxt

    sorted_boundaries = sorted(boundaries)
    starts = [0] + sorted_boundaries
    ends = sorted_boundaries + [n_bins]
    return [all_bins[s:e] for s, e in zip(starts, ends) if s < e]

V1_CATEGORY_TO_STATE = {
    "开局动作": "Parachute",
    "核心战斗": "Combat",
    "关键成长": "Looting",
    "战术状态": "Rotate",
    "死亡": "Unknown",
}

def map_v1_category(category: str) -> str:
    base_cat = category.replace("(补)", "").strip()
    return V1_CATEGORY_TO_STATE.get(base_cat, "Unknown")

def build_bins_for_v1_windows(events: List[Dict], v1_windows: List[Dict],
                               bin_size: float = 1.0) -> List[List[Dict]]:
    all_window_bins = []

    for win in v1_windows:
        w_start = win["start"]
        w_end = win["end"]

        win_events = [e for e in events if w_start <= e["time"] < w_end]

        if not win_events and w_end > w_start:
            state = map_v1_category(win["category"])
            synthetic_bin = {
                "index": 0,
                "start": w_start,
                "end": w_end,
                "tags": [],
                "events": [],
                "states": {state} if state != "Unknown" else set(),
                "evidence": init_evidence_vector(),
                "dominant_state": state,
            }
            all_window_bins.append([synthetic_bin])
            continue

        num_bins = max(1, int((w_end - w_start) / bin_size))

        bins = []
        for i in range(num_bins):
            bins.append({
                "index": i,
                "start": w_start + i * bin_size,
                "end": w_start + (i + 1) * bin_size,
                "tags": [],
                "events": [],
                "states": set(),
                "evidence": init_evidence_vector(),
                "dominant_state": "Unknown",
            })

        for e in win_events:
            idx = int((e["time"] - w_start) / bin_size)
            idx = min(idx, num_bins - 1)
            bins[idx]["tags"].append(e["tag"])
            bins[idx]["events"].append(e)
            mapped = map_tag_to_states(e)
            bins[idx]["states"].update(mapped)
            update_evidence(bins[idx]["evidence"], e)

        for b in bins:
            if b["states"]:
                state_counts = defaultdict(int)
                for e in b["events"]:
                    for s in map_tag_to_states(e):
                        state_counts[s] += 1
                if state_counts:
                    b["dominant_state"] = max(state_counts, key=state_counts.get)
            else:
                state = map_v1_category(win["category"])
                b["states"] = {state} if state != "Unknown" else set()
                b["dominant_state"] = state

        all_window_bins.append(bins)

    return all_window_bins

def compute_window_metrics(window_bins_list: List[List[Dict]]) -> Dict[str, Any]:
    per_window = []
    n = len(window_bins_list)

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
            "consistency": window_consistency(wb),
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

    inter_metrics = []
    for i in range(n - 1):
        w1 = window_bins_list[i]
        w2 = window_bins_list[i + 1]
        d_state = state_divergence(w1, w2)
        d_evidence = evidence_separability(w1, w2)
        d_combined = inter_window_separability(w1, w2)
        inter_metrics.append({
            "pair": f"{i}-{i+1}",
            "state_divergence": d_state,
            "evidence_separability": d_evidence,
            "combined_separability": d_combined,
        })

    intra_keys = ["entropy", "purity",
                  "transition_rate", "lsr", "consistency",
                  "margin_norm", "conf_ratio", "ent_clarity",
                  "local_margin", "support_rate", "tag_clarity"]

    avg_intra = {}
    for key in intra_keys:
        vals = [m[key] for m in per_window]
        avg_intra[key] = sum(vals) / len(vals) if vals else 0.0

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

    avg_inter = {}
    for key in ["state_divergence", "evidence_separability", "combined_separability"]:
        vals = [m[key] for m in inter_metrics]
        avg_inter[key] = sum(vals) / len(vals) if vals else 0.0

    pair_weights = []
    for i in range(n - 1):
        dur1 = per_window[i]["duration"]
        dur2 = per_window[i + 1]["duration"]
        pair_weights.append((dur1 + dur2) / 2.0)

    total_pair_weight = sum(pair_weights)
    wavg_inter = {}
    for key in ["state_divergence", "evidence_separability", "combined_separability"]:
        if total_pair_weight > 0 and inter_metrics:
            wavg_inter[key] = sum(
                inter_metrics[i][key] * pair_weights[i]
                for i in range(len(inter_metrics))
            ) / total_pair_weight
        else:
            wavg_inter[key] = 0.0

    mbu_result = compute_mbu(window_bins_list, bin_size=1.0)

    per_boundary = mbu_result.get("per_boundary", [])
    if per_boundary and pair_weights:
        n_pairs = min(len(per_boundary), len(pair_weights))
        total_pw = sum(pair_weights[:n_pairs])
        if total_pw > 0:
            mbu_result["wavg_mbu"] = sum(
                per_boundary[i]["mbu"] * pair_weights[i]
                for i in range(n_pairs)
            ) / total_pw
            mbu_result["wpositive_ratio"] = sum(
                (1.0 if per_boundary[i]["mbu"] > 0 else 0.0) * pair_weights[i]
                for i in range(n_pairs)
            ) / total_pw
        else:
            mbu_result["wavg_mbu"] = 0.0
            mbu_result["wpositive_ratio"] = 0.0
    else:
        mbu_result["wavg_mbu"] = 0.0
        mbu_result["wpositive_ratio"] = 0.0

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

def print_comparison(name: str, result: Dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Total windows: {result['num_windows']}")

    avg = result["avg_intra"]
    wavg = result.get("wavg_intra", avg)
    avg_inter = result["avg_inter"]
    mbu = result.get("mbu", {})

    print(f"\n  ┌─── Primary Metrics (duration-weighted) ──────────────────────┐")
    print(f"  │  wSemCons (Weighted Semantic Consistency) : {wavg['consistency']:.4f}          │")
    print(f"  │  wTagClarity (Weighted Label Clarity)     : {wavg.get('tag_clarity', 0):.4f}          │")
    print(f"  │  Sep (Inter-window Separability)          : {avg_inter['combined_separability']:.4f}          │")
    print(f"  │  mMBU (Median Boundary Utility)           : {mbu.get('median_mbu', 0):.4f}          │")
    print(f"  │  MBU+ (Positive Boundary Ratio)           : {mbu.get('positive_ratio', 0):.2%}           │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    print(f"\n  Per-window Details:")
    print(f"  {'Win':>4s}  {'Time':>14s}  {'Dur':>6s}  {'State':<12s}  "
          f"{'SemC':>7s}  {'TC':>6s}  "
          f"{'Ent':>6s}  {'Pur':>6s}  {'TR':>6s}  {'LSR':>6s}  "
          f"{'MgN':>6s}  {'CfR':>6s}  {'ECl':>6s}  {'SpR':>6s}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*6}  {'-'*12}  "
          f"{'-'*7}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for m in result["per_window"]:
        s_fmt = format_time(m["start"])
        e_fmt = format_time(m["end"])
        print(f"  {m['window_id']:4d}  {s_fmt}-{e_fmt}  {m['duration']:6.1f}  "
              f"{m['dominant_state']:<12s}  "
              f"{m['consistency']:7.4f}  {m.get('tag_clarity', 0):6.3f}  "
              f"{m['entropy']:6.4f}  {m['purity']:6.4f}  "
              f"{m['transition_rate']:6.4f}  {m['lsr']:6.4f}  "
              f"{m.get('margin_norm', 0):6.3f}  {m.get('conf_ratio', 0):6.3f}  "
              f"{m.get('ent_clarity', 0):6.3f}  {m.get('support_rate', 0):6.3f}")

    print(f"\n  --- Sub-metric Averages: Duration-Weighted (Primary) ---")
    print(f"  [wSemCons breakdown]  Ent={wavg['entropy']:.4f}  Pur={wavg['purity']:.4f}  "
          f"TR={wavg['transition_rate']:.4f}  LSR={wavg['lsr']:.4f}")
    print(f"  [wTagClarity breakdown]  MgN={wavg.get('margin_norm', 0):.4f}  "
          f"CfR={wavg.get('conf_ratio', 0):.4f}  ECl={wavg.get('ent_clarity', 0):.4f}  "
          f"LcM={wavg.get('local_margin', 0):.4f}  SpR={wavg.get('support_rate', 0):.4f}")
    print(f"\n  --- Sub-metric Averages: Equal-Weight (Reference) ---")
    print(f"  [SemCons breakdown]  Ent={avg['entropy']:.4f}  Pur={avg['purity']:.4f}  "
          f"TR={avg['transition_rate']:.4f}  LSR={avg['lsr']:.4f}")
    print(f"  [TagClarity breakdown]  MgN={avg.get('margin_norm', 0):.4f}  "
          f"CfR={avg.get('conf_ratio', 0):.4f}  ECl={avg.get('ent_clarity', 0):.4f}  "
          f"LcM={avg.get('local_margin', 0):.4f}  SpR={avg.get('support_rate', 0):.4f}")
    print(f"\n  [Sep breakdown]  SDiv={avg_inter['state_divergence']:.4f}  "
          f"ESep={avg_inter['evidence_separability']:.4f}  "
          f"CSep={avg_inter['combined_separability']:.4f}")

    if mbu and mbu.get("per_boundary"):
        print(f"  [MBU breakdown]  Avg={mbu['avg_mbu']:.4f}  "
              f"Med={mbu['median_mbu']:.4f}  "
              f"Min={mbu['min_mbu']:.4f}  Max={mbu['max_mbu']:.4f}  "
              f"Pos={mbu['positive_ratio']:.2%} "
              f"({sum(1 for b in mbu['per_boundary'] if b['mbu'] > 0)}"
              f"/{len(mbu['per_boundary'])} justified)")
    print(f"{'=' * 70}")

def print_side_by_side(results_dict: Dict[str, Dict]) -> None:
    labels = list(results_dict.keys())
    results = results_dict

    primary_col_defs = [
        ("num_windows",                        "#Win",       None),
        ("wavg_intra.consistency",             "wSemC↑",     True),
        ("wavg_intra.tag_clarity",             "wTagCl↑",    True),
        ("avg_intra.consistency",              "SemC↑",      True),
        ("avg_intra.tag_clarity",              "TagCl↑",     True),
        ("avg_inter.combined_separability",    "Sep↑",       True),
        ("wavg_inter.combined_separability",   "wSep↑",      True),
        ("mbu.median_mbu",                     "mMBU↑",      True),
        ("mbu.positive_ratio",                 "MBU+↑",      True),
        ("mbu.wavg_mbu",                       "wMBU↑",      True),
        ("mbu.wpositive_ratio",                "wMBU+↑",     True),
    ]

    appendix_col_defs = [
        ("wavg_intra.entropy",               "wEnt↓",   False),
        ("wavg_intra.purity",                "wPur↑",   True),
        ("wavg_intra.transition_rate",       "wTR↓",    False),
        ("wavg_intra.lsr",                   "wLSR↑",   True),
        ("wavg_intra.margin_norm",           "wMgN↑",   True),
        ("wavg_intra.conf_ratio",            "wCfR↑",   True),
        ("wavg_intra.ent_clarity",           "wECl↑",   True),
        ("wavg_intra.support_rate",          "wSpR↑",   True),
        ("avg_inter.state_divergence",       "SDiv↑",   True),
        ("avg_inter.evidence_separability",  "ESep↑",   True),
        ("mbu.avg_mbu",                      "MBU↑",    True),
        ("mbu.min_mbu",                      "mnMBU",   True),
        ("mbu.max_mbu",                      "mxMBU",   True),
    ]

    col_defs = primary_col_defs

    def get_val(result, key_path):
        parts = key_path.split(".")
        v = result
        for p in parts:
            v = v[p]
        return v

    model_col_w = max(max(len(l) for l in labels), 8)
    metric_col_w = max(7, max(len(cd[1]) for cd in col_defs))

    n_metrics = len(col_defs)
    total_width = model_col_w + 4 + n_metrics * (metric_col_w + 1) + 2

    print(f"\n{'=' * total_width}")
    print(f"  SIDE-BY-SIDE COMPARISON (transposed)")
    print(f"{'=' * total_width}")

    header = f"  {'Model':<{model_col_w}s}"
    for _, short_hdr, _ in col_defs:
        header += f" {short_hdr:>{metric_col_w}s}"
    print(header)

    sep = f"  {'-' * model_col_w}"
    for _ in col_defs:
        sep += f" {'-' * metric_col_w}"
    print(sep)

    all_vals = {}
    for label in labels:
        for key_path, _, _ in col_defs:
            all_vals[(label, key_path)] = get_val(results[label], key_path)

    best_for_metric = {}
    for key_path, _, higher_better in col_defs:
        if higher_better is None:
            best_for_metric[key_path] = None
            continue
        metric_vals = {l: all_vals[(l, key_path)] for l in labels}
        if higher_better:
            best_for_metric[key_path] = max(metric_vals, key=metric_vals.get)
        else:
            best_for_metric[key_path] = min(metric_vals, key=metric_vals.get)

    pct_keys = {"mbu.positive_ratio", "mbu.wpositive_ratio"}

    for label in labels:
        row = f"  {label:<{model_col_w}s}"
        for key_path, _, higher_better in col_defs:
            val = all_vals[(label, key_path)]
            is_best = (best_for_metric.get(key_path) == label)
            if isinstance(val, int):
                cell = f"{val:>{metric_col_w}d}"
            elif key_path in pct_keys:
                pct_str = f"{val * 100:.1f}%"
                cell = f"{pct_str:>{metric_col_w}s}"
            else:
                cell = f"{val:>{metric_col_w}.4f}"
            if is_best and higher_better is not None:
                cell = cell[:-1] + "*"
            row += f" {cell}"
        print(row)

    print(sep)
    print(f"  (* = best for that metric)")
    print(f"{'=' * total_width}")

    col_defs_app = appendix_col_defs
    n_metrics_app = len(col_defs_app)
    metric_col_w_app = max(7, max(len(cd[1]) for cd in col_defs_app))
    total_width_app = model_col_w + 4 + n_metrics_app * (metric_col_w_app + 1) + 2

    print(f"\n{'=' * total_width_app}")
    print(f"  APPENDIX: Sub-metric Details")
    print(f"{'=' * total_width_app}")

    header_app = f"  {'Model':<{model_col_w}s}"
    for _, short_hdr, _ in col_defs_app:
        header_app += f" {short_hdr:>{metric_col_w_app}s}"
    print(header_app)

    sep_app = f"  {'-' * model_col_w}"
    for _ in col_defs_app:
        sep_app += f" {'-' * metric_col_w_app}"
    print(sep_app)

    all_vals_app = {}
    for label in labels:
        for key_path, _, _ in col_defs_app:
            all_vals_app[(label, key_path)] = get_val(results[label], key_path)

    best_for_app = {}
    for key_path, _, higher_better in col_defs_app:
        if higher_better is None:
            best_for_app[key_path] = None
            continue
        metric_vals = {l: all_vals_app[(l, key_path)] for l in labels}
        if higher_better:
            best_for_app[key_path] = max(metric_vals, key=metric_vals.get)
        else:
            best_for_app[key_path] = min(metric_vals, key=metric_vals.get)

    for label in labels:
        row = f"  {label:<{model_col_w}s}"
        for key_path, _, higher_better in col_defs_app:
            val = all_vals_app[(label, key_path)]
            is_best = (best_for_app.get(key_path) == label)
            if isinstance(val, int):
                cell = f"{val:>{metric_col_w_app}d}"
            else:
                cell = f"{val:>{metric_col_w_app}.4f}"
            if is_best and higher_better is not None:
                cell = cell[:-1] + "*"
            row += f" {cell}"
        print(row)

    print(sep_app)
    print(f"  (* = best for that metric)")
    print(f"{'=' * total_width_app}")

def build_bins_for_llm_windows(events: List[Dict], llm_windows: List[Dict],
                                bin_size: float = 1.0) -> List[List[Dict]]:
    all_window_bins = []

    for win in llm_windows:
        w_start = win["start"]
        w_end = win["end"]
        llm_state = win.get("state", "Unknown")

        win_events = [e for e in events if w_start <= e["time"] < w_end]

        if not win_events and w_end > w_start:
            synthetic_bin = {
                "index": 0,
                "start": w_start,
                "end": w_end,
                "tags": [],
                "events": [],
                "states": {llm_state} if llm_state != "Unknown" else set(),
                "evidence": init_evidence_vector(),
                "dominant_state": llm_state,
            }
            all_window_bins.append([synthetic_bin])
            continue

        num_bins = max(1, int((w_end - w_start) / bin_size))

        bins = []
        for i in range(num_bins):
            bins.append({
                "index": i,
                "start": w_start + i * bin_size,
                "end": w_start + (i + 1) * bin_size,
                "tags": [],
                "events": [],
                "states": set(),
                "evidence": init_evidence_vector(),
                "dominant_state": "Unknown",
            })

        for e in win_events:
            idx = int((e["time"] - w_start) / bin_size)
            idx = min(idx, num_bins - 1)
            bins[idx]["tags"].append(e["tag"])
            bins[idx]["events"].append(e)
            mapped = map_tag_to_states(e)
            bins[idx]["states"].update(mapped)
            update_evidence(bins[idx]["evidence"], e)

        for b in bins:
            if b["states"]:
                state_counts = defaultdict(int)
                for e in b["events"]:
                    for s in map_tag_to_states(e):
                        state_counts[s] += 1
                if state_counts:
                    b["dominant_state"] = max(state_counts, key=state_counts.get)
            else:
                b["states"] = {llm_state} if llm_state != "Unknown" else set()
                b["dominant_state"] = llm_state

        all_window_bins.append(bins)

    return all_window_bins

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation approaches")
    parser.add_argument("--input", "-i", required=True, help="Path to tlog text file")
    parser.add_argument("--bin_size", "-b", type=float, default=1.0,
                        help="Time bin size in seconds (default: 1.0)")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory for JSON results")
    parser.add_argument("--fixed_window", "-fw", type=float, default=30.0,
                        help="Fixed window size in seconds (default: 30.0)")
    parser.add_argument("--random_baseline", action="store_true", default=False,
                        help="Evaluate a random-window baseline")
    parser.add_argument("--random_mode", choices=["matched_count", "matched_duration"],
                        default="matched_count",
                        help="Random baseline matching strategy")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random baseline seed")
    parser.add_argument("--llm", action="store_true", default=False,
                        help="Run LLM-based segmentation")
    parser.add_argument("--llm_token", default=None,
                        help="API token for LLM (or set VENUS_API_TOKEN env var)")
    parser.add_argument("--llm_model", default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--llm_base_url", default=DEFAULT_BASE_URL,
                        help="LLM API base URL")
    parser.add_argument("--llm_result", default=None,
                        help="Path to pre-computed LLM segments JSON (skip inference)")
    parser.add_argument("--extra_results", nargs="+", default=[],
                        help="Paths to additional segmentation result JSONs "
                             "(e.g. baseline1.json baseline2.json). "
                             "Each must have a 'windows' key with [{start, end, state}, ...]")
    parser.add_argument("--realtime_result", default=None,
                        help="Path to pubg_real_time_results.json (multi-match format). "
                             "Use with --match_id to select a specific match.")
    parser.add_argument("--match_id", default=None,
                        help="Match ID to extract from --realtime_result file "
                             "(e.g. 7578333527687976403_2778393426)")
    args = parser.parse_args()

    input_path = args.input
    bin_size = args.bin_size

    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), "output")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing tlog: {input_path}")
    events = parse_tlog(input_path)
    print(f"Parsed {len(events)} events, time range: "
          f"{events[0]['time']:.1f}s - {events[-1]['time']:.1f}s")

    if segment_match_log_refined is None:
        print("\n[1/4] Refined legacy oracle is unavailable in this release; using the public oracle.")
        refined_oracle_windows = segment_match_log(events, bin_size=bin_size)
    else:
        print("\n[1/4] Running refined oracle segmentation...")
        refined_oracle_windows = segment_match_log_refined(events, bin_size=bin_size, verbose=True)

    gt_all_bins = build_time_bins(events, bin_size)

    refined_oracle_window_bins_list = []
    for w in refined_oracle_windows:
        t_start_offset = events[0]["time"]
        start_idx = int((w["start_time"] - t_start_offset) / bin_size)
        end_idx = int((w["end_time"] - t_start_offset) / bin_size)
        start_idx = max(0, start_idx)
        end_idx = min(len(gt_all_bins), end_idx)
        wb = gt_all_bins[start_idx:end_idx]
        if wb:
            refined_oracle_window_bins_list.append(wb)

    refined_oracle_result = compute_window_metrics(refined_oracle_window_bins_list)
    print_comparison("Refined-Oracle", refined_oracle_result)

    print("\n[2/4] Running segment_gt (utility-aware)...")
    gt_windows = segment_match_log(events, bin_size=bin_size)

    gt_window_bins_list = []
    for w in gt_windows:
        t_start_offset = events[0]["time"]
        start_idx = int((w["start_time"] - t_start_offset) / bin_size)
        end_idx = int((w["end_time"] - t_start_offset) / bin_size)
        start_idx = max(0, start_idx)
        end_idx = min(len(gt_all_bins), end_idx)
        wb = gt_all_bins[start_idx:end_idx]
        if wb:
            gt_window_bins_list.append(wb)

    gt_result = compute_window_metrics(gt_window_bins_list)
    print_comparison("Offline-Oracle", gt_result)

    heuristic_result = None
    if TLogParser is None:
        print("\n[3/4] Legacy heuristic baseline is unavailable in this release; skipping.")
    else:
        print("\n[3/4] Running legacy heuristic baseline...")
        heuristic_parser = TLogParser()
        heuristic_windows_raw = heuristic_parser.process(input_path)
        print(f"  Heuristic baseline produced {len(heuristic_windows_raw)} windows")
        heuristic_window_bins_list = build_bins_for_v1_windows(events, heuristic_windows_raw, bin_size)
        heuristic_result = compute_window_metrics(heuristic_window_bins_list)
        print_comparison("Heuristic-Baseline", heuristic_result)

    fixed_sec = args.fixed_window
    print(f"\n[4/4] Running fixed-window segmentation ({fixed_sec}s)...")

    fixed_window_bins_list = build_fixed_window_bins(gt_all_bins, fixed_sec, bin_size)
    print(f"  Fixed-window produced {len(fixed_window_bins_list)} windows")

    fixed_result = compute_window_metrics(fixed_window_bins_list)
    print_comparison(f"fixed-window ({fixed_sec}s)", fixed_result)

    random_result = None
    if args.random_baseline:
        rng = random.Random(args.random_seed)
        target_num_windows = refined_oracle_result["num_windows"] if args.random_mode == "matched_count" else None
        target_mean_duration = (
            refined_oracle_result["total_duration"] / max(refined_oracle_result["num_windows"], 1)
            if args.random_mode == "matched_duration" else None
        )
        random_window_bins_list = build_random_window_bins(
            gt_all_bins,
            rng,
            target_num_windows=target_num_windows,
            target_mean_duration=target_mean_duration,
            min_window_seconds=8.0,
            bin_size=bin_size,
        )
        print(f"\n[Random] Running random-window baseline ({args.random_mode})...")
        print(f"  Random baseline produced {len(random_window_bins_list)} windows")
        random_result = compute_window_metrics(random_window_bins_list)
        print_comparison(f"random-window ({args.random_mode})", random_result)

    llm_result = None
    llm_windows = None

    if args.llm_result:
        print(f"\n[4/4] Loading pre-computed LLM segmentation from {args.llm_result}...")
        with open(args.llm_result, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        llm_windows = llm_data.get("windows", [])
        print(f"  Loaded {len(llm_windows)} LLM windows")

    elif args.llm:
        if segment_with_llm is None or create_client is None:
            print("\nWARNING: segment_llm module is unavailable in this environment. Skipping LLM.")
        else:
            token = args.llm_token or os.environ.get("VENUS_API_TOKEN", "")
            if not token:
                print("\nWARNING: --llm specified but no token provided. "
                      "Use --llm_token or set VENUS_API_TOKEN. Skipping LLM.")
            else:
                print(f"\n[LLM] Running LLM segmentation ({args.llm_model})...")
                client = create_client(token, args.llm_base_url)
                llm_windows = segment_with_llm(
                    events, client,
                    model=args.llm_model,
                    verbose=True,
                )

                basename_llm = os.path.splitext(os.path.basename(input_path))[0]
                llm_output_path = os.path.join(output_dir, f"{basename_llm}_llm_segments.json")
                os.makedirs(output_dir, exist_ok=True)
                with open(llm_output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "method": "llm_segmentation",
                        "model": args.llm_model,
                        "num_windows": len(llm_windows),
                        "total_duration": round(
                            sum(w["end"] - w["start"] for w in llm_windows), 2),
                        "windows": llm_windows,
                    }, f, ensure_ascii=False, indent=2)
                print(f"  LLM segments saved to: {llm_output_path}")

    if llm_windows is not None:
        llm_window_bins_list = build_bins_for_llm_windows(events, llm_windows, bin_size)
        llm_result = compute_window_metrics(llm_window_bins_list)
        print_comparison("segment_llm (LLM-based)", llm_result)

    if args.realtime_result and args.match_id:
        rt_path = args.realtime_result
        match_id = args.match_id
        if not os.path.isfile(rt_path):
            print(f"WARNING: realtime_result file not found: {rt_path}")
        else:
            print(f"\n[RT] Loading real-time result for match {match_id} from {rt_path}...")
            with open(rt_path, 'r', encoding='utf-8') as f:
                rt_data = json.load(f)

            if match_id not in rt_data:
                print(f"  WARNING: match_id '{match_id}' not found in {rt_path}")
                print(f"  Available match IDs (first 10): {list(rt_data.keys())[:10]}")
            else:
                rt_match = rt_data[match_id]
                converted_windows = []
                for w in rt_match["windows"]:
                    converted_windows.append({
                        "start": w["start_time"],
                        "end": w["end_time"],
                        "state": w.get("predicted_label", "Unknown"),
                        "duration": w.get("duration", w["end_time"] - w["start_time"]),
                    })

                converted_path = os.path.join(
                    output_dir, f"{match_id}_realtime_converted.json")
                converted_data = {
                    "method": "pubg_realtime_vlm",
                    "source": rt_path,
                    "match_id": match_id,
                    "num_windows": len(converted_windows),
                    "total_duration": rt_match.get("total_duration", 0),
                    "windows": converted_windows,
                }
                with open(converted_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, ensure_ascii=False, indent=2)
                print(f"  Converted {len(converted_windows)} windows, saved to: {converted_path}")

                args.extra_results.append(converted_path)

    extra_results = {}

    for extra_path in args.extra_results:
        if not os.path.isfile(extra_path):
            print(f"\nWARNING: extra result file not found: {extra_path}, skipping.")
            continue

        print(f"\n[Extra] Loading external segmentation from {extra_path}...")
        with open(extra_path, 'r', encoding='utf-8') as f:
            extra_data = json.load(f)

        extra_windows = extra_data.get("windows", [])
        extra_method = extra_data.get("method", os.path.splitext(os.path.basename(extra_path))[0])

        label = extra_method
        label_map = {
            "decision_head_LLM_baseline0": "baseline0_llm",
            "decision_head_baseline1": "baseline1_llm",
            "decision_head_vision_mlp_baseline1": "baseline1_mlp",
            "decision_head_vision_gru_baseline1": "baseline1_gru",
            "decision_head_vision_transformer_baseline1": "baseline1_transformer",
            "decision_head_vision_adaptor_llm_baseline2": "baseline2_adaptor_llm",
            "pubg_realtime_vlm": "Dispider",
            "epistream_policy": "EpiStream",
        }
        display_label = label_map.get(label, label)

        print(f"  Method: {extra_method}, Windows: {len(extra_windows)}")

        if not extra_windows:
            print(f"  WARNING: no windows found in {extra_path}, skipping.")
            continue

        extra_window_bins_list = build_bins_for_llm_windows(events, extra_windows, bin_size)
        extra_metric = compute_window_metrics(extra_window_bins_list)
        print_comparison(f"{display_label}", extra_metric)
        extra_results[display_label] = extra_metric

    comparison_results = {
        "Refined-Oracle": refined_oracle_result,
        "Offline-Oracle": gt_result,
        f"fixed_{int(fixed_sec)}s": fixed_result,
    }
    if heuristic_result is not None:
        comparison_results["Heuristic-Baseline"] = heuristic_result
    if random_result is not None:
        comparison_results[f"random_{args.random_mode}"] = random_result
    if llm_result is not None:
        comparison_results["segment_llm"] = llm_result

    for label, result in extra_results.items():
        comparison_results[label] = result

    print_side_by_side(comparison_results)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_evaluation.json")

    def result_to_json(result):
        mbu = result.get("mbu", {})
        mbu_json = {
            "avg_mbu": round(mbu.get("avg_mbu", 0.0), 4),
            "median_mbu": round(mbu.get("median_mbu", 0.0), 4),
            "min_mbu": round(mbu.get("min_mbu", 0.0), 4),
            "max_mbu": round(mbu.get("max_mbu", 0.0), 4),
            "positive_ratio": round(mbu.get("positive_ratio", 0.0), 4),
            "per_boundary": [
                {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in b.items()}
                for b in mbu.get("per_boundary", [])
            ],
        }
        return {
            "num_windows": result["num_windows"],
            "avg_intra": {k: round(v, 4) for k, v in result["avg_intra"].items()},
            "avg_inter": {k: round(v, 4) for k, v in result["avg_inter"].items()},
            "mbu": mbu_json,
            "per_window": [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                for m in result["per_window"]
            ],
        }

    def wavg_to_json(result):
        wavg = result.get("wavg_intra", {})
        return {k: round(v, 4) for k, v in wavg.items()} if wavg else {}

    output_data = {
        "input_file": input_path,
        "bin_size": bin_size,
        "fixed_window_seconds": fixed_sec,
        "refined_oracle": {**result_to_json(refined_oracle_result), "wavg_intra": wavg_to_json(refined_oracle_result)},
        "offline_oracle": {**result_to_json(gt_result), "wavg_intra": wavg_to_json(gt_result)},
        f"fixed_window_{int(fixed_sec)}s": {**result_to_json(fixed_result), "wavg_intra": wavg_to_json(fixed_result)},
    }
    if heuristic_result is not None:
        output_data["heuristic_baseline"] = {
            **result_to_json(heuristic_result),
            "wavg_intra": wavg_to_json(heuristic_result),
        }
    if random_result is not None:
        output_data[f"random_window_{args.random_mode}"] = {
            **result_to_json(random_result),
            "wavg_intra": wavg_to_json(random_result),
        }
    if llm_result is not None:
        output_data["segment_llm"] = {**result_to_json(llm_result), "wavg_intra": wavg_to_json(llm_result)}

    for label, result in extra_results.items():
        output_data[label] = {**result_to_json(result), "wavg_intra": wavg_to_json(result)}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation results saved to: {output_path}")

if __name__ == "__main__":
    main()
