
import os
import sys
import json
import re
import argparse
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from map import (
    parse_tlog,
    map_tag_to_states,
)
from segment_gt import build_time_bins

FRAME_BASE_DIR = os.environ.get("FRAME_BASE_DIR", "/path/to/frame_screenshots")
TLOG_BASE_DIR = os.environ.get("TLOG_BASE_DIR", "/path/to/tlog_files")
MEMORY_CACHE_PATH = os.environ.get("MEMORY_CACHE_PATH") or os.path.join(
    PROJECT_DIR, "cache", "memory", "GT-Original.json"
)

DEFAULT_INTERVAL_DENSE = 0.5
DEFAULT_INTERVAL_SPARSE = 2.0
DEFAULT_BOUNDARY_WINDOW = 10.0
DEFAULT_NUM_FRAMES = 8
DEFAULT_BIN_SIZE = 1.0
BOUNDARY_TOLERANCE = 1.5
MIN_WINDOW_SECS = 5.0
NUM_PREV_WINDOWS = 2

TEST_MATCH_IDS = {
    "7578333527687976403_2778393426",
}

TRIGGER_EVENT_MAP = {
    "敌人进入视野": "enemy_seen",
    "射击武器": "shot_fired",
    "射击命中玩家": "shot_hit",
    "攻击玩家": "dealt_damage",
    "受到伤害": "hit_taken",
    "被玩家攻击": "hit_taken",
    "击倒玩家": "enemy_knocked",
    "淘汰玩家": "enemy_eliminated",
    "上载具": "vehicle_enter",
    "下载具": "vehicle_exit",
    "进入房子": "enter_building",
    "离开房子": "leave_building",
    "拾取物品": "loot_start",
    "装备物品": "gear_update",
    "完成使用物品": "healing_used",
    "开始换弹": "reload_start",
    "换弹完成": "reload_done",
    "落地": "landed",
    "离开飞机": "leave_plane",
    "开伞": "open_parachute",
}

def compute_sampling_points(
    video_duration: float,
    boundary_times: List[float],
    interval_dense: float = DEFAULT_INTERVAL_DENSE,
    interval_sparse: float = DEFAULT_INTERVAL_SPARSE,
    boundary_window: float = DEFAULT_BOUNDARY_WINDOW,
) -> List[float]:
    sampling_points = set()
    
    boundary_regions = []
    for bt in boundary_times:
        start = max(0, bt - boundary_window)
        end = min(video_duration, bt + boundary_window)
        boundary_regions.append((start, end))
    
    boundary_regions.sort()
    merged_regions = []
    for start, end in boundary_regions:
        if merged_regions and start <= merged_regions[-1][1]:
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], end))
        else:
            merged_regions.append((start, end))
    
    for start, end in merged_regions:
        t = start
        while t <= end:
            if 0 <= t <= video_duration:
                sampling_points.add(t)
            t += interval_dense
    
    t = 0
    while t <= video_duration:
        in_boundary_region = any(start <= t <= end for start, end in merged_regions)
        if not in_boundary_region:
            sampling_points.add(t)
        t += interval_sparse
    
    sampling_points.add(0.0)
    sampling_points.add(video_duration)
    
    return sorted(sampling_points)

def get_sampling_statistics(
    sampling_points: List[float],
    boundary_times: List[float],
    boundary_window: float,
) -> Dict[str, Any]:
    total_samples = len(sampling_points)
    
    boundary_samples = 0
    for t in sampling_points:
        if any(abs(t - bt) <= boundary_window for bt in boundary_times):
            boundary_samples += 1
    
    stable_samples = total_samples - boundary_samples
    
    if len(sampling_points) > 1:
        intervals = [sampling_points[i+1] - sampling_points[i] 
                    for i in range(len(sampling_points)-1)]
        avg_interval = np.mean(intervals)
        median_interval = np.median(intervals)
    else:
        avg_interval = median_interval = 0.0
    
    return {
        "total_samples": total_samples,
        "boundary_region_samples": boundary_samples,
        "stable_region_samples": stable_samples,
        "avg_interval": round(avg_interval, 3),
        "median_interval": round(median_interval, 3),
    }

def parse_frame_timestamp(filename: str) -> Optional[float]:
    m = re.search(r'-([\d.]+)\.jpg$', filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def load_frame_index(match_id: str) -> List[Tuple[float, str]]:
    frame_dir = os.path.join(FRAME_BASE_DIR, match_id)
    if not os.path.isdir(frame_dir):
        return []
    
    frames = []
    for fname in os.listdir(frame_dir):
        if not fname.endswith(".jpg"):
            continue
        ts = parse_frame_timestamp(fname)
        if ts is not None:
            frames.append((ts, os.path.join(frame_dir, fname)))
    
    frames.sort(key=lambda x: x[0])
    return frames

def get_recent_frames(
    frame_index: List[Tuple[float, str]],
    current_time: float,
    num_frames: int,
) -> List[Dict[str, Any]]:
    candidates = [(ts, path) for ts, path in frame_index if ts <= current_time]
    if not candidates:
        return []
    selected = candidates[-num_frames:]
    return [{"timestamp": round(ts, 3), "path": path} for ts, path in selected]

def normalize_cache_window(window: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_time": float(window["start"]),
        "end_time": float(window["end"]),
        "duration": float(window.get("duration", window["end"] - window["start"])),
        "label": window.get("original_label") or window.get("label") or window.get("state") or "Unknown",
    }

def load_memory_cache_matches() -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(MEMORY_CACHE_PATH):
        return {}

    with open(MEMORY_CACHE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    matches = raw.get("matches", {}) if isinstance(raw, dict) else {}
    cache_map: Dict[str, Dict[str, Any]] = {}

    for match_id, match_data in matches.items():
        if not isinstance(match_data, dict):
            continue

        raw_windows = match_data.get("windows", [])
        norm_windows = [normalize_cache_window(w) for w in raw_windows if isinstance(w, dict)]
        if not norm_windows:
            continue

        cache_map[match_id] = {
            "match_id": match_id,
            "windows": norm_windows,
            "cards_rule": match_data.get("cards_rule", []),
            "cards_vlm": match_data.get("cards_vlm", []),
            "match_duration": float(match_data.get("match_duration", norm_windows[-1]["end_time"])),
        }

    return cache_map

def load_match_data(match_id: str, cache_map: Dict[str, Dict[str, Any]]) -> Optional[Dict]:
    return cache_map.get(match_id)

def get_boundary_times(gt_data: Dict) -> List[float]:
    windows = gt_data["windows"]
    boundaries = []
    for i in range(len(windows) - 1):
        boundaries.append(windows[i]["end_time"])
    return sorted(boundaries)

def is_boundary_point(
    current_time: float,
    boundary_times: List[float],
    tolerance: float = BOUNDARY_TOLERANCE,
) -> bool:
    for bt in boundary_times:
        if abs(current_time - bt) <= tolerance:
            return True
    return False

def find_current_window_idx(current_time: float, gt_windows: List[Dict]) -> int:
    for i, w in enumerate(gt_windows):
        if w["start_time"] <= current_time < w["end_time"]:
            return i
        if i == len(gt_windows) - 1 and current_time <= w["end_time"]:
            return i
    return len(gt_windows) - 1

def find_nearest_boundary(current_time: float, boundary_times: List[float]) -> Dict[str, float]:
    if not boundary_times:
        return {"boundary_time": 0.0, "distance": float('inf')}
    distances = [abs(current_time - bt) for bt in boundary_times]
    min_idx = np.argmin(distances)
    return {
        "boundary_time": boundary_times[min_idx],
        "distance": distances[min_idx],
    }

def get_window_state_label(window: Dict[str, Any]) -> str:
    return window.get("dominant_state") or window.get("label") or window.get("state") or "Unknown"

def get_window_events(
    events: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
) -> List[Dict[str, Any]]:
    return [e for e in events if start_time <= e["time"] < end_time]

def infer_enemy_contact(window_events: List[Dict[str, Any]]) -> str:
    tags = {e["tag"] for e in window_events}
    if any(tag in tags for tag in ("射击武器", "射击命中玩家", "攻击玩家", "受到伤害", "被玩家攻击", "击倒玩家", "淘汰玩家")):
        return "active_fight"
    if "敌人进入视野" in tags:
        return "visible"
    return "none"

def infer_team_state(window_events: List[Dict[str, Any]]) -> str:
    teammate_counts = [
        e.get("payload", {}).get("teammate_count", 0)
        for e in window_events
        if e["tag"] in ("敌人进入视野", "敌人离开视野")
    ]
    peak = max(teammate_counts) if teammate_counts else 0
    if peak >= 2:
        return "grouped"
    if peak == 1:
        return "teammate_nearby"
    return "isolated"

def infer_resource_state(window_events: List[Dict[str, Any]]) -> str:
    tags = [e["tag"] for e in window_events]
    if any(tag in tags for tag in ("完成使用物品",)):
        return "healing_needed"
    if sum(1 for tag in tags if tag in ("拾取物品", "装备物品", "更换配件", "卸下装备")) >= 2:
        return "looting"
    if sum(1 for tag in tags if tag in ("开始换弹", "换弹完成")) >= 3:
        return "ammo_low"
    return "stable"

def infer_position_state(
    dominant_state: str,
    window_events: List[Dict[str, Any]],
) -> str:
    tags = {e["tag"] for e in window_events}
    if "进入房子" in tags and "离开房子" not in tags:
        return "indoor"
    if dominant_state in ("Rotate", "Parachute") or any(tag in tags for tag in ("上载具", "下载具")):
        return "rotating"
    if dominant_state in ("Combat", "Observe"):
        return "near_cover"
    return "stable"

def infer_episode_type(
    dominant_state: str,
    enemy_contact: str,
    resource_state: str,
    position_state: str,
) -> str:
    if dominant_state == "Parachute":
        return "landing"
    if enemy_contact == "active_fight":
        return "active_engagement"
    if enemy_contact == "visible":
        return "early_contest"
    if position_state == "rotating":
        return "midgame_rotation"
    if resource_state == "looting":
        return "looting_phase"
    if dominant_state == "Observe":
        return "scouting_hold"
    return "transition_phase"

def infer_tactical_intent(
    enemy_contact: str,
    resource_state: str,
    position_state: str,
) -> str:
    if enemy_contact == "active_fight":
        return "engage"
    if enemy_contact == "visible":
        return "prepare_engagement"
    if resource_state == "healing_needed":
        return "recover"
    if resource_state == "looting":
        return "loot"
    if position_state == "rotating":
        return "rotate"
    return "hold"

def infer_phase_status(
    current_duration: float,
    enemy_contact: str,
    resource_state: str,
    trigger_events: List[str],
) -> str:
    if enemy_contact == "active_fight" or "enemy_seen" in trigger_events:
        return "escalating"
    if resource_state == "healing_needed":
        return "resolving"
    if current_duration <= 10.0:
        return "onset"
    return "ongoing"

def infer_future_hint(
    tactical_intent: str,
    enemy_contact: str,
    resource_state: str,
) -> str:
    if enemy_contact == "active_fight":
        return "engage"
    if enemy_contact == "visible":
        return "prepare_engagement"
    if resource_state == "looting":
        return "loot"
    if resource_state == "healing_needed":
        return "recover"
    if tactical_intent == "rotate":
        return "rotate"
    return "hold"

def extract_trigger_events(window_events: List[Dict[str, Any]], max_events: int = 4) -> List[str]:
    triggers = []
    for e in window_events:
        mapped = TRIGGER_EVENT_MAP.get(e["tag"])
        if mapped and mapped not in triggers:
            triggers.append(mapped)
        if len(triggers) >= max_events:
            break
    return triggers

def infer_warning_ready(
    enemy_contact: str,
    team_state: str,
    resource_state: str,
    position_state: str,
) -> bool:
    signals = 0
    if enemy_contact in ("visible", "active_fight"):
        signals += 1
    if team_state == "isolated":
        signals += 1
    if resource_state in ("ammo_low", "healing_needed"):
        signals += 1
    if position_state in ("exposed",):
        signals += 1
    return signals >= 2 or enemy_contact == "active_fight"

def summarize_window_schema(
    unit_id: str,
    start_time: float,
    end_time: float,
    dominant_state: str,
    window_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    current_duration = max(0.0, end_time - start_time)
    trigger_events = extract_trigger_events(window_events)
    enemy_contact = infer_enemy_contact(window_events)
    team_state = infer_team_state(window_events)
    resource_state = infer_resource_state(window_events)
    position_state = infer_position_state(dominant_state, window_events)
    tactical_intent = infer_tactical_intent(enemy_contact, resource_state, position_state)
    phase_status = infer_phase_status(current_duration, enemy_contact, resource_state, trigger_events)
    future_hint = infer_future_hint(tactical_intent, enemy_contact, resource_state)
    warning_ready = infer_warning_ready(enemy_contact, team_state, resource_state, position_state)

    return {
        "unit_id": unit_id,
        "episode_type": infer_episode_type(dominant_state, enemy_contact, resource_state, position_state),
        "tactical_intent": tactical_intent,
        "phase_status": phase_status,
        "state_summary": {
            "enemy_contact": enemy_contact,
            "team_state": team_state,
            "resource_state": resource_state,
            "position_state": position_state,
        },
        "trigger_events": trigger_events,
        "future_hint": future_hint,
        "warning_ready": warning_ready,
    }

def format_schema_block(tag_name: str, schema: Dict[str, Any]) -> str:
    state_summary = schema["state_summary"]
    trigger_text = ", ".join(schema["trigger_events"]) if schema["trigger_events"] else "none"
    ready_text = "true" if schema["warning_ready"] else "false"
    lines = [
        f"<{tag_name}>",
        f"episode_type={schema['episode_type']}",
        f"tactical_intent={schema['tactical_intent']}",
        f"phase_status={schema['phase_status']}",
        "state_summary={"
        f"enemy_contact={state_summary['enemy_contact']}; "
        f"team_state={state_summary['team_state']}; "
        f"resource_state={state_summary['resource_state']}; "
        f"position_state={state_summary['position_state']}"
        "}",
        f"trigger_events=[{trigger_text}]",
        f"future_hint={schema['future_hint']}",
        f"warning_ready={ready_text}",
        f"</{tag_name}>",
    ]
    return "\n".join(lines)

def format_cached_card_block(tag_name: str, card: Dict[str, Any]) -> str:
    state_summary = card.get("state_summary", {}) or {}
    trigger_events = card.get("trigger_events", []) or []
    trigger_text = ", ".join(trigger_events) if trigger_events else "none"
    ready_text = "true" if bool(card.get("warning_ready", False)) else "false"
    lines = [
        f"<{tag_name}>",
        f"episode_type={card.get('episode_type', 'unknown')}",
        f"tactical_intent={card.get('tactical_intent', 'unknown')}",
        f"phase_status={card.get('phase_status', 'ongoing')}",
        "state_summary={"
        f"enemy_contact={state_summary.get('enemy_contact', 'unknown')}; "
        f"team_state={state_summary.get('team_state', 'unknown')}; "
        f"resource_state={state_summary.get('resource_state', 'unknown')}; "
        f"position_state={state_summary.get('position_state', 'unknown')}"
        "}",
        f"trigger_events=[{trigger_text}]",
        f"future_hint={card.get('future_hint', 'hold')}",
        f"warning_ready={ready_text}",
        f"</{tag_name}>",
    ]
    return "\n".join(lines)

def build_memory_text(
    match_data: Dict[str, Any],
    current_window_idx: int,
    events: List[Dict[str, Any]],
    memory_source: str = "vlm",
) -> str:
    gt_windows = match_data["windows"]
    cards_vlm = match_data.get("cards_vlm", [])
    cards_rule = match_data.get("cards_rule", [])
    prev_windows = gt_windows[max(0, current_window_idx - NUM_PREV_WINDOWS):current_window_idx]
    blocks = ["<MEMORY>"]
    if not prev_windows:
        blocks.append("(none)")
    else:
        prev_start_idx = max(0, current_window_idx - NUM_PREV_WINDOWS)
        for offset, w in enumerate(prev_windows):
            tag_name = f"PREV_{len(prev_windows) - offset}"
            cache_idx = prev_start_idx + offset

            cached_card = None
            if memory_source == "vlm" and cache_idx < len(cards_vlm):
                cached_card = cards_vlm[cache_idx]
            elif memory_source == "rule" and cache_idx < len(cards_rule):
                cached_card = cards_rule[cache_idx]
            elif cache_idx < len(cards_vlm):
                cached_card = cards_vlm[cache_idx]
            elif cache_idx < len(cards_rule):
                cached_card = cards_rule[cache_idx]

            if isinstance(cached_card, dict):
                blocks.append(format_cached_card_block(tag_name, cached_card))
                continue

            unit_id = f"prev_{offset+1}_{int(w['start_time'])}_{int(w['end_time'])}"
            window_events = get_window_events(events, w["start_time"], w["end_time"])
            schema = summarize_window_schema(
                unit_id=unit_id,
                start_time=w["start_time"],
                end_time=w["end_time"],
                dominant_state=get_window_state_label(w),
                window_events=window_events,
            )
            blocks.append(format_schema_block(tag_name, schema))
    blocks.append("</MEMORY>")
    return "\n".join(blocks)

def build_current_window_text(
    current_window: Dict[str, Any],
    current_time: float,
    events: List[Dict[str, Any]],
) -> str:
    window_events = get_window_events(events, current_window["start_time"], current_time)
    schema = summarize_window_schema(
        unit_id=f"current_{int(current_window['start_time'])}_{int(current_time)}",
        start_time=current_window["start_time"],
        end_time=current_time,
        dominant_state=get_window_state_label(current_window),
        window_events=window_events,
    )
    return format_schema_block("CURRENT_WINDOW", schema)

def construct_sample(
    match_id: str,
    current_time: float,
    match_data: Dict,
    events: List[Dict[str, Any]],
    frame_index: List[Tuple[float, str]],
    boundary_times: List[float],
    num_frames: int,
    memory_source: str,
) -> Optional[Dict[str, Any]]:
    gt_windows = match_data["windows"]
    window_idx = find_current_window_idx(current_time, gt_windows)
    current_window = gt_windows[window_idx]
    
    window_duration = current_time - current_window["start_time"]
    if window_duration < MIN_WINDOW_SECS:
        return None
    
    frames = get_recent_frames(frame_index, current_time, num_frames)
    if not frames:
        return None
    
    is_boundary = is_boundary_point(current_time, boundary_times)
    label = "<CLOSE>" if is_boundary else "<CONTINUE>"
    
    nearest_boundary = find_nearest_boundary(current_time, boundary_times)
    
    memory_text = build_memory_text(match_data, window_idx, events, memory_source=memory_source)
    current_window_text = build_current_window_text(current_window, current_time, events)
    prompt = (
        f"{memory_text}\n\n"
        f"{current_window_text}\n\n"
        f"<CURRENT_VIDEO>\n[{len(frames)} frames]\n</CURRENT_VIDEO>\n\n"
        f"<BOUNDARY_QUERY>\nShould the current semantic window be closed now?"
    )
    
    return {
        "match_id": match_id,
        "sample_time": round(current_time, 3),
        "window_start_time": current_window["start_time"],
        "current_duration": round(window_duration, 1),
        "current_window_idx": window_idx,
        "current_window_state": get_window_state_label(current_window),
        "prompt": prompt,
        "label": label,
        "frame_paths": [f["path"] for f in frames],
        "frame_timestamps": [f["timestamp"] for f in frames],
        "num_frames": len(frames),
        "is_boundary": is_boundary,
        "nearest_boundary": nearest_boundary,
    }

def process_match(
    match_id: str,
    cache_map: Dict[str, Dict[str, Any]],
    interval_dense: float,
    interval_sparse: float,
    boundary_window: float,
    num_frames: int,
    memory_source: str,
) -> Tuple[List[Dict], Dict[str, Any]]:
    print(f"\nProcessing {match_id}...")
    
    match_data = load_match_data(match_id, cache_map)
    if not match_data:
        print(f"  ⚠️  No cached GT-Original match data found")
        return [], {}
    
    frame_index = load_frame_index(match_id)
    if not frame_index:
        print(f"  ⚠️  No frames found")
        return [], {}

    tlog_path = os.path.join(TLOG_BASE_DIR, f"{match_id}.txt")
    if os.path.isfile(tlog_path):
        events = parse_tlog(tlog_path)
    else:
        print(f"  ⚠️  No tlog found, using empty event list")
        events = []
    
    video_duration = min(frame_index[-1][0], match_data["match_duration"])
    boundary_times = get_boundary_times(match_data)
    
    sampling_points = compute_sampling_points(
        video_duration,
        boundary_times,
        interval_dense,
        interval_sparse,
        boundary_window,
    )
    
    stats = get_sampling_statistics(sampling_points, boundary_times, boundary_window)
    
    print(f"  Duration: {video_duration:.1f}s")
    print(f"  Boundaries: {len(boundary_times)}")
    print(f"  Sampling points: {stats['total_samples']}")
    print(f"    - Boundary regions: {stats['boundary_region_samples']}")
    print(f"    - Stable regions: {stats['stable_region_samples']}")
    print(f"    - Avg interval: {stats['avg_interval']:.2f}s")
    
    samples = []
    for t in sampling_points:
        sample = construct_sample(
            match_id,
            t,
            match_data,
            events,
            frame_index,
            boundary_times,
            num_frames,
            memory_source,
        )
        if sample:
            samples.append(sample)
    
    print(f"  ✅ Generated {len(samples)} samples")
    
    return samples, stats

def main():
    parser = argparse.ArgumentParser(description="Intelligent sampling dataset construction")
    parser.add_argument("--interval_dense", type=float, default=DEFAULT_INTERVAL_DENSE,
                       help="Dense sampling interval around boundaries (default: 0.5s)")
    parser.add_argument("--interval_sparse", type=float, default=DEFAULT_INTERVAL_SPARSE,
                       help="Sparse sampling interval in stable regions (default: 2.0s)")
    parser.add_argument("--boundary_window", type=float, default=DEFAULT_BOUNDARY_WINDOW,
                       help="Window size around boundaries for dense sampling (default: 10s)")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                       help="Number of frames per sample (default: 8)")
    parser.add_argument("--memory_source", choices=["vlm", "rule"], default="vlm",
                       help="Which cached memory cards to use for MEMORY blocks (default: vlm)")
    parser.add_argument("--output_dir", type=str, 
                       default="../dataset_v1/output_intelligent",
                       help="Output directory")
    parser.add_argument("--test_only", action="store_true",
                       help="Process only test set matches")
    
    args = parser.parse_args()
    
    output_dir = os.path.join(PROJECT_DIR, args.output_dir.lstrip("../"))
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("🧠 Intelligent Sampling Dataset Construction")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dense interval (boundary ±{args.boundary_window}s): {args.interval_dense}s")
    print(f"  Sparse interval (stable regions): {args.interval_sparse}s")
    print(f"  Frames per sample: {args.num_frames}")
    print(f"  Memory source: {args.memory_source}")
    print(f"  Output directory: {output_dir}")
    
    cache_map = load_memory_cache_matches()
    if not cache_map:
        print(f"\n❌ Memory cache not found or empty: {MEMORY_CACHE_PATH}")
        return
    
    all_match_ids = set(cache_map.keys())
    
    test_ids = all_match_ids & TEST_MATCH_IDS
    train_val_ids = all_match_ids - TEST_MATCH_IDS
    
    train_val_list = sorted(train_val_ids)
    split_idx = int(len(train_val_list) * 0.8)
    train_ids = set(train_val_list[:split_idx])
    val_ids = set(train_val_list[split_idx:])
    
    if args.test_only:
        match_ids_to_process = {"test": test_ids}
    else:
        match_ids_to_process = {"train": train_ids, "val": val_ids, "test": test_ids}
    
    global_stats = defaultdict(lambda: {"samples": 0, "boundary_samples": 0, "stable_samples": 0})
    
    for split, ids in match_ids_to_process.items():
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} set ({len(ids)} matches)")
        print(f"{'='*80}")
        
        all_samples = []
        for match_id in sorted(ids):
            samples, stats = process_match(
                match_id,
                cache_map,
                args.interval_dense,
                args.interval_sparse,
                args.boundary_window,
                args.num_frames,
                args.memory_source,
            )
            all_samples.extend(samples)
            if stats:
                global_stats[split]["samples"] += stats["total_samples"]
                global_stats[split]["boundary_samples"] += stats["boundary_region_samples"]
                global_stats[split]["stable_samples"] += stats["stable_region_samples"]
        
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Saved {len(all_samples)} samples to {output_file}")
    
    print(f"\n{'='*80}")
    print("📊 Global Statistics")
    print(f"{'='*80}")
    for split, stats in global_stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Total samples: {stats['samples']:,}")
        print(f"  Boundary region: {stats['boundary_samples']:,} ({stats['boundary_samples']/stats['samples']*100:.1f}%)")
        print(f"  Stable region: {stats['stable_samples']:,} ({stats['stable_samples']/stats['samples']*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("📈 Comparison with Uniform Sampling (1.0s)")
    print(f"{'='*80}")
    train_samples = global_stats['train']['samples']
    uniform_samples = 53582
    reduction = (1 - train_samples / uniform_samples) * 100
    print(f"  Original (uniform 1.0s): {uniform_samples:,} samples")
    print(f"  Intelligent sampling: {train_samples:,} samples")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Estimated training time reduction: {reduction:.1f}%")
    
    print(f"\n✅ Dataset construction complete!")

if __name__ == "__main__":
    main()
