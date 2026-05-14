
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any, Optional
from copy import deepcopy

LINE_PATTERN = re.compile(r'^\[(\d+\.?\d*)\]\s+(.+)$')

def parse_tlog(filepath: str) -> List[Dict[str, Any]]:
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = LINE_PATTERN.match(line)
            if not m:
                continue
            timestamp = float(m.group(1))
            raw_text = m.group(2)
            tag, payload = classify_event(raw_text)
            events.append({
                "time": timestamp,
                "raw": raw_text,
                "tag": tag,
                "payload": payload,
            })
    events.sort(key=lambda e: e["time"])
    return events

def classify_event(text: str) -> Tuple[str, Dict[str, Any]]:
    payload: Dict[str, Any] = {}

    if text.startswith("射击武器:"):
        payload["weapon"] = _extract_after(text, "射击武器:", ",")
        fired_m = re.search(r'发射:(\d+)发', text)
        hit_m = re.search(r'命中:(\d+)发', text)
        if fired_m:
            payload["fired"] = int(fired_m.group(1))
        if hit_m:
            payload["hit"] = int(hit_m.group(1))
        return ("射击武器", payload)

    if text.startswith("射击命中玩家:"):
        return ("射击命中玩家", payload)

    if text.startswith("攻击玩家:"):
        dmg_m = re.search(r'伤害:([\d.]+)', text)
        if dmg_m:
            payload["damage"] = float(dmg_m.group(1))
        return ("攻击玩家", payload)

    if text.startswith("受到伤害:"):
        dmg_m = re.search(r'受到伤害:([\d.]+)', text)
        if dmg_m:
            payload["damage"] = float(dmg_m.group(1))
        return ("受到伤害", payload)

    if text.startswith("被玩家:") and "攻击" in text:
        dmg_m = re.search(r'伤害:([\d.]+)', text)
        if dmg_m:
            payload["damage"] = float(dmg_m.group(1))
        return ("被玩家攻击", payload)

    if text.startswith("击倒玩家:"):
        return ("击倒玩家", payload)

    if text.startswith("淘汰玩家:"):
        return ("淘汰玩家", payload)

    if text.startswith("开始换弹"):
        return ("开始换弹", payload)

    if text.startswith("换弹完成"):
        return ("换弹完成", payload)

    if re.search(r'\d+名玩家.*进入视野', text):
        enemy_ids, teammate_ids = _parse_player_ids(text)
        payload["enemy_ids"] = enemy_ids
        payload["teammate_ids"] = teammate_ids
        payload["enemy_count"] = len(enemy_ids)
        payload["teammate_count"] = len(teammate_ids)
        return ("敌人进入视野", payload)

    if re.search(r'\d+名玩家.*离开视野', text):
        enemy_ids, teammate_ids = _parse_player_ids(text)
        payload["enemy_ids"] = enemy_ids
        payload["teammate_ids"] = teammate_ids
        payload["enemy_count"] = len(enemy_ids)
        payload["teammate_count"] = len(teammate_ids)
        return ("敌人离开视野", payload)

    if text.startswith("拾取物品:"):
        item = _extract_after(text, "拾取物品:", ",")
        payload["item"] = item
        return ("拾取物品", payload)

    if text.startswith("丢弃物品:"):
        return ("丢弃物品", payload)

    if text.startswith("卸下装备"):
        return ("卸下装备", payload)

    if text.startswith("装备物品:"):
        return ("装备物品", payload)

    if text.startswith("完成使用物品:"):
        return ("完成使用物品", payload)

    if text.startswith("枪械:") and "更换配件" in text:
        return ("更换配件", payload)

    if text == "进入房子":
        return ("进入房子", payload)

    if text == "离开房子":
        return ("离开房子", payload)

    if text.startswith("玩家上载具:"):
        vehicle = _extract_after(text, "玩家上载具:", None)
        payload["vehicle"] = vehicle
        return ("上载具", payload)

    if text.startswith("玩家下载具:"):
        vehicle = _extract_after(text, "玩家下载具:", None)
        payload["vehicle"] = vehicle
        return ("下载具", payload)

    if text == "玩家离开飞机":
        return ("离开飞机", payload)

    if text == "玩家开伞":
        return ("开伞", payload)

    if text == "玩家落地":
        return ("落地", payload)

    if text.startswith("玩家状态更新:"):
        state_m = re.search(r'状态:([^,]+)', text)
        if state_m:
            payload["status"] = state_m.group(1).strip()

        pos_m = re.search(r'位置:\(([^)]+)\)', text)
        if pos_m:
            coords = pos_m.group(1).split(',')
            if len(coords) == 3:
                payload["position"] = [float(c) for c in coords]

        speed_m = re.search(r'载具速度:([\d.]+)km/h', text)
        if speed_m:
            payload["vehicle_speed"] = float(speed_m.group(1))

        hp_m = re.search(r'血量:([\d.]+)', text)
        if hp_m:
            payload["hp"] = float(hp_m.group(1))

        return ("状态更新", payload)

    if "对局" in text and "开始" in text:
        return ("对局开始", payload)

    if "被淘汰" in text or "存活" in text:
        return ("对局结束", payload)

    if text.startswith("玩家基础信息"):
        return ("玩家信息", payload)

    if "进入滞空" in text or "结束滞空" in text:
        return ("滞空状态", payload)

    return ("其他", payload)

def _extract_after(text: str, prefix: str, delimiter: Optional[str]) -> str:
    start = text.find(prefix)
    if start < 0:
        return ""
    start += len(prefix)
    if delimiter:
        end = text.find(delimiter, start)
        if end < 0:
            return text[start:].strip()
        return text[start:end].strip()
    return text[start:].strip()

def _parse_player_ids(text: str) -> Tuple[List[str], List[str]]:
    m = re.search(r'名玩家\((.+?)\)(?:进入|离开)视野', text)
    if not m:
        return [], []

    raw = m.group(1)
    enemy_ids: List[str] = []
    teammate_ids: List[str] = []

    for entry in re.finditer(r'([^,()]+)\(([^)]+)\)', raw):
        uid = entry.group(1).strip()
        tag = entry.group(2).strip()
        if tag == '队友':
            teammate_ids.append(uid)
        else:
            enemy_ids.append(uid)

    return enemy_ids, teammate_ids

class VisibilityTracker:

    def __init__(self):
        self.visible_enemies: Set[str] = set()
        self.visible_teammates: Set[str] = set()
        self.peak_enemies: int = 0
        self.peak_teammates: int = 0

    def update(self, event: Dict) -> None:
        tag = event["tag"]
        payload = event.get("payload", {})

        if tag == "敌人进入视野":
            for eid in payload.get("enemy_ids", []):
                self.visible_enemies.add(eid)
            for tid in payload.get("teammate_ids", []):
                self.visible_teammates.add(tid)
        elif tag == "敌人离开视野":
            for eid in payload.get("enemy_ids", []):
                self.visible_enemies.discard(eid)
            for tid in payload.get("teammate_ids", []):
                self.visible_teammates.discard(tid)

        self.peak_enemies = max(self.peak_enemies, len(self.visible_enemies))
        self.peak_teammates = max(self.peak_teammates, len(self.visible_teammates))

    def reset_peaks(self) -> None:
        self.peak_enemies = len(self.visible_enemies)
        self.peak_teammates = len(self.visible_teammates)

    @property
    def current_enemy_count(self) -> int:
        return len(self.visible_enemies)

    @property
    def current_teammate_count(self) -> int:
        return len(self.visible_teammates)

def map_tag_to_states(event: Dict) -> Set[str]:
    return map_tag_to_states_with_phase(event, in_parachute_phase=False)

def map_tag_to_states_with_phase(event: Dict, in_parachute_phase: bool = False) -> Set[str]:
    tag = event["tag"]
    payload = event.get("payload", {})

    if tag in ("拾取物品", "丢弃物品", "卸下装备", "装备物品", "完成使用物品", "更换配件"):
        return {"Looting"}

    if tag in ("射击武器", "射击命中玩家", "攻击玩家", "受到伤害", "被玩家攻击",
               "击倒玩家", "淘汰玩家", "开始换弹", "换弹完成"):
        return {"Combat"}

    if tag in ("上载具", "下载具", "进入房子", "离开房子"):
        return {"Rotate"}

    if tag == "状态更新":
        status = payload.get("status", "")
        vehicle_speed = payload.get("vehicle_speed", None)

        if "驾驶载具" in status or (vehicle_speed is not None and vehicle_speed > 1.0):
            return {"Rotate"}

        if "跳伞" in status or "在飞机上" in status:
            if in_parachute_phase:
                return {"Parachute"}
            else:
                return set()

        if "移动中" in status:
            return {"Rotate"}

        return set()

    if tag in ("离开飞机", "开伞"):
        return {"Parachute"}

    if tag == "落地":
        return {"Parachute"}

    if tag in ("敌人进入视野",):
        return {"Observe"}

    if tag in ("对局开始", "对局结束", "玩家信息", "敌人离开视野", "滞空状态", "其他"):
        return set()

    return set()

def init_evidence_vector() -> Dict[str, float]:
    return {
        "combat_count": 0.0,
        "hit_count": 0.0,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "loot_count": 0.0,
        "equip_count": 0.0,
        "rotate_count": 0.0,
        "observe_count": 0.0,
        "recover_count": 0.0,
        "vehicle_count": 0.0,
        "enemy_visible_count": 0.0,
        "teammate_visible_count": 0.0,
        "peak_enemy_visible": 0.0,
        "peak_teammate_visible": 0.0,
        "knockdown_count": 0.0,
        "eliminate_count": 0.0,
    }

def update_evidence(
    ev: Dict[str, float],
    event: Dict,
    tracker: Optional["VisibilityTracker"] = None,
) -> None:
    tag = event["tag"]
    payload = event.get("payload", {})

    if tag in ("射击武器", "攻击玩家"):
        ev["combat_count"] += 1

    if tag == "射击命中玩家":
        ev["hit_count"] += 1

    if tag == "攻击玩家":
        ev["damage_dealt"] += payload.get("damage", 0.0)

    if tag in ("受到伤害", "被玩家攻击"):
        ev["damage_taken"] += payload.get("damage", 0.0)

    if tag in ("拾取物品", "丢弃物品", "卸下装备"):
        ev["loot_count"] += 1

    if tag in ("装备物品", "完成使用物品", "更换配件"):
        ev["equip_count"] += 1

    if tag in ("进入房子", "离开房子", "上载具", "下载具"):
        ev["rotate_count"] += 1

    if tag == "状态更新":
        status = payload.get("status", "")
        if "移动中" in status or "驾驶载具" in status:
            ev["rotate_count"] += 0.5

    if tracker is not None:
        tracker.update(event)
        ev["enemy_visible_count"] = float(tracker.current_enemy_count)
        ev["teammate_visible_count"] = float(tracker.current_teammate_count)
        ev["peak_enemy_visible"] = float(tracker.peak_enemies)
        ev["peak_teammate_visible"] = float(tracker.peak_teammates)
        if tag == "敌人进入视野":
            ev["observe_count"] += 1
    else:
        if tag in ("敌人进入视野",):
            ev["observe_count"] += 1
            ev["enemy_visible_count"] += payload.get("enemy_count", 1)
            ev["teammate_visible_count"] += payload.get("teammate_count", 0)

    if tag == "击倒玩家":
        ev["knockdown_count"] += 1

    if tag == "淘汰玩家":
        ev["eliminate_count"] += 1

    if tag in ("上载具", "下载具"):
        ev["vehicle_count"] += 1

def compute_dominant_state(bin_item: Dict) -> str:
    states = bin_item.get("states", set())
    if not states:
        return "Unknown"

    state_counts = defaultdict(int)
    for s in states:
        state_counts[s] += 1

    if not state_counts:
        return "Unknown"

    return max(state_counts, key=state_counts.get)
