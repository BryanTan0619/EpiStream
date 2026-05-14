
import math
from collections import defaultdict
from typing import Any, List, Dict, Tuple

from map import init_evidence_vector

ALL_STATES = ["Looting", "Combat", "Rotate", "Recover", "Observe", "Parachute", "Unknown"]

def per_bin_state_distribution(bin_item: Dict) -> Dict[str, float]:
    states = list(bin_item["states"])
    dist = {s: 0.0 for s in ALL_STATES}

    if not states:
        dist["Unknown"] = 1.0
        return dist

    w = 1.0 / len(states)
    for s in states:
        if s in dist:
            dist[s] += w
    return dist

def window_state_distribution(window_bins: List[Dict]) -> Dict[str, float]:
    agg = defaultdict(float)
    T = len(window_bins)
    if T == 0:
        return {s: 1.0 / len(ALL_STATES) for s in ALL_STATES}

    for b in window_bins:
        pt = per_bin_state_distribution(b)
        for s in ALL_STATES:
            agg[s] += pt[s]

    for s in ALL_STATES:
        agg[s] /= T

    return dict(agg)

def state_entropy_bin_based(window_bins: List[Dict]) -> float:
    pw = window_state_distribution(window_bins)
    K = len(ALL_STATES)
    H = 0.0
    for s in ALL_STATES:
        p = pw[s]
        if p > 0:
            H -= p * math.log(p + 1e-12)
    max_H = math.log(K + 1e-12)
    return H / max_H if max_H > 0 else 0.0

def state_purity_bin_based(window_bins: List[Dict]) -> float:
    pw = window_state_distribution(window_bins)
    return max(pw.values()) if pw else 0.0

def _dominant_state_sequence(window_bins: List[Dict]) -> List[str]:
    dom_list = []
    for b in window_bins:
        pw = per_bin_state_distribution(b)
        s = max(pw, key=pw.get)
        dom_list.append(s)
    return dom_list

def _compute_runs(dom_sequence: List[str]) -> List[Tuple[str, int]]:
    if not dom_sequence:
        return []

    runs = []
    cur_state = dom_sequence[0]
    cur_len = 1
    for i in range(1, len(dom_sequence)):
        if dom_sequence[i] == cur_state:
            cur_len += 1
        else:
            runs.append((cur_state, cur_len))
            cur_state = dom_sequence[i]
            cur_len = 1
    runs.append((cur_state, cur_len))
    return runs

def run_state_distribution(window_bins: List[Dict]) -> Dict[str, float]:
    dist = {s: 0.0 for s in ALL_STATES}

    if not window_bins:
        for s in ALL_STATES:
            dist[s] = 1.0 / len(ALL_STATES)
        return dist

    dom_seq = _dominant_state_sequence(window_bins)
    runs = _compute_runs(dom_seq)
    total_runs = len(runs)

    if total_runs == 0:
        for s in ALL_STATES:
            dist[s] = 1.0 / len(ALL_STATES)
        return dist

    for state, _ in runs:
        if state in dist:
            dist[state] += 1.0

    for s in ALL_STATES:
        dist[s] /= total_runs

    return dist

def state_entropy(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0

    q = run_state_distribution(window_bins)
    K = len(ALL_STATES)
    H = 0.0
    for s in ALL_STATES:
        p = q[s]
        if p > 0:
            H -= p * math.log(p + 1e-12)
    max_H = math.log(K + 1e-12)
    return H / max_H if max_H > 0 else 0.0

def state_purity(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0

    q = run_state_distribution(window_bins)
    return max(q.values()) if q else 0.0

state_entropy_run_based = state_entropy
state_purity_run_based = state_purity

def state_transition_rate(window_bins: List[Dict]) -> float:
    if len(window_bins) <= 1:
        return 0.0

    dom_seq = _dominant_state_sequence(window_bins)
    runs = _compute_runs(dom_seq)
    num_runs = len(runs)
    K = len(ALL_STATES)

    return min(num_runs - 1, K) / K

def longest_state_run_ratio(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0

    dom_seq = _dominant_state_sequence(window_bins)
    runs = _compute_runs(dom_seq)
    if not runs:
        return 0.0

    max_run_len = max(length for _, length in runs)
    return max_run_len / len(window_bins)

def window_consistency(window_bins: List[Dict],
                       w1: float = 1/3, w2: float = 1/6,
                       w3: float = 1/6, w4: float = 1/3,
                       **kwargs) -> float:
    if not window_bins:
        return 0.0

    H  = state_entropy(window_bins)
    P  = state_purity(window_bins)
    TR = state_transition_rate(window_bins)
    MN = margin_norm(window_bins)

    return w1 * (1 - H) + w2 * P + w3 * (1 - TR) + w4 * MN

def window_consistency_length_aware(
    window_bins: List[Dict],
    tau: float = 9.0,
    w1: float = 1/3, w2: float = 1/6,
    w3: float = 1/6, w4: float = 1/3,
) -> float:
    if not window_bins:
        return 0.0

    C = window_consistency(window_bins, w1, w2, w3, w4)
    T = len(window_bins)

    dom_seq = _dominant_state_sequence(window_bins)
    R = len(_compute_runs(dom_seq))

    if R >= 2:
        return C

    conf = 1.0 - math.exp(-T / tau)
    return C * conf

def aggregate_evidence(window_bins: List[Dict]) -> Dict[str, float]:
    ev = init_evidence_vector()
    for b in window_bins:
        for k in ev:
            ev[k] += b["evidence"][k]
    return ev

def state_divergence(W1: List[Dict], W2: List[Dict]) -> float:
    p1 = window_state_distribution(W1)
    p2 = window_state_distribution(W2)
    return 0.5 * sum(abs(p1.get(s, 0) - p2.get(s, 0)) for s in ALL_STATES)

def evidence_separability(W1_bins: List[Dict], W2_bins: List[Dict]) -> float:
    e1 = aggregate_evidence(W1_bins)
    e2 = aggregate_evidence(W2_bins)

    len1 = max(len(W1_bins), 1)
    len2 = max(len(W2_bins), 1)

    dist = 0.0
    for k in e1:
        dist += abs(e1[k] / len1 - e2[k] / len2)

    return dist

def inter_window_separability(W1: List[Dict], W2: List[Dict],
                              l1: float = 0.5, l2: float = 0.5) -> float:
    d_state = state_divergence(W1, W2)
    d_evidence = evidence_separability(W1, W2)
    return l1 * d_state + l2 * d_evidence

def short_window_penalty(window_bins: List[Dict], bin_size: float,
                         min_window_secs: float) -> float:
    duration = len(window_bins) * bin_size
    if duration < min_window_secs:
        return (min_window_secs - duration) / min_window_secs
    return 0.0

def long_window_penalty(window_bins: List[Dict], bin_size: float,
                        max_window_secs: float) -> float:
    duration = len(window_bins) * bin_size
    if duration > max_window_secs:
        return min((duration - max_window_secs) / max_window_secs, 1.0)
    return 0.0

def overlap_penalty(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0
    multi_state_bins = sum(1 for b in window_bins if len(b["states"]) > 1)
    return multi_state_bins / max(len(window_bins), 1)

def evaluate_cut_gain(bins: List[Dict], cur_start: int, cut_idx: int,
                      bin_size: float, *,
                      alpha: float = 0.4, beta: float = 0.4,
                      gamma: float = 0.2, delta: float = 0.2,
                      w1: float = 0.25, w2: float = 0.25,
                      w3: float = 0.25, w4: float = 0.25,
                      l1: float = 0.5, l2: float = 0.5,
                      lookahead_bins: int = 15,
                      next_boundary_idx: int = -1,
                      min_window_secs: float = 5.0,
                      max_window_secs: float = 180.0) -> float:
    W_left = bins[cur_start:cut_idx]

    if next_boundary_idx > cut_idx:
        dynamic_la = next_boundary_idx - cut_idx
        dynamic_la = max(dynamic_la, max(int(min_window_secs / bin_size), 2))
        dynamic_la = min(dynamic_la, lookahead_bins * 3)
        horizon = min(len(bins), cut_idx + dynamic_la)
    else:
        horizon = min(len(bins), cut_idx + lookahead_bins)
    W_right = bins[cut_idx:horizon]

    min_bins = max(int(min_window_secs / bin_size), 1)

    if len(W_left) < min_bins:
        return -1e9

    if len(W_right) < 2:
        return -1e9

    left_c = window_consistency(W_left, w1, w2, w3, w4)
    right_c = window_consistency(W_right, w1, w2, w3, w4)
    sep = inter_window_separability(W_left, W_right, l1, l2)

    penalty = 0.0
    penalty += short_window_penalty(W_left, bin_size, min_window_secs)
    penalty += long_window_penalty(W_left, bin_size, max_window_secs)
    penalty += 0.5 * overlap_penalty(W_left)
    penalty += 0.5 * overlap_penalty(W_right)

    utility_cut = alpha * left_c + beta * right_c + gamma * sep - delta * penalty

    W_long = bins[cur_start:horizon]
    nocut_c = window_consistency(W_long, w1, w2, w3, w4)

    penalty_nocut = 0.0
    penalty_nocut += long_window_penalty(W_long, bin_size, max_window_secs)
    penalty_nocut += 0.5 * overlap_penalty(W_long)
    utility_nocut = nocut_c - delta * penalty_nocut

    return utility_cut - utility_nocut

def boundary_utility_single(W_left: List[Dict], W_right: List[Dict],
                            bin_size: float = 1.0, *,
                            alpha: float = 0.4, beta: float = 0.4,
                            gamma: float = 0.2, delta: float = 0.1,
                            w1: float = 1/3, w2: float = 1/6,
                            w3: float = 1/6, w4: float = 1/3,
                            l1: float = 0.5, l2: float = 0.5,
                            min_window_secs: float = 5.0,
                            max_window_secs: float = 180.0) -> float:
    if not W_left or not W_right:
        return 0.0

    left_c = window_consistency(W_left, w1, w2, w3, w4)
    right_c = window_consistency(W_right, w1, w2, w3, w4)
    sep = inter_window_separability(W_left, W_right, l1, l2)

    penalty_cut = 0.0
    penalty_cut += short_window_penalty(W_left, bin_size, min_window_secs)
    penalty_cut += short_window_penalty(W_right, bin_size, min_window_secs)
    penalty_cut += long_window_penalty(W_left, bin_size, max_window_secs)
    penalty_cut += long_window_penalty(W_right, bin_size, max_window_secs)
    penalty_cut += 0.3 * overlap_penalty(W_left)
    penalty_cut += 0.3 * overlap_penalty(W_right)

    U_cut = alpha * left_c + beta * right_c + gamma * sep - delta * penalty_cut

    W_merged = W_left + W_right
    merged_c = window_consistency(W_merged, w1, w2, w3, w4)

    penalty_merged = 0.0
    penalty_merged += short_window_penalty(W_merged, bin_size, min_window_secs)
    penalty_merged += long_window_penalty(W_merged, bin_size, max_window_secs)
    penalty_merged += 0.3 * overlap_penalty(W_merged)

    U_merged = merged_c - delta * penalty_merged

    return U_cut - U_merged

def compute_mbu(window_bins_list: List[List[Dict]],
                bin_size: float = 1.0, **kwargs) -> Dict:
    n = len(window_bins_list)
    if n <= 1:
        return {
            "per_boundary": [],
            "avg_mbu": 0.0,
            "positive_ratio": 0.0,
            "median_mbu": 0.0,
            "min_mbu": 0.0,
            "max_mbu": 0.0,
        }

    per_boundary = []
    for i in range(n - 1):
        W_left = window_bins_list[i]
        W_right = window_bins_list[i + 1]

        mbu = boundary_utility_single(W_left, W_right, bin_size, **kwargs)

        left_dur = (W_left[-1]["end"] - W_left[0]["start"]) if W_left else 0.0
        right_dur = (W_right[-1]["end"] - W_right[0]["start"]) if W_right else 0.0

        per_boundary.append({
            "boundary_idx": i,
            "boundary_time": W_left[-1]["end"] if W_left else 0.0,
            "mbu": mbu,
            "left_duration": round(left_dur, 2),
            "right_duration": round(right_dur, 2),
        })

    mbu_values = [pb["mbu"] for pb in per_boundary]
    sorted_mbu = sorted(mbu_values)
    num_boundaries = len(mbu_values)
    mid = num_boundaries // 2
    median_mbu = (sorted_mbu[mid] if num_boundaries % 2 == 1
                  else (sorted_mbu[mid - 1] + sorted_mbu[mid]) / 2.0)

    return {
        "per_boundary": per_boundary,
        "avg_mbu": sum(mbu_values) / num_boundaries,
        "positive_ratio": sum(1 for v in mbu_values if v > 0) / num_boundaries,
        "median_mbu": median_mbu,
        "min_mbu": min(mbu_values),
        "max_mbu": max(mbu_values),
    }

_EPS = 1e-12

def _sorted_scores(window_bins: List[Dict]) -> List[float]:
    q = run_state_distribution(window_bins)
    scores = [q.get(s, 0.0) for s in ALL_STATES]
    scores.sort(reverse=True)
    return scores

def margin_norm(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0
    scores = _sorted_scores(window_bins)
    s1 = scores[0]
    s2 = scores[1] if len(scores) > 1 else 0.0
    return (s1 - s2) / (s1 + _EPS)

def conf_ratio(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0
    scores = _sorted_scores(window_bins)
    total = sum(scores)
    return scores[0] / (total + _EPS)

def ent_clarity(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0
    q = run_state_distribution(window_bins)
    scores = [q.get(s, 0.0) for s in ALL_STATES]
    total = sum(scores) + _EPS

    K = len(ALL_STATES)
    H = 0.0
    for sc in scores:
        p = sc / total
        if p > 0:
            H -= p * math.log(p + _EPS)
    max_H = math.log(K + _EPS)
    return 1.0 - H / max_H if max_H > 0 else 0.0

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def local_margin_norm(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.5

    q = run_state_distribution(window_bins)
    y_star = max(q, key=q.get)

    dom_seq = _dominant_state_sequence(window_bins)
    runs = _compute_runs(dom_seq)

    if not runs:
        return 0.5

    total = 0.0
    bin_offset = 0
    for run_state, run_len in runs:
        run_bins = window_bins[bin_offset:bin_offset + run_len]
        bin_offset += run_len

        run_dist = {s: 0.0 for s in ALL_STATES}
        for b in run_bins:
            pt = per_bin_state_distribution(b)
            for s in ALL_STATES:
                run_dist[s] += pt.get(s, 0.0)
        for s in ALL_STATES:
            run_dist[s] /= max(len(run_bins), 1)

        score_y = run_dist.get(y_star, 0.0)
        max_other = max(
            (run_dist.get(s, 0.0) for s in ALL_STATES if s != y_star),
            default=0.0,
        )
        margin = score_y - max_other
        total += _sigmoid(margin)

    return total / len(runs)

def support_rate(window_bins: List[Dict]) -> float:
    if not window_bins:
        return 0.0

    q = run_state_distribution(window_bins)
    y_star = max(q, key=q.get)

    dom_seq = _dominant_state_sequence(window_bins)
    runs = _compute_runs(dom_seq)

    if not runs:
        return 0.0

    supported = sum(1 for state, _ in runs if state == y_star)
    return supported / len(runs)

def tag_clarity(window_bins: List[Dict],
                b1: float = 1/3, b2: float = 1/3,
                b3: float = 1/3, b4: float = 0.0) -> float:
    if not window_bins:
        return 0.0

    mn = margin_norm(window_bins)
    cr = conf_ratio(window_bins)
    ec = ent_clarity(window_bins)
    sr = support_rate(window_bins)

    return b1 * mn + b2 * cr + b3 * ec + b4 * sr

def compute_tag_clarity_stats(window_bins_list: List[List[Dict]]) -> Dict[str, Any]:
    per_window = []
    for i, wb in enumerate(window_bins_list):
        mn = margin_norm(wb)
        cr = conf_ratio(wb)
        ec = ent_clarity(wb)
        lm = local_margin_norm(wb)
        sr = support_rate(wb)
        tc = tag_clarity(wb)
        per_window.append({
            "window_id": i,
            "margin_norm": round(mn, 4),
            "conf_ratio": round(cr, 4),
            "ent_clarity": round(ec, 4),
            "local_margin": round(lm, 4),
            "support_rate": round(sr, 4),
            "tag_clarity": round(tc, 4),
        })

    n = len(per_window)
    if n == 0:
        return {
            "per_window": [],
            "avg_margin_norm": 0.0,
            "avg_conf_ratio": 0.0,
            "avg_ent_clarity": 0.0,
            "avg_local_margin": 0.0,
            "avg_support_rate": 0.0,
            "avg_tag_clarity": 0.0,
        }

    return {
        "per_window": per_window,
        "avg_margin_norm": round(sum(pw["margin_norm"] for pw in per_window) / n, 4),
        "avg_conf_ratio": round(sum(pw["conf_ratio"] for pw in per_window) / n, 4),
        "avg_ent_clarity": round(sum(pw["ent_clarity"] for pw in per_window) / n, 4),
        "avg_local_margin": round(sum(pw["local_margin"] for pw in per_window) / n, 4),
        "avg_support_rate": round(sum(pw["support_rate"] for pw in per_window) / n, 4),
        "avg_tag_clarity": round(sum(pw["tag_clarity"] for pw in per_window) / n, 4),
    }
