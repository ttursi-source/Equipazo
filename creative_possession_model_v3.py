#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
More than a Hack 2027 — Phase 1: Ball Possession (FC Barcelona vs Athletic Club)
Architecture V3 (Hybrid Fusion):
  - Combines V2's high-accuracy frame emissions (Causal Contact-Chain + Boundary/Tempo Dead-Ball
    detector, which achieved macro_f1 = 0.6613)
  - With V1's high-inertia Semi-Markov Viterbi & Spell Consolidation (which achieved 0 spell count
    error and top spell_f1_mean = 0.4471), PLUS a Boundary Refinement pass that snaps spell
    start/end transitions to exact physical ball-contact or boundary-exit frames.

Usage (Windows):
    python creative_possession_model_v3.py --data-dir ./data --output evaluation/results/creative_v3_submission.csv
"""

import os
import sys
import math
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ==============================================================================
# 1. MATCH CONSTANTS & ALIGNMENT
# ==============================================================================
N_VIDEO_FRAMES = 199319
FPS = 25.0
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

HALF_BLOCKS = [
    (1, 11185, 90651),   # 1st Half: 79,467 frames
    (2, 110808, 190197)  # 2nd Half: 79,390 frames
]

HOME_SLOTS = list(range(0, 24))   # Barcelona ('H'), Track 24 (idx 23) is GK
AWAY_SLOTS = list(range(24, 46))  # Athletic Club ('A'), Track 46 (idx 45) is GK
N_SLOTS = 46


# ==============================================================================
# 2. DATA LOADERS
# ==============================================================================
def load_smart_tagging_xml(xml_path):
    if not os.path.isfile(xml_path):
        print(f"[WARN] Smart Tagging file not found at {xml_path}.")
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []

    for inst in root.findall(".//instance"):
        start_sec = float(inst.findtext("start", "0"))
        end_sec = float(inst.findtext("end", "0"))
        code = (inst.findtext("code") or "").strip().upper()

        labels = {}
        for lbl in inst.findall("label"):
            grp = (lbl.findtext("group") or "").strip()
            txt = (lbl.findtext("text") or "").strip()
            labels[grp] = txt

        start_frame = max(0, min(N_VIDEO_FRAMES - 1, int(round(start_sec * FPS))))
        end_frame = max(0, min(N_VIDEO_FRAMES - 1, int(round(end_sec * FPS))))

        events.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "code": code,
            "team": labels.get("Team", "N/A"),
            "type": labels.get("Type", ""),
            "half": labels.get("Half", ""),
        })
    print(f"[INFO] Loaded {len(events)} Smart Tagging tactical instances.")
    return events


def load_atd_raw_txt(txt_path, n_frames=N_VIDEO_FRAMES):
    player_x = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    player_y = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    player_s = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_s = np.full(n_frames, np.nan, dtype=np.float32)

    print(f"[INFO] Reading raw ATD tracking from {txt_path} ...")
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue

            frame_id = int(parts[0])
            v_frame = frame_id - 1
            if v_frame < 0 or v_frame >= n_frames:
                continue

            p_chunks = parts[1].split(";")
            for s_idx, chunk in enumerate(p_chunks[:N_SLOTS]):
                vals = chunk.split(",")
                if len(vals) >= 2 and vals[0] not in ("nan", "NaN", ""):
                    try:
                        px = float(vals[0])
                        py = float(vals[1])
                        ps = float(vals[2]) if len(vals) > 2 and vals[2] not in ("nan", "NaN", "") else np.nan
                        if not (math.isnan(px) or math.isnan(py)):
                            player_x[v_frame, s_idx] = px * PITCH_LENGTH
                            player_y[v_frame, s_idx] = py * PITCH_WIDTH
                            player_s[v_frame, s_idx] = ps
                    except ValueError:
                        pass

            b_vals = parts[2].split(",")
            if len(b_vals) >= 2 and b_vals[0] not in ("nan", "NaN", ""):
                try:
                    bx = float(b_vals[0])
                    by = float(b_vals[1])
                    bs = float(b_vals[2]) if len(b_vals) > 2 and b_vals[2] not in ("nan", "NaN", "") else np.nan
                    if not (math.isnan(bx) or math.isnan(by)):
                        ball_x[v_frame] = bx * PITCH_LENGTH
                        ball_y[v_frame] = by * PITCH_WIDTH
                        ball_s[v_frame] = bs
                except ValueError:
                    pass

    return player_x, player_y, player_s, ball_x, ball_y, ball_s


# ==============================================================================
# 3. BALL KINEMATICS & BOUNDARY / GAP GEOMETRY
# ==============================================================================
def analyze_ball_and_gaps(ball_x, ball_y, player_x, player_y):
    n = len(ball_x)
    ball_vis = ~np.isnan(ball_x) & ~np.isnan(ball_y)
    idx = np.arange(n)
    vis_idx = idx[ball_vis]

    valid_p = ~np.isnan(player_x)
    p_count = np.sum(valid_p, axis=1)
    safe_px = np.where(valid_p, player_x, 0.0)
    safe_py = np.where(valid_p, player_y, 0.0)
    mean_px = np.where(p_count > 0, np.sum(safe_px, axis=1) / np.maximum(1, p_count), PITCH_LENGTH / 2.0)
    mean_py = np.where(p_count > 0, np.sum(safe_py, axis=1) / np.maximum(1, p_count), PITCH_WIDTH / 2.0)

    bx_interp = np.interp(idx, vis_idx, ball_x[vis_idx]) if len(vis_idx) > 0 else mean_px.copy()
    by_interp = np.interp(idx, vis_idx, ball_y[vis_idx]) if len(vis_idx) > 0 else mean_py.copy()

    prev_vis_idx = np.maximum.accumulate(np.where(ball_vis, idx, 0))
    next_vis_idx = np.minimum.accumulate(np.where(ball_vis, idx, n - 1)[::-1])[::-1]

    gap_len = np.where(ball_vis, 0, next_vis_idx - prev_vis_idx)
    time_since_vis = idx - prev_vis_idx
    time_until_vis = next_vis_idx - idx

    last_vis_x = ball_x[prev_vis_idx]
    last_vis_y = ball_y[prev_vis_idx]
    next_vis_x = ball_x[next_vis_idx]
    next_vis_y = ball_y[next_vis_idx]

    alpha_focal = np.clip((gap_len - 55.0) / 150.0, 0.0, 0.60)
    bx_est = (1.0 - alpha_focal) * bx_interp + alpha_focal * mean_px
    by_est = (1.0 - alpha_focal) * by_interp + alpha_focal * mean_py

    k3 = np.array([0.25, 0.50, 0.25], dtype=np.float32)
    bx_s = np.convolve(bx_est, k3, mode="same")
    by_s = np.convolve(by_est, k3, mode="same")
    vx = np.gradient(bx_s) * FPS
    vy = np.gradient(by_s) * FPS
    ball_vel = np.hypot(vx, vy)

    ax = np.gradient(vx) * FPS
    ay = np.gradient(vy) * FPS
    ball_acc = np.hypot(ax, ay)

    return (bx_est, by_est, ball_vis, gap_len, time_since_vis, time_until_vis,
            last_vis_x, last_vis_y, next_vis_x, next_vis_y, ball_vel, ball_acc)


# ==============================================================================
# 4. CALIBRATED DEAD-BALL ('D') DETECTOR (V3 HYBRID)
# ==============================================================================
def compute_dead_ball_probability_v3(
    player_x, player_y, player_s, ball_x, ball_y,
    bx_est, by_est, ball_vis, gap_len, time_since_vis, time_until_vis,
    last_vis_x, last_vis_y, next_vis_x, next_vis_y, ball_vel,
    min_dist_H, min_dist_A, events
):
    n = len(ball_x)
    graded_mask = np.zeros(n, dtype=bool)
    for _, f_start, f_end in HALF_BLOCKS:
        graded_mask[f_start:f_end + 1] = True

    valid_s = ~np.isnan(player_s)
    s_count = np.sum(valid_s, axis=1)
    safe_s = np.where(valid_s, player_s, 0.0)
    mean_s = np.where(s_count > 0, np.sum(safe_s, axis=1) / np.maximum(1, s_count), 0.0)

    med_s = np.median(mean_s[graded_mask])
    iqr_s = max(1e-3, np.percentile(mean_s[graded_mask], 75) - np.percentile(mean_s[graded_mask], 25))
    z_tempo = (mean_s - med_s) / iqr_s

    k40 = np.ones(40, dtype=np.float32) / 40.0
    k75 = np.ones(75, dtype=np.float32) / 75.0
    tempo_fast = np.convolve(z_tempo, k40, mode="same")
    tempo_slow = np.convolve(z_tempo, k75, mode="same")

    def is_restart_zone(x, y):
        near_sideline = (y < 2.2) | (y > PITCH_WIDTH - 2.2)
        near_goalline = (x < 2.2) | (x > PITCH_LENGTH - 2.2)
        in_goal_box = ((x < 7.5) | (x > PITCH_LENGTH - 7.5)) & (y > 22.5) & (y < 45.5)
        return near_sideline | near_goalline | in_goal_box

    strictly_out = ball_vis & (
        (ball_x < 0.35) | (ball_x > PITCH_LENGTH - 0.35) |
        (ball_y < 0.35) | (ball_y > PITCH_WIDTH - 0.35)
    )
    vis_near_edge = ball_vis & is_restart_zone(ball_x, ball_y)
    exit_near_edge = (~ball_vis) & is_restart_zone(last_vis_x, last_vis_y)
    entry_near_edge = (~ball_vis) & is_restart_zone(next_vis_x, next_vis_y)

    # Stricter than V2 so brief aerial balls near the sideline don't split a live possession!
    boundary_gap = (gap_len >= 38) & (exit_near_edge | entry_near_edge) & (tempo_fast < 0.15)
    midfield_stoppage_gap = (gap_len >= 65) & (tempo_slow < -0.20)

    slow_ball = ball_vis & (ball_vel < 2.8)
    k25 = np.ones(25, dtype=np.float32) / 25.0
    slow_ball_density = np.convolve(slow_ball.astype(np.float32), k25, mode="same")
    waiting_for_restart = (slow_ball_density > 0.70) & (vis_near_edge | (tempo_fast < -0.25))

    st_dead_prior = np.zeros(n, dtype=np.float32)
    st_restart_team = np.zeros(n, dtype=np.int32)

    for ev in events:
        s, e = ev["start_frame"], ev["end_frame"]
        code = ev["code"]
        sp_type = ev["type"].upper()
        team = ev["team"].upper()
        is_barca = "BARCELONA" in team
        is_ath = "ATLETIC" in team or "ATHLETIC" in team

        if code == "GOALS":
            search_end = min(e, s + int(12 * FPS))
            goal_frame = s + int(5 * FPS)
            for f_idx in range(s, search_end):
                if not ball_vis[f_idx] or ball_x[f_idx] < 1.5 or ball_x[f_idx] > PITCH_LENGTH - 1.5:
                    goal_frame = f_idx
                    break
            st_dead_prior[goal_frame:e + 1] += 5.0

        elif code == "SET PIECES":
            if "KICK OFF" in sp_type:
                pre_ko = max(0, s - int(15 * FPS))
                st_dead_prior[pre_ko:s + int(0.8 * FPS)] += 4.0
                continue

            w_s = min(e, s + int(2.5 * FPS))
            w_e = max(w_s + 1, e - int(3.5 * FPS))
            if w_e > w_s and w_e <= n:
                active_ball = (ball_vis & (ball_vel > 5.8) & (~strictly_out)).astype(np.float32)
                best_kick = w_s + int(0.45 * (w_e - w_s))
                best_score = -1e9
                for cand in range(w_s, w_e, 3):
                    pre_sl = slice(max(s, cand - 50), cand)
                    post_sl = slice(cand, min(e, cand + 35))
                    pre_deadness = 1.0 - np.mean(active_ball[pre_sl]) - 0.25 * np.mean(tempo_fast[pre_sl])
                    post_liveness = np.mean(active_ball[post_sl]) + 0.15 * np.max(ball_vel[post_sl]) / 15.0
                    score = pre_deadness + post_liveness
                    if score > best_score:
                        best_score = score
                        best_kick = cand

                f_stop = s
                for back_f in range(best_kick - 5, max(0, s - int(2 * FPS)), -1):
                    if ball_vis[back_f] and ball_vel[back_f] > 8.5 and not is_restart_zone(ball_x[back_f], ball_y[back_f]):
                        f_stop = back_f + 1
                        break
                else:
                    f_stop = max(0, s)

                st_dead_prior[f_stop:best_kick] += 2.7

                post_end = min(n, best_kick + int(3.8 * FPS))
                if is_barca:
                    st_restart_team[best_kick:post_end] = 1
                elif is_ath:
                    st_restart_team[best_kick:post_end] = -1
                else:
                    chk = slice(max(0, best_kick - 18), min(n, best_kick + 15))
                    if np.nanmedian(min_dist_H[chk]) <= np.nanmedian(min_dist_A[chk]):
                        st_restart_team[best_kick:post_end] = 1
                    else:
                        st_restart_team[best_kick:post_end] = -1

    fast_live_ball = ball_vis & (~strictly_out) & (ball_vel > 6.0) & (tempo_fast > -0.35)

    logit_D = (
        -1.08  # Balanced prior between V1 (-1.15) and V2 (-0.95)
        + st_dead_prior
        + 2.40 * strictly_out.astype(np.float32)
        + 1.35 * boundary_gap.astype(np.float32)
        + 1.25 * midfield_stoppage_gap.astype(np.float32)
        + 1.20 * waiting_for_restart.astype(np.float32)
        + 0.85 * np.clip(-tempo_slow - 0.25, -0.7, 2.0)
        + 0.65 * np.clip((gap_len - 55.0) / 55.0, 0.0, 1.5) * np.where(tempo_fast < 0.0, 1.0, 0.25)
        - 2.15 * fast_live_ball.astype(np.float32)
    )

    return logit_D, st_restart_team, tempo_fast


# ==============================================================================
# 5. CAUSAL CONTACT-CHAIN OWNERSHIP & GAUSSIAN PITCH CONTROL (V3 HYBRID)
# ==============================================================================
def compute_team_ownership_potentials_v3(
    player_x, player_y, bx_est, by_est, ball_vis, gap_len, ball_vel, ball_acc,
    events, st_restart_team
):
    n = len(bx_est)

    dx_H = player_x[:, HOME_SLOTS] - bx_est[:, None]
    dy_H = player_y[:, HOME_SLOTS] - by_est[:, None]
    dist_H_all = np.hypot(dx_H, dy_H)

    dx_A = player_x[:, AWAY_SLOTS] - bx_est[:, None]
    dy_A = player_y[:, AWAY_SLOTS] - by_est[:, None]
    dist_A_all = np.hypot(dx_A, dy_A)

    min_dist_H = np.nanmin(np.where(np.isnan(dist_H_all), 999.0, dist_H_all), axis=1)
    min_dist_A = np.nanmin(np.where(np.isnan(dist_A_all), 999.0, dist_A_all), axis=1)

    sorted_H = np.sort(np.where(np.isnan(dist_H_all), 999.0, dist_H_all), axis=1)
    sorted_A = np.sort(np.where(np.isnan(dist_A_all), 999.0, dist_A_all), axis=1)
    sec_dist_H = sorted_H[:, 1]
    sec_dist_A = sorted_A[:, 1]

    ctrl_close_H = np.nansum(np.exp(-0.5 * (dist_H_all / 1.9) ** 2), axis=1)
    ctrl_close_A = np.nansum(np.exp(-0.5 * (dist_A_all / 1.9) ** 2), axis=1)
    ctrl_supp_H = np.nansum(np.exp(-0.5 * (dist_H_all / 6.0) ** 2), axis=1)
    ctrl_supp_A = np.nansum(np.exp(-0.5 * (dist_A_all / 6.0) ** 2), axis=1)

    touch_H = ball_vis & (
        ((min_dist_H < 1.85) & (min_dist_A - min_dist_H > 0.60)) |
        ((min_dist_H < 1.25) & (min_dist_A - min_dist_H > 0.30))
    )
    touch_A = ball_vis & (
        ((min_dist_A < 1.85) & (min_dist_H - min_dist_A > 0.60)) |
        ((min_dist_A < 1.25) & (min_dist_H - min_dist_A > 0.30))
    )

    k4 = np.ones(4, dtype=np.float32)
    sust_H = np.convolve(touch_H.astype(np.float32), k4, mode="same") >= 2.0
    sust_A = np.convolve(touch_A.astype(np.float32), k4, mode="same") >= 2.0

    anchor_team = np.zeros(n, dtype=np.int32)
    anchor_team[sust_H & (~sust_A)] = 1
    anchor_team[sust_A & (~sust_H)] = -1
    anchor_team[st_restart_team != 0] = st_restart_team[st_restart_team != 0]

    chain_signal = np.zeros(n, dtype=np.float32)
    anchor_indices = np.nonzero(anchor_team != 0)[0]

    if len(anchor_indices) > 0:
        cleaned_anchors = anchor_team.copy()
        runs = []
        cur_t = anchor_team[anchor_indices[0]]
        run_s = anchor_indices[0]
        run_e = anchor_indices[0]
        for idx_a in anchor_indices[1:]:
            t_val = anchor_team[idx_a]
            if t_val == cur_t and (idx_a - run_e) <= 35:
                run_e = idx_a
            else:
                runs.append([cur_t, run_s, run_e])
                cur_t = t_val
                run_s = idx_a
                run_e = idx_a
        runs.append([cur_t, run_s, run_e])

        for r_i in range(1, len(runs) - 1):
            t_val, r_s, r_e = runs[r_i]
            prev_t, prev_s, prev_e = runs[r_i - 1]
            next_t, next_s, next_e = runs[r_i + 1]
            if prev_t == next_t and prev_t != t_val and (r_e - r_s) < 12 and (next_s - prev_e) < 100:
                cleaned_anchors[r_s:r_e + 1] = 0

        valid_idx = np.nonzero(cleaned_anchors != 0)[0]
        for k_i in range(len(valid_idx) - 1):
            f1 = valid_idx[k_i]
            f2 = valid_idx[k_i + 1]
            t1 = cleaned_anchors[f1]
            t2 = cleaned_anchors[f2]
            span = f2 - f1
            if t1 == t2 and span <= 160:
                chain_signal[f1:f2 + 1] = float(t1)
            elif t1 != t2 and span <= 125:
                hold_end = max(f1, f2 - 5)
                chain_signal[f1:hold_end] = 0.80 * float(t1)
                chain_signal[hold_end:f2] = 0.25 * float(t1)

    prior_H = np.zeros(n, dtype=np.float32)
    prior_A = np.zeros(n, dtype=np.float32)
    OFFENSIVE_CODES = {
        "BUILD UP": 1.3, "PROGRESSION": 1.5, "CREATING CHANCES": 1.1,
        "BALL IN FINAL THIRD": 1.1, "BALL IN THE BOX": 1.1,
        "PLAYERS IN THE BOX": 0.9, "ATTACKING TRANSITION": 1.4, "LONG BALL": 1.0,
    }
    DEFENSIVE_CODES = {
        "DEFENDING IN ATTACKING THIRD": 1.3, "DEFENDING IN MIDDLE THIRD": 1.4,
        "DEFENDING IN DEFENSIVE THIRD": 1.1, "DEFENSIVE TRANSITION": 1.3,
    }
    for ev in events:
        s, e = ev["start_frame"], ev["end_frame"]
        code = ev["code"]
        team = ev["team"].upper()
        is_barca = "BARCELONA" in team
        is_ath = "ATLETIC" in team or "ATHLETIC" in team
        if code in OFFENSIVE_CODES:
            w = OFFENSIVE_CODES[code]
            if is_barca:
                prior_H[s:e + 1] += w
            elif is_ath:
                prior_A[s:e + 1] += w
        elif code in DEFENSIVE_CODES:
            w = DEFENSIVE_CODES[code]
            if is_barca:
                prior_A[s:e + 1] += w
            elif is_ath:
                prior_H[s:e + 1] += w

    st_diff = 1.45 * np.tanh((prior_H - prior_A) / 2.5)

    vis_weight = np.exp(-gap_len / 25.0)
    dist_diff = np.clip(min_dist_A - min_dist_H, -14.0, 14.0)
    sec_diff = np.clip(sec_dist_A - sec_dist_H, -14.0, 14.0)

    diff_HA = (
        1.15 * chain_signal +
        0.50 * vis_weight * dist_diff +
        0.92 * vis_weight * (ctrl_close_H - ctrl_close_A) +
        0.28 * (ctrl_supp_H - ctrl_supp_A) +
        0.10 * sec_diff +
        (0.65 + 0.35 * (1.0 - vis_weight)) * st_diff +
        1.85 * st_restart_team.astype(np.float32)
    )

    return diff_HA, min_dist_H, min_dist_A, ctrl_close_H, ctrl_close_A, anchor_team


# ==============================================================================
# 6. HIGH-INERTIA VITERBI + BOUNDARY SNAP REFINEMENT (FOR SPELL-F1 & MACRO-F1)
# ==============================================================================
def get_runs(arr):
    if len(arr) == 0:
        return []
    diffs = np.diff(arr)
    change_idx = np.nonzero(diffs != 0)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(arr)]))
    return [[int(arr[s]), int(s), int(e), int(e - s)] for s, e in zip(starts, ends)]


def snap_transitions_to_contacts(states, anchor_team_half, radius=12):
    """
    Once spell count and macro-structure are locked in by Viterbi + Consolidation,
    this pass refines H <-> A turnover boundaries within +/- `radius` frames (0.48s)
    so the transition lands on the exact first frame of physical contact by the new team!
    """
    out = states.copy()
    runs = get_runs(out)
    for i in range(len(runs) - 1):
        st_curr, s1, e1, len1 = runs[i]
        st_next, s2, e2, len2 = runs[i + 1]
        # Direct turnover between H (0) and A (1)
        if st_curr in (0, 1) and st_next in (0, 1) and st_curr != st_next:
            boundary = e1  # == s2
            target_anchor = 1 if st_next == 0 else -1
            w_s = max(s1 + 10, boundary - radius)
            w_e = min(e2 - 10, boundary + radius)
            if w_e > w_s:
                cand_idx = np.nonzero(anchor_team_half[w_s:w_e] == target_anchor)[0]
                if len(cand_idx) > 0:
                    new_b = w_s + int(cand_idx[0])
                    if new_b < boundary:
                        out[new_b:boundary] = st_next
                    elif new_b > boundary:
                        out[boundary:new_b] = st_curr
    return out


def decode_possession_v3(diff_HA, logit_D, ctrl_H, ctrl_A, anchor_team):
    n = len(diff_HA)
    score_H = 0.5 * diff_HA
    score_A = -0.5 * diff_HA
    score_D = logit_D

    logits = np.stack([score_H, score_A, score_D], axis=1)
    logits -= np.max(logits, axis=1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))

    labels = np.full(n, "X", dtype="<U1")
    state_map = np.array(["H", "A", "D"], dtype="<U1")

    for half_id, f_start, f_end in HALF_BLOCKS:
        sl = slice(f_start, f_end + 1)
        lp = log_probs[sl]
        T, K = lp.shape

        # Use V1's proven high-inertia transition matrix so we preserve exact spell counts!
        p_stay_team = 0.992
        p_stay_dead = 0.9935
        rem_t = 1.0 - p_stay_team
        rem_d = 1.0 - p_stay_dead
        trans = np.array([
            [p_stay_team, 0.68 * rem_t, 0.32 * rem_t],
            [0.68 * rem_t, p_stay_team, 0.32 * rem_t],
            [0.50 * rem_d, 0.50 * rem_d, p_stay_dead]
        ], dtype=np.float64)
        log_trans = np.log(trans)

        dp = np.zeros((T, K), dtype=np.float64)
        backtrack = np.zeros((T, K), dtype=np.int32)
        dp[0] = lp[0]

        obs_scale = 0.185
        scaled_obs = obs_scale * lp.astype(np.float64)

        for t in range(1, T):
            scores = dp[t - 1, :, None] + log_trans
            best_prev = np.argmax(scores, axis=0)
            dp[t] = scores[best_prev, np.arange(K)] + scaled_obs[t]
            backtrack[t] = best_prev

        states = np.zeros(T, dtype=np.int32)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = backtrack[t + 1, states[t + 1]]

        # Use V1's proven consolidation thresholds (min_dead=75, min_pos=45)
        states = consolidate_spells_v3(states, ctrl_H[sl], ctrl_A[sl], min_pos_frames=45, min_dead_frames=75)
        # Snap turnover boundaries to exact contact frames
        states = snap_transitions_to_contacts(states, anchor_team[sl], radius=12)
        labels[sl] = state_map[states]

    return labels


def consolidate_spells_v3(states, ctrl_H, ctrl_A, min_pos_frames=45, min_dead_frames=75):
    out = states.copy()

    # Pass 1: Suppress false micro-stoppages ('D' < 75 frames / 3.0s)
    runs = get_runs(out)
    for i, (st, s, e, length) in enumerate(runs):
        if st == 2 and length < min_dead_frames:
            prev_st = runs[i - 1][0] if i > 0 else (runs[i + 1][0] if i + 1 < len(runs) else 0)
            next_st = runs[i + 1][0] if i + 1 < len(runs) else prev_st
            if prev_st == next_st:
                out[s:e] = prev_st
            else:
                mid = (s + e) // 2
                out[s:mid] = prev_st
                out[mid:e] = next_st

    # Pass 2: Absorb deflections / uncontrolled touches (< 45 frames / 1.8s)
    for _ in range(2):
        runs = get_runs(out)
        changed = False
        for i in range(1, len(runs) - 1):
            st, s, e, length = runs[i]
            prev_st = runs[i - 1][0]
            next_st = runs[i + 1][0]
            if st in (0, 1) and length < min_pos_frames:
                team_ctrl = np.mean(ctrl_H[s:e]) if st == 0 else np.mean(ctrl_A[s:e])
                if prev_st == next_st and team_ctrl < 1.38:
                    out[s:e] = prev_st
                    changed = True
                elif length < 25:
                    out[s:e] = prev_st
                    changed = True
        if not changed:
            break

    return out


# ==============================================================================
# 7. MAIN ENTRYPOINT
# ==============================================================================
def generate_submission_v3(data_dir, output_csv):
    tactical_dir = os.path.join(data_dir, "tactical")
    atd_txt = os.path.join(tactical_dir, "BAR-ATH_tactical_atd.txt")
    st_xml = os.path.join(tactical_dir, "BAR-ATH_tactical_smart_tagging.xml")

    if not os.path.isfile(atd_txt):
        alt_txt = os.path.join(data_dir, "BAR-ATH_tactical_atd.txt")
        if os.path.isfile(alt_txt):
            atd_txt = alt_txt
            st_xml = os.path.join(data_dir, "BAR-ATH_tactical_smart_tagging.xml")
        else:
            raise FileNotFoundError(f"Could not find BAR-ATH_tactical_atd.txt in {tactical_dir} or {data_dir}.")

    events = load_smart_tagging_xml(st_xml)
    player_x, player_y, player_s, ball_x, ball_y, ball_s = load_atd_raw_txt(atd_txt, N_VIDEO_FRAMES)

    print("[INFO] Analyzing ball kinematics, pass flights, and boundary gap geometry...")
    (bx_est, by_est, ball_vis, gap_len, time_since_vis, time_until_vis,
     last_vis_x, last_vis_y, next_vis_x, next_vis_y, ball_vel, ball_acc) = analyze_ball_and_gaps(
        ball_x, ball_y, player_x, player_y
    )

    dx_H = player_x[:, HOME_SLOTS] - bx_est[:, None]
    dy_H = player_y[:, HOME_SLOTS] - by_est[:, None]
    min_dist_H = np.nanmin(np.where(np.isnan(dx_H), 999.0, np.hypot(dx_H, dy_H)), axis=1)

    dx_A = player_x[:, AWAY_SLOTS] - bx_est[:, None]
    dy_A = player_y[:, AWAY_SLOTS] - by_est[:, None]
    min_dist_A = np.nanmin(np.where(np.isnan(dx_A), 999.0, np.hypot(dx_A, dy_A)), axis=1)

    print("[INFO] Computing V3 Hybrid Dead-Ball ('D') detector...")
    logit_D, st_restart_team, tempo_fast = compute_dead_ball_probability_v3(
        player_x, player_y, player_s, ball_x, ball_y,
        bx_est, by_est, ball_vis, gap_len, time_since_vis, time_until_vis,
        last_vis_x, last_vis_y, next_vis_x, next_vis_y, ball_vel,
        min_dist_H, min_dist_A, events
    )

    print("[INFO] Computing V3 Causal Contact-Chain ownership and Gaussian Pitch Control...")
    diff_HA, min_dist_H, min_dist_A, ctrl_H, ctrl_A, anchor_team = compute_team_ownership_potentials_v3(
        player_x, player_y, bx_est, by_est, ball_vis, gap_len, ball_vel, ball_acc,
        events, st_restart_team
    )

    print("[INFO] Decoding possession spells with V3 High-Inertia Viterbi + Contact Boundary Snap...")
    labels = decode_possession_v3(diff_HA, logit_D, ctrl_H, ctrl_A, anchor_team)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_sub = pd.DataFrame({
        "frame": np.arange(N_VIDEO_FRAMES, dtype=np.int32),
        "label": labels
    })
    df_sub.to_csv(output_csv, index=False, lineterminator="\n")
    print(f"[SUCCESS] Saved V3 submission to: {output_csv}")

    graded_labels = np.concatenate([labels[s:e+1] for _, s, e in HALF_BLOCKS])
    u, counts = np.unique(graded_labels, return_counts=True)
    stats = dict(zip(u, counts))
    total_g = len(graded_labels)
    print("[STATS] V3 Graded frame distribution:")
    for k in ["H", "A", "D", "X"]:
        c = stats.get(k, 0)
        print(f"    {k}: {c:6d} frames ({100.0 * c / total_g:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creative V3 Hybrid Contact-Chain HSMM Model")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("MTAH_DATA_DIR", "./data"))
    parser.add_argument("--output", type=str, default="evaluation/results/creative_v3_submission.csv")
    args = parser.parse_args()
    generate_submission_v3(args.data_dir, args.output)
