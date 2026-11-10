#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
More than a Hack 2027 — Phase 1: Ball Possession (FC Barcelona vs Athletic Club)
Architecture V2: Contact-Chain Physics + Boundary/Tempo Dead-Ball Detector + Semi-Markov Decoder

Key upgrades over V1 based on diagnostic analysis:
  1. Causal Contact-Chain Ownership: Between confirmed ball touches, possession is carried
     forward from the last controlling team ("In phases where the ball is contested or loose,
     possession stays with the last team that had it" - CHALLENGE.md), eliminating midpoint
     transition latency during passes and long balls.
  2. High-Recall Dead-Ball ('D') Detection: Removes the V1 penalty that suppressed 'D' when a
     set-piece taker stood near the ball. Detects boundary exits, goal-kick/corner waits,
     stationary pre-kick setups, and low-tempo occlusion stretches.
  3. Scoped Smart Tagging Priors: Prevents fixed 30s/35s SportsCode template windows from
     overriding visible player-ball contacts or masking dead-ball stoppages.

Usage (Windows):
    python creative_possession_model_v2.py --data-dir ./data --output evaluation/results/creative_v2_submission.csv
"""

import os
import sys
import math
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ==============================================================================
# 1. MATCH CONSTANTS & ALIGNMENT (FROM games.py & BAR-ATH_tactical_atd.xml)
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
    """
    Computes:
      - Reconstructed ball coordinates (bx_est, by_est)
      - True ball speed when visible (ball_vel_vis) and interpolated speed (ball_vel)
      - Gap metadata: for each missing ball frame, how long is the gap (`gap_len`),
        where did the ball disappear (`last_vis_x, last_vis_y`), and where does it
        reappear (`next_vis_x, next_vis_y`).
    """
    n = len(ball_x)
    ball_vis = ~np.isnan(ball_x) & ~np.isnan(ball_y)
    idx = np.arange(n)
    vis_idx = idx[ball_vis]

    # Collective player centroid (avoiding empty-slice warnings on ungraded frames)
    valid_p = ~np.isnan(player_x)
    p_count = np.sum(valid_p, axis=1)
    safe_px = np.where(valid_p, player_x, 0.0)
    safe_py = np.where(valid_p, player_y, 0.0)
    mean_px = np.where(p_count > 0, np.sum(safe_px, axis=1) / np.maximum(1, p_count), PITCH_LENGTH / 2.0)
    mean_py = np.where(p_count > 0, np.sum(safe_py, axis=1) / np.maximum(1, p_count), PITCH_WIDTH / 2.0)

    bx_interp = np.interp(idx, vis_idx, ball_x[vis_idx]) if len(vis_idx) > 0 else mean_px.copy()
    by_interp = np.interp(idx, vis_idx, ball_y[vis_idx]) if len(vis_idx) > 0 else mean_py.copy()

    # Anchor indices for every frame: previous visible frame and next visible frame
    prev_vis_idx = np.maximum.accumulate(np.where(ball_vis, idx, 0))
    next_vis_idx = np.minimum.accumulate(np.where(ball_vis, idx, n - 1)[::-1])[::-1]

    gap_len = np.where(ball_vis, 0, next_vis_idx - prev_vis_idx)
    time_since_vis = idx - prev_vis_idx
    time_until_vis = next_vis_idx - idx

    last_vis_x = ball_x[prev_vis_idx]
    last_vis_y = ball_y[prev_vis_idx]
    next_vis_x = ball_x[next_vis_idx]
    next_vis_y = ball_y[next_vis_idx]

    # Blend with player focal point only on very long gaps (> 75 frames / 3s)
    alpha_focal = np.clip((gap_len - 75.0) / 150.0, 0.0, 0.55)
    bx_est = (1.0 - alpha_focal) * bx_interp + alpha_focal * mean_px
    by_est = (1.0 - alpha_focal) * by_interp + alpha_focal * mean_py

    # Smooth ball coordinates slightly (3-frame moving average) before taking velocity derivative
    k3 = np.array([0.25, 0.50, 0.25], dtype=np.float32)
    bx_s = np.convolve(bx_est, k3, mode="same")
    by_s = np.convolve(by_est, k3, mode="same")
    vx = np.gradient(bx_s) * FPS
    vy = np.gradient(by_s) * FPS
    ball_vel = np.hypot(vx, vy)

    # Acceleration magnitude (m/s^2) to detect kicks, rebounds, and stops
    ax = np.gradient(vx) * FPS
    ay = np.gradient(vy) * FPS
    ball_acc = np.hypot(ax, ay)

    return (bx_est, by_est, ball_vis, gap_len, time_since_vis, time_until_vis,
            last_vis_x, last_vis_y, next_vis_x, next_vis_y, ball_vel, ball_acc)


# ==============================================================================
# 4. HIGH-RECALL + HIGH-PRECISION DEAD BALL ('D') DETECTOR
# ==============================================================================
def compute_dead_ball_probability(
    player_x, player_y, player_s, ball_x, ball_y,
    bx_est, by_est, ball_vis, gap_len, time_since_vis, time_until_vis,
    last_vis_x, last_vis_y, next_vis_x, next_vis_y, ball_vel,
    min_dist_H, min_dist_A, events
):
    """
    Detects Dead Ball ('D') intervals by combining 4 physical mechanisms:
      1. Out-of-bounds / Boundary Stoppages:
         - Ball visible outside or on the touchline/goal-line.
         - Ball missing (`gap_len >= 18`) where either the exit point (`last_vis`) or
           re-entry/restart point (`next_vis`) is near a boundary (throw-in, corner,
           goal kick box).
      2. Stationary / Pre-Kick Set-Piece Setup:
         - Before a throw-in, free kick, corner, or goal kick, the ball is stationary or
           carried slowly (`ball_vel < 3.2 m/s`) while a single player stands by it waiting
           to kick, followed by a restart acceleration.
      3. Collective Low-Tempo Walk:
         - When play stops, the 75th-percentile and mean player speed drop markedly.
      4. Smart Tagging `GOALS` and `SET PIECES` Windows:
         - Inside `SET PIECES` windows, locates the stoppage start and the exact restart kick.
    """
    n = len(ball_x)
    graded_mask = np.zeros(n, dtype=bool)
    for _, f_start, f_end in HALF_BLOCKS:
        graded_mask[f_start:f_end + 1] = True

    # --- A. Collective Player Tempo (Safe from NaN warnings) ---
    valid_s = ~np.isnan(player_s)
    s_count = np.sum(valid_s, axis=1)
    safe_s = np.where(valid_s, player_s, 0.0)
    mean_s = np.where(s_count > 0, np.sum(safe_s, axis=1) / np.maximum(1, s_count), 0.0)

    # Also compute fraction of tracked players moving slower than walking speed (~6.5 km/h or ~1.8 m/s)
    # Using adaptive median on graded frames so unit (kph vs m/s) is invariant:
    med_s = np.median(mean_s[graded_mask])
    iqr_s = max(1e-3, np.percentile(mean_s[graded_mask], 75) - np.percentile(mean_s[graded_mask], 25))
    z_tempo = (mean_s - med_s) / iqr_s

    # Smooth tempo over 1.6s (40 frames) and 3.0s (75 frames)
    k40 = np.ones(40, dtype=np.float32) / 40.0
    k75 = np.ones(75, dtype=np.float32) / 75.0
    tempo_fast = np.convolve(z_tempo, k40, mode="same")
    tempo_slow = np.convolve(z_tempo, k75, mode="same")

    # --- B. Boundary & Restart Zone Geometry ---
    # Is a coordinate near touchline (y < 2.2 or y > 65.8), goal line (x < 2.2 or x > 103.1),
    # or inside the 6-yard goal-kick box ((x < 7.5 or x > 97.8) and 22 < y < 46)?
    def is_restart_zone(x, y):
        near_sideline = (y < 2.4) | (y > PITCH_WIDTH - 2.4)
        near_goalline = (x < 2.4) | (x > PITCH_LENGTH - 2.4)
        in_goal_box = ((x < 7.8) | (x > PITCH_LENGTH - 7.8)) & (y > 22.0) & (y < 46.0)
        return near_sideline | near_goalline | in_goal_box

    strictly_out = ball_vis & (
        (ball_x < 0.35) | (ball_x > PITCH_LENGTH - 0.35) |
        (ball_y < 0.35) | (ball_y > PITCH_WIDTH - 0.35)
    )
    vis_near_edge = ball_vis & is_restart_zone(ball_x, ball_y)
    exit_near_edge = (~ball_vis) & is_restart_zone(last_vis_x, last_vis_y)
    entry_near_edge = (~ball_vis) & is_restart_zone(next_vis_x, next_vis_y)

    # Gap is a "boundary stoppage gap" if the ball disappears for >= 25 frames (1.0s)
    # and either exited near the edge or re-enters at a restart zone (throw-in, goal kick, corner)
    boundary_gap = (gap_len >= 25) & (exit_near_edge | entry_near_edge)

    # Gap is an "open-field stoppage gap" if the ball disappears for >= 55 frames (2.2s)
    # and player tempo is below normal
    midfield_stoppage_gap = (gap_len >= 55) & (tempo_slow < -0.15)

    # --- C. Stationary / Slow Ball Waiting to Restart ---
    # When a player places the ball for a free kick, goal kick, corner, or holds it for a throw-in:
    # Ball speed is low (< 3.0 m/s) for at least 1 second while tempo is low or near edge
    slow_ball = ball_vis & (ball_vel < 3.2)
    k25 = np.ones(25, dtype=np.float32) / 25.0
    slow_ball_density = np.convolve(slow_ball.astype(np.float32), k25, mode="same")

    # Pre-kick waiting state: ball is slow/stationary near boundary or during low tempo
    waiting_for_restart = (slow_ball_density > 0.65) & (vis_near_edge | (tempo_fast < -0.20))

    # --- D. Smart Tagging SET PIECES & GOALS Integration ---
    st_dead_prior = np.zeros(n, dtype=np.float32)
    st_restart_team = np.zeros(n, dtype=np.int32)  # +1 for H, -1 for A after restart kick

    for ev in events:
        s, e = ev["start_frame"], ev["end_frame"]
        code = ev["code"]
        sp_type = ev["type"].upper()
        team = ev["team"].upper()
        is_barca = "BARCELONA" in team
        is_ath = "ATLETIC" in team or "ATHLETIC" in team

        if code == "GOALS":
            # In Smart Tagging, GOALS runs from the shot (~4-8s before goal) to Kick Off (After Goal)
            # Find exact moment after `s` where ball enters goal (x < 2 or x > 103) or disappears
            search_end = min(e, s + int(12 * FPS))
            goal_frame = s + int(5 * FPS)
            for f_idx in range(s, search_end):
                if not ball_vis[f_idx] or ball_x[f_idx] < 1.5 or ball_x[f_idx] > PITCH_LENGTH - 1.5:
                    goal_frame = f_idx
                    break
            st_dead_prior[goal_frame:e + 1] += 5.0

        elif code == "SET PIECES":
            if "KICK OFF" in sp_type:
                # Before kickoff, ball is dead
                pre_ko = max(0, s - int(15 * FPS))
                st_dead_prior[pre_ko:s + int(1.0 * FPS)] += 4.0
                continue

            # For Throw-in, Corner Kick, Goal Kick, Free Kick (35s window [s, e]):
            # Within [s, e], find the stoppage interval [f_stop, f_kick]:
            # `f_kick` is when the ball is launched back into active play (high ball_vel + visible)
            # typically between s + 3s and e - 4s.
            w_s = min(e, s + int(2.5 * FPS))
            w_e = max(w_s + 1, e - int(3.5 * FPS))
            if w_e > w_s and w_e <= n:
                # Look for the transition from (missing or slow ball / low tempo) -> (fast visible ball)
                # Compute forward 1.2s active ball movement minus backward 1.5s activity
                active_ball = (ball_vis & (ball_vel > 5.5) & (~strictly_out)).astype(np.float32)
                best_kick = w_s + int(0.45 * (w_e - w_s))
                best_score = -1e9
                for cand in range(w_s, w_e, 3):
                    pre_sl = slice(max(s, cand - 50), cand)
                    post_sl = slice(cand, min(e, cand + 35))
                    # Before kick: ball is missing, slow, or tempo is low
                    pre_deadness = 1.0 - np.mean(active_ball[pre_sl]) - 0.25 * np.mean(tempo_fast[pre_sl])
                    # After kick: ball is moving fast or tempo rises
                    post_liveness = np.mean(active_ball[post_sl]) + 0.15 * np.max(ball_vel[post_sl]) / 15.0
                    score = pre_deadness + post_liveness
                    if score > best_score:
                        best_score = score
                        best_kick = cand

                # Find when the stoppage started (`f_stop`) before `best_kick`:
                # Walk backwards from `best_kick` up to `s - 3s` while ball was not in fast live play
                f_stop = s
                for back_f in range(best_kick - 5, max(0, s - int(3 * FPS)), -1):
                    if ball_vis[back_f] and ball_vel[back_f] > 8.5 and not is_restart_zone(ball_x[back_f], ball_y[back_f]):
                        f_stop = back_f + 1
                        break
                else:
                    f_stop = max(0, s)

                st_dead_prior[f_stop:best_kick] += 2.8

                # Assign restart possession boost for 3.5 seconds after `best_kick`
                post_end = min(n, best_kick + int(3.5 * FPS))
                if is_barca:
                    st_restart_team[best_kick:post_end] = 1
                elif is_ath:
                    st_restart_team[best_kick:post_end] = -1
                else:
                    # For Throw-in (Team = N/A), check which team had the closest player right at the restart!
                    chk = slice(max(0, best_kick - 15), min(n, best_kick + 15))
                    if np.nanmedian(min_dist_H[chk]) <= np.nanmedian(min_dist_A[chk]):
                        st_restart_team[best_kick:post_end] = 1
                    else:
                        st_restart_team[best_kick:post_end] = -1

    # --- E. Combine into Calibrated Dead-Ball Logit (`logit_D`) ---
    # Notice we DO NOT penalize `D` just because a single player is close to the ball!
    # Instead, we only penalize `D` when the ball is visibly moving fast (> 6.5 m/s) inside the pitch
    # while collective player tempo is normal/high!
    fast_live_ball = ball_vis & (~strictly_out) & (ball_vel > 6.2) & (tempo_fast > -0.35)

    logit_D = (
        -0.95  # base prior
        + st_dead_prior
        + 2.50 * strictly_out.astype(np.float32)
        + 1.65 * boundary_gap.astype(np.float32)
        + 1.35 * midfield_stoppage_gap.astype(np.float32)
        + 1.40 * waiting_for_restart.astype(np.float32)
        + 0.90 * np.clip(-tempo_slow - 0.20, -0.6, 2.0)
        + 0.70 * np.clip((gap_len - 45.0) / 50.0, 0.0, 1.5) * np.where(tempo_fast < 0.0, 1.0, 0.25)
        - 2.10 * fast_live_ball.astype(np.float32)
    )

    return logit_D, st_restart_team, tempo_fast


# ==============================================================================
# 5. CAUSAL CONTACT-CHAIN OWNERSHIP & GAUSSIAN PITCH CONTROL
# ==============================================================================
def compute_team_ownership_potentials(
    player_x, player_y, bx_est, by_est, ball_vis, gap_len, ball_vel, ball_acc,
    events, st_restart_team
):
    """
    Computes H vs A ownership potentials using:
      1. Direct Contact Detection & Causal Carry-Forward:
         - Identifies frames where a player is in physical contact/control range (< 1.85m)
           of the visible ball.
         - Carries the last controlling team forward across pass flights and loose-ball gaps
           (implementing CHALLENGE.md: "In phases where the ball is contested or loose,
           possession stays with the last team that had it").
      2. Pass-Chain Bridge:
         - If touch `k` and touch `k+1` belong to the same team within 4.5 seconds, the entire
           intervening pass flight receives a strong boost for that team.
      3. Multi-Scale Gaussian Pitch Control:
         - Close control (sigma = 1.8m) + support ring (sigma = 5.5m).
      4. Scoped Smart Tagging Priors:
         - Applied softly during occlusions so fixed 30s template windows never override
           real visible contacts.
    """
    n = len(bx_est)

    # Distances from all Home (0..23) and Away (24..45) slots to estimated ball
    dx_H = player_x[:, HOME_SLOTS] - bx_est[:, None]
    dy_H = player_y[:, HOME_SLOTS] - by_est[:, None]
    dist_H_all = np.hypot(dx_H, dy_H)

    dx_A = player_x[:, AWAY_SLOTS] - bx_est[:, None]
    dy_A = player_y[:, AWAY_SLOTS] - by_est[:, None]
    dist_A_all = np.hypot(dx_A, dy_A)

    min_dist_H = np.nanmin(np.where(np.isnan(dist_H_all), 999.0, dist_H_all), axis=1)
    min_dist_A = np.nanmin(np.where(np.isnan(dist_A_all), 999.0, dist_A_all), axis=1)

    # Gaussian Pitch Control
    ctrl_close_H = np.nansum(np.exp(-0.5 * (dist_H_all / 1.8) ** 2), axis=1)
    ctrl_close_A = np.nansum(np.exp(-0.5 * (dist_A_all / 1.8) ** 2), axis=1)
    ctrl_supp_H = np.nansum(np.exp(-0.5 * (dist_H_all / 5.5) ** 2), axis=1)
    ctrl_supp_A = np.nansum(np.exp(-0.5 * (dist_A_all / 5.5) ** 2), axis=1)

    # --- Causal Contact-Chain Tracker ---
    # A frame has a "Clear Touch / Control" when the ball is visible and one team is within 1.85m
    # with at least 0.6m margin over the opponent (or within 1.3m with 0.35m margin).
    touch_H = ball_vis & (
        ((min_dist_H < 1.85) & (min_dist_A - min_dist_H > 0.60)) |
        ((min_dist_H < 1.25) & (min_dist_A - min_dist_H > 0.30))
    )
    touch_A = ball_vis & (
        ((min_dist_A < 1.85) & (min_dist_H - min_dist_A > 0.60)) |
        ((min_dist_A < 1.25) & (min_dist_H - min_dist_A > 0.30))
    )

    # Require a touch to be sustained for at least 2 frames within a 4-frame window to reject
    # single-frame tracker noise or pure pass-by deflections at 25m/s
    k4 = np.ones(4, dtype=np.float32)
    sust_H = np.convolve(touch_H.astype(np.float32), k4, mode="same") >= 2.0
    sust_A = np.convolve(touch_A.astype(np.float32), k4, mode="same") >= 2.0

    # Also incorporate Smart Tagging restart kicks as anchors
    anchor_team = np.zeros(n, dtype=np.int32)  # +1 = H, -1 = A, 0 = none
    anchor_team[sust_H & (~sust_A)] = 1
    anchor_team[sust_A & (~sust_H)] = -1
    anchor_team[st_restart_team != 0] = st_restart_team[st_restart_team != 0]

    # Pass-Chain & Causal Carry-Forward Signal (`chain_signal` in [-1, +1])
    chain_signal = np.zeros(n, dtype=np.float32)
    anchor_indices = np.nonzero(anchor_team != 0)[0]

    if len(anchor_indices) > 0:
        # Step 1: Absorb isolated single-touch deflections (< 8 frames) sandwiched between same team
        # e.g., H anchor -> brief A deflection (< 0.3s) -> H anchor within 2.5s
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
            # If brief touch (< 10 frames / 0.4s) sandwiched between same opposing team within 3.5s
            if prev_t == next_t and prev_t != t_val and (r_e - r_s) < 10 and (next_s - prev_e) < 90:
                cleaned_anchors[r_s:r_e + 1] = 0

        # Step 2: Build causal carry-forward and same-team pass bridge
        valid_idx = np.nonzero(cleaned_anchors != 0)[0]
        for k_i in range(len(valid_idx) - 1):
            f1 = valid_idx[k_i]
            f2 = valid_idx[k_i + 1]
            t1 = cleaned_anchors[f1]
            t2 = cleaned_anchors[f2]
            span = f2 - f1
            if t1 == t2 and span <= 150:  # Completed pass / carry within 6 seconds
                chain_signal[f1:f2 + 1] = float(t1)
            elif t1 != t2 and span <= 125:
                # Turnover / interception: causal carry-forward holds t1 until f2!
                # Decay slightly only in the last 10 frames before f2
                hold_end = max(f1, f2 - 6)
                chain_signal[f1:hold_end] = 0.85 * float(t1)
                chain_signal[hold_end:f2] = 0.30 * float(t1)

    # --- Smart Tagging Phase Priors (Soft & Scoped) ---
    prior_H = np.zeros(n, dtype=np.float32)
    prior_A = np.zeros(n, dtype=np.float32)
    OFFENSIVE_CODES = {
        "BUILD UP": 1.1, "PROGRESSION": 1.3, "CREATING CHANCES": 0.8,
        "BALL IN FINAL THIRD": 0.9, "BALL IN THE BOX": 0.9,
        "PLAYERS IN THE BOX": 0.7, "ATTACKING TRANSITION": 1.2, "LONG BALL": 0.8,
    }
    DEFENSIVE_CODES = {
        "DEFENDING IN ATTACKING THIRD": 1.1, "DEFENDING IN MIDDLE THIRD": 1.2,
        "DEFENDING IN DEFENSIVE THIRD": 0.8, "DEFENSIVE TRANSITION": 1.1,
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

    st_diff = 1.1 * np.tanh((prior_H - prior_A) / 2.5)

    # Weight instantaneous proximity strongly when ball is visible and near a player,
    # and rely on `chain_signal` + `st_diff` during pass flights and occlusions!
    vis_weight = np.exp(-gap_len / 20.0)
    dist_diff = np.clip(min_dist_A - min_dist_H, -12.0, 12.0)

    # Combined H vs A log-odds (`diff_HA`)
    diff_HA = (
        1.45 * chain_signal +
        0.45 * vis_weight * dist_diff +
        0.90 * vis_weight * (ctrl_close_H - ctrl_close_A) +
        0.25 * (ctrl_supp_H - ctrl_supp_A) +
        (0.45 + 0.55 * (1.0 - vis_weight)) * st_diff +
        1.80 * st_restart_team.astype(np.float32)
    )

    return diff_HA, min_dist_H, min_dist_A, ctrl_close_H, ctrl_close_A, chain_signal


# ==============================================================================
# 6. SEMI-MARKOV VITERBI DECODER & DOMAIN CONSOLIDATION
# ==============================================================================
def decode_possession_v2(diff_HA, logit_D, ctrl_H, ctrl_A, chain_signal):
    """
    Decodes each half into ('H', 'A', 'D') with exact Viterbi + domain spell consolidation.
    """
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

        # Transition probabilities:
        # Allow slightly easier entry into 'D' than V1 so real stoppages aren't missed!
        p_stay_team = 0.988
        p_stay_dead = 0.990
        rem_t = 1.0 - p_stay_team
        rem_d = 1.0 - p_stay_dead
        trans = np.array([
            [p_stay_team, 0.58 * rem_t, 0.42 * rem_t],
            [0.58 * rem_t, p_stay_team, 0.42 * rem_t],
            [0.50 * rem_d, 0.50 * rem_d, p_stay_dead]
        ], dtype=np.float64)
        log_trans = np.log(trans)

        dp = np.zeros((T, K), dtype=np.float64)
        backtrack = np.zeros((T, K), dtype=np.int32)
        dp[0] = lp[0]

        obs_scale = 0.22
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

        # Post-processing consolidation:
        # 1. Suppress false micro-stoppages ('D' < 50 frames / 2.0s)
        states = consolidate_runs(states, ctrl_H[sl], ctrl_A[sl], min_pos_frames=40, min_dead_frames=50)
        labels[sl] = state_map[states]

    return labels


def get_runs(arr):
    if len(arr) == 0:
        return []
    diffs = np.diff(arr)
    change_idx = np.nonzero(diffs != 0)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(arr)]))
    return [[int(arr[s]), int(s), int(e), int(e - s)] for s, e in zip(starts, ends)]


def consolidate_runs(states, ctrl_H, ctrl_A, min_pos_frames=40, min_dead_frames=50):
    out = states.copy()

    # Pass 1: Remove micro-stoppages ('D' == 2 shorter than min_dead_frames = 2.0s)
    runs = get_runs(out)
    for i, (st, s, e, length) in enumerate(runs):
        if st == 2 and length < min_dead_frames:
            prev_st = runs[i - 1][0] if i > 0 else (runs[i + 1][0] if i + 1 < len(runs) else 0)
            next_st = runs[i + 1][0] if i + 1 < len(runs) else prev_st
            out[s:e] = prev_st

    # Pass 2: Absorb uncontrolled deflections / duels (< min_pos_frames = 1.6s)
    for _ in range(2):
        runs = get_runs(out)
        changed = False
        for i in range(1, len(runs) - 1):
            st, s, e, length = runs[i]
            prev_st = runs[i - 1][0]
            next_st = runs[i + 1][0]
            if st in (0, 1) and length < min_pos_frames:
                team_ctrl = np.mean(ctrl_H[s:e]) if st == 0 else np.mean(ctrl_A[s:e])
                if prev_st == next_st and team_ctrl < 1.45:
                    out[s:e] = prev_st
                    changed = True
                elif length < 22:  # < 0.88s blip at a transition
                    out[s:e] = prev_st
                    changed = True
        if not changed:
            break

    return out


# ==============================================================================
# 7. MAIN ENTRYPOINT
# ==============================================================================
def generate_submission_v2(data_dir, output_csv):
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

    # Preliminary distances for set-piece throw-in team inference
    dx_H = player_x[:, HOME_SLOTS] - bx_est[:, None]
    dy_H = player_y[:, HOME_SLOTS] - by_est[:, None]
    min_dist_H = np.nanmin(np.where(np.isnan(dx_H), 999.0, np.hypot(dx_H, dy_H)), axis=1)

    dx_A = player_x[:, AWAY_SLOTS] - bx_est[:, None]
    dy_A = player_y[:, AWAY_SLOTS] - by_est[:, None]
    min_dist_A = np.nanmin(np.where(np.isnan(dx_A), 999.0, np.hypot(dx_A, dy_A)), axis=1)

    print("[INFO] Computing physical Dead-Ball ('D') detector (boundary exits + pre-kick waits + tempo)...")
    logit_D, st_restart_team, tempo_fast = compute_dead_ball_probability(
        player_x, player_y, player_s, ball_x, ball_y,
        bx_est, by_est, ball_vis, gap_len, time_since_vis, time_until_vis,
        last_vis_x, last_vis_y, next_vis_x, next_vis_y, ball_vel,
        min_dist_H, min_dist_A, events
    )

    print("[INFO] Computing Causal Contact-Chain ownership and Gaussian Pitch Control...")
    diff_HA, min_dist_H, min_dist_A, ctrl_H, ctrl_A, chain_signal = compute_team_ownership_potentials(
        player_x, player_y, bx_est, by_est, ball_vis, gap_len, ball_vel, ball_acc,
        events, st_restart_team
    )

    print("[INFO] Decoding possession spells with V2 Semi-Markov Viterbi...")
    labels = decode_possession_v2(diff_HA, logit_D, ctrl_H, ctrl_A, chain_signal)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_sub = pd.DataFrame({
        "frame": np.arange(N_VIDEO_FRAMES, dtype=np.int32),
        "label": labels
    })
    df_sub.to_csv(output_csv, index=False, lineterminator="\n")
    print(f"[SUCCESS] Saved V2 submission to: {output_csv}")

    graded_labels = np.concatenate([labels[s:e+1] for _, s, e in HALF_BLOCKS])
    u, counts = np.unique(graded_labels, return_counts=True)
    stats = dict(zip(u, counts))
    total_g = len(graded_labels)
    print("[STATS] V2 Graded frame distribution:")
    for k in ["H", "A", "D", "X"]:
        c = stats.get(k, 0)
        print(f"    {k}: {c:6d} frames ({100.0 * c / total_g:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creative V2 Contact-Chain HSMM Model for More than a Hack 2027")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("MTAH_DATA_DIR", "./data"))
    parser.add_argument("--output", type=str, default="evaluation/results/creative_v2_submission.csv")
    args = parser.parse_args()
    generate_submission_v2(args.data_dir, args.output)
