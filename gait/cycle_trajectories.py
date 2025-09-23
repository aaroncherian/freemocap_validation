from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd
from skellymodels.managers.human import Human

# ----------------------- Config -----------------------
N_POINTS = 101  # 0..100% GC inclusive works nicely with 101 points
RIGHT_MARKERS = {
    "right_knee": "knee",
    "right_ankle": "ankle",
    "right_heel": "heel",
    "right_foot_index": "toe",  # your toe marker
}

# ----------------------- Helpers -----------------------
def unit_linspace(n_points:int=N_POINTS):
    # 0..100% inclusive to match gait convention
    return np.linspace(0, 100, n_points)

def normalize_gait_cycle(gait_cycle: np.ndarray, n_points:int=N_POINTS) -> np.ndarray:
    """
    Resample a per-frame (T x 3) trajectory to n_points along a unit arc.
    """
    num_frames = gait_cycle.shape[0]
    # simple normalized time axis [0..1]
    x = np.linspace(0.0, 1.0, num=num_frames)
    x_new = np.linspace(0.0, 1.0, num=n_points)
    # interp each coord independently
    out = np.empty((n_points, 3), dtype=float)
    for d in range(3):
        out[:, d] = np.interp(x_new, x, gait_cycle[:, d])
    return out

def cycles_to_dataframe(marker_cycles: np.ndarray,
                        system: str,
                        marker_name: str) -> pd.DataFrame:
    """
    marker_cycles: (n_cycles, n_points, 3)
    Returns long DataFrame with columns:
      [system, marker, stride, percent_gait_cycle, x, y, z]
    """
    n_cycles, n_points, _ = marker_cycles.shape
    time_norm = unit_linspace(n_points)

    records = []
    for stride_idx in range(n_cycles):
        # iterate over normalized time (0..100%)
        for t_idx in range(n_points):
            x, y, z = marker_cycles[stride_idx, t_idx]
            records.append({
                "system": system,
                "marker": marker_name,
                "stride": stride_idx,
                "percent_gait_cycle": time_norm[t_idx],
                "x": float(x),
                "y": float(y),
                "z": float(z),
            })
    return pd.DataFrame.from_records(records)

# ----------------------- Main -----------------------
def get_joint_strides(path_to_recording: Path,
                      frame_range: range,
                      tracker: str) -> pd.DataFrame:
    """
    Build stride-normalized trajectories for right knee/ankle/heel/toe for:
      - Qualisys (events source)
      - FreeMoCap (tracker arg)

    Uses right heel strikes from qualisys gait_events.csv to slice strides.
    Saves a combined CSV to: validation/<tracker>/joint_strides.csv
    Returns the combined DataFrame.
    """
    # Paths
    path_qtm_events = path_to_recording / "validation" / "qualisys" / "gait_events.csv"
    path_qtm = path_to_recording / "validation" / "qualisys"
    path_fmc = path_to_recording / "validation" / tracker
    out_csv = path_to_recording / "validation" / tracker / f"{tracker}_joint_trajectory_by_stride.csv"

    # Load humans
    qtm_h: Human = Human.from_data(path_qtm)
    fmc_h: Human = Human.from_data(path_fmc)

    qtm = qtm_h.body.xyz.as_dict
    fmc = fmc_h.body.xyz.as_dict

    # Load gait events (from Qualisys)
    gait_events = pd.read_csv(path_qtm_events)

    # Right heel strikes within the analysis frame_range
    mask = (
        (gait_events["foot"] == "right") &
        (gait_events["event"] == "heel_strike")
    )
    heelstrikes = [fr for fr in gait_events.loc[mask, "frame"].to_list()
                   if fr in frame_range]

    # Build slices [HS_i : HS_{i+1}) ensuring they’re within the provided range
    slices = []
    for i in range(len(heelstrikes) - 1):
        s = slice(heelstrikes[i], heelstrikes[i + 1])
        # sanity: ensure non-empty and within range
        if s.start < s.stop and s.start in frame_range and (s.stop - 1) in frame_range:
            slices.append(s)

    if len(slices) == 0:
        raise ValueError("No valid right-foot heel-strike stride slices found in the given frame_range.")

    # Collect normalized cycles for each marker & system
    all_dfs = []

    for key, short_name in RIGHT_MARKERS.items():
        if key not in qtm or key not in fmc:
            raise KeyError(f"Marker '{key}' not present in one of the systems (qualisys or {tracker}).")

        # Qualisys cycles (n_cycles, n_points, 3)
        qtm_cycles = np.stack([normalize_gait_cycle(qtm[key][s]) for s in slices], axis=0)
        df_qtm = cycles_to_dataframe(qtm_cycles, system="qualisys", marker_name=short_name)
        all_dfs.append(df_qtm)

        # FreeMoCap cycles
        fmc_cycles = np.stack([normalize_gait_cycle(fmc[key][s]) for s in slices], axis=0)
        df_fmc = cycles_to_dataframe(fmc_cycles, system=tracker, marker_name=short_name)
        all_dfs.append(df_fmc)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Ensure output directory exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False)

    print(f"[joint_strides] Saved {len(df_all):,} rows to {out_csv}")
    print(f"  Strides detected: {len(slices)}")
    print(f"  Markers: {list(RIGHT_MARKERS.values())}")
    return df_all

# ----------------------- Script entry -----------------------
if __name__ == "__main__":
    frame_range = range(600, 2350)
    path_to_recording = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1")
    tracker = "mediapipe_dlc"
    get_joint_strides(path_to_recording, frame_range, tracker)
