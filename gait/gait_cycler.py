from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd

# ----------------------- Helpers (kept close to original) -----------------------
def normalize_gait_cycle(gait_cycle: np.ndarray, n_points: int = 100) -> np.ndarray:
    num_frames = gait_cycle.shape[0]
    x = np.linspace(0, 10, num=num_frames)
    x_new = np.linspace(0, 10, num=n_points)
    return np.apply_along_axis(lambda v: np.interp(x_new, x, v), axis=0, arr=gait_cycle)

def cycles_to_dataframe(cycles: np.ndarray, system: str, angle_name: str) -> pd.DataFrame:
    """
    cycles: shape (n_cycles, n_points)
    -> DataFrame columns: system, angle, stride, percent_gait_cycle, value
    """
    n_cycles, n_points = cycles.shape
    time_norm = np.linspace(0, 100, n_points)

    records = []
    for stride_idx in range(n_cycles):
        for t_idx, angle_value in enumerate(cycles[stride_idx]):
            records.append({
                "system": system,
                "angle": angle_name,
                "stride": stride_idx,
                "percent_gait_cycle": time_norm[t_idx],
                "value": float(angle_value),
            })
    return pd.DataFrame.from_records(records)

# ----------------------- Main -----------------------
def get_angle_strides(path_to_recording: Path,
                           frame_range: range,
                           tracker: str,
                           n_points: int = 100,
                           angle_whitelist: list[str] | None = None) -> pd.DataFrame:
    """
    Slice strides by right heel strikes and export EVERY joint-angle column
    from both systems into one long CSV:
        validation/<tracker>/<tracker>_joint_angle_by_stride.csv
    """
    # Paths
    path_events = path_to_recording/'validation'/'qualisys'/'gait_events.csv'
    path_qtm_angles = path_to_recording/'validation'/'qualisys'/'qualisys_joint_angles.csv'
    path_fmc_angles = path_to_recording/'validation'/tracker/f'{tracker}_joint_angles.csv'
    out_csv = path_to_recording/'validation'/tracker/f'{tracker}_joint_angle_by_stride.csv'

    # Load
    events = pd.read_csv(path_events)
    qtm_df = pd.read_csv(path_qtm_angles)
    fmc_df = pd.read_csv(path_fmc_angles)

    # Decide which columns are angles (numeric + shared). You can pass a whitelist to restrict.
    def numeric_cols(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=[np.number]).columns.tolist()

    qtm_cols = numeric_cols(qtm_df)
    fmc_cols = numeric_cols(fmc_df)
    shared = [c for c in qtm_cols if c in fmc_cols]

    # Drop obvious non-angle numeric columns if present
    for meta in ["frame", "time", "percent_gait_cycle"]:
        if meta in shared:
            shared.remove(meta)

    if angle_whitelist:
        angle_cols = [c for c in shared if c in angle_whitelist]
    else:
        angle_cols = shared

    if not angle_cols:
        raise ValueError("No overlapping numeric angle columns found between Qualisys and tracker CSVs.")

    # Heel strikes (right) within frame_range
    mask = (events["foot"] == "right") & (events["event"] == "heel_strike")
    heelstrikes = [fr for fr in events.loc[mask, "frame"].to_list() if fr in frame_range]

    heelstrike_slices = []
    for i in range(len(heelstrikes) - 1):
        s = slice(heelstrikes[i], heelstrikes[i + 1])
        if s.start < s.stop and s.start in frame_range and (s.stop - 1) in frame_range:
            heelstrike_slices.append(s)

    if not heelstrike_slices:
        raise ValueError("No valid right-foot heel-strike stride slices found in the given frame_range.")

    # If CSVs have a 'frame' column, align by it; otherwise use positional index
    use_frame_index_qtm = "frame" in qtm_df.columns
    use_frame_index_fmc = "frame" in fmc_df.columns
    if use_frame_index_qtm:
        qtm_df = qtm_df.set_index("frame")
    if use_frame_index_fmc:
        fmc_df = fmc_df.set_index("frame")

    all_parts = []

    # Loop over every angle column, build cycles for both systems, append long-form
    for angle in angle_cols:
        qtm_cycles = []
        fmc_cycles = []

        for s in heelstrike_slices:
            if use_frame_index_qtm:
                q_slice = qtm_df.loc[s.start:s.stop - 1, angle].to_numpy()
            else:
                q_slice = qtm_df[angle].iloc[s].to_numpy()

            if use_frame_index_fmc:
                f_slice = fmc_df.loc[s.start:s.stop - 1, angle].to_numpy()
            else:
                f_slice = fmc_df[angle].iloc[s].to_numpy()

            if q_slice.size < 2 or f_slice.size < 2:
                continue

            qtm_cycles.append(normalize_gait_cycle(q_slice, n_points=n_points))
            fmc_cycles.append(normalize_gait_cycle(f_slice, n_points=n_points))

        if not qtm_cycles or not fmc_cycles:
            # No usable strides for this angle (skip quietly)
            continue

        qtm_arr = np.stack(qtm_cycles, axis=0)  # (n_cycles, n_points)
        fmc_arr = np.stack(fmc_cycles, axis=0)

        all_parts.append(cycles_to_dataframe(qtm_arr, system="qualisys", angle_name=angle))
        all_parts.append(cycles_to_dataframe(fmc_arr, system=tracker, angle_name=angle))

    if not all_parts:
        raise ValueError("No valid cycles generated for any angle.")

    out_df = pd.concat(all_parts, ignore_index=True)

    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"[joint_angle_by_stride] angles={len(out_df['angle'].unique())}, "
          f"strides≈{out_df['stride'].max()+1 if not out_df.empty else 0}, rows={len(out_df):,}")
    print(f"Saved -> {out_csv}")
    return out_df

# ----------------------- Script entry -----------------------
if __name__ == "__main__":
    frame_range = range(600, 2350)
    path_to_recording = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
    tracker = "mediapipe_dlc"

    # angle_whitelist=None uses all shared numeric columns.
    get_angle_strides(path_to_recording, frame_range, tracker, n_points=101, angle_whitelist=None)
