import pandas as pd
import numpy as np
from pathlib import Path

conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }

tracker = "rtmpose_dlc"

joints_to_load = [
    dict(joint="knee", side="right", component="flex_ext"),
    dict(joint="ankle", side="right", component="dorsi_plantar"),
]

all_rows = []

def load_angle_summary_for_tracker(
    conditions: dict[str, Path | str],
    tracker_dir: str,
    *,
    joint: str,
    side: str = "right",
    component: str = "dorsi_plantar",
) -> pd.DataFrame:
    """
    Read:
        <root>/validation/<tracker_dir>/joint_angles/joint_angles_per_stride.csv
    and return:
        ['system', 'condition', 'joint', 'percent_gait_cycle', 'mean', 'std']
    """
    all_summaries: list[pd.DataFrame] = []

    for cond, root in conditions.items():
        root = Path(root)
        csv_path = (
            root
            / "validation"
            / tracker_dir
            / "joint_angles"
            / "joint_angles_per_stride.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing summary CSV for condition '{cond}' and tracker '{tracker_dir}': {csv_path}"
            )

        df = pd.read_csv(csv_path)

        if "joint" in df.columns:
            df = df[df["joint"] == joint]
        if "side" in df.columns:
            df = df[df["side"] == side]
        if "component" in df.columns:
            df = df[df["component"] == component]
        
        if df.empty:
            raise ValueError(
                f"No rows in {csv_path} for joint={joint}, side={side}, component={component}"
            )

        df["condition"] = cond

        all_summaries.append(df)

    out = pd.concat(all_summaries, ignore_index=True)
    print(f"[{tracker_dir}] loaded rows ({joint=}):", len(out))
    return out

for jt in joints_to_load:
    for system in (tracker, "qualisys"):
        all_rows.append(
            load_angle_summary_for_tracker(
                {k: Path(v) for k, v in conditions.items()},
                tracker_dir=system,
                joint=jt["joint"],
                side=jt["side"],
                component=jt["component"],
            )
        )

summary_all = pd.concat(all_rows, ignore_index=True)

df_stance = summary_all[
    (summary_all["percent_gait_cycle"] <= 60) & (summary_all["percent_gait_cycle"] >= 20) &
    (summary_all["joint"] == "ankle")
]

ankle_peaks = df_stance.groupby(["cycle", "tracker", "condition"]).agg(max = ("angle", "max")).reset_index()
max_peaks = ankle_peaks.groupby(["condition", "tracker"]).agg(mean = ("max", "mean"), std = ("max", "std")).reset_index()

f = 2
# # Restrict to stance


# # Find index of minimum angle per stride
# idx = df_stance.groupby(
#     ["cycle", "tracker", "condition"]
# )["angle"].idxmax()

# # Extract those rows
# ankle_peaks = df_stance.loc[idx].copy()

# # Rename for clarity
# ankle_peaks = ankle_peaks.rename(columns={
#     "angle": "peak_value",
#     "percent_gait_cycle": "peak_time_pct"
# })

# import matplotlib.pyplot as plt

# def debug_plot_strides(df, tracker, condition):
#     df_plot = df.query(
#         "joint=='ankle' and tracker==@tracker and condition==@condition"
#     ).copy()

#     fig, ax = plt.subplots(figsize=(6,4))

#     # plot each stride
#     for cyc, g in df_plot.groupby("cycle"):
#         g = g.sort_values("percent_gait_cycle")
#         ax.plot(
#             g["percent_gait_cycle"],
#             g["angle"],
#             color="gray",
#             alpha=0.2
#         )

#     # plot mean curve
#     mean_curve = (
#         df_plot.groupby("percent_gait_cycle")["angle"]
#         .mean()
#         .reset_index()
#     )

#     ax.plot(
#         mean_curve["percent_gait_cycle"],
#         mean_curve["angle"],
#         color="black",
#         linewidth=2,
#         label="mean"
#     )

#     ax.set_title(f"{tracker} – {condition}")
#     ax.set_xlabel("Gait cycle (%)")
#     ax.set_ylabel("Ankle angle (deg)")
#     ax.legend()
#     plt.show()

# debug_plot_strides(summary_all, "rtmpose_dlc", "pos_5_6")

f = 2

