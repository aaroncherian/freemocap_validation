import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Your recordings dictionary
recordings = {
    "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
    "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
    "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
    "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
}

# Optional: fixed colors per condition for readability
COND_COLORS = {
    "neg_5":  "#94342b",
    "neg_25": "#d39182",
    "neutral": "#524F4F",
    "pos_25": "#7bb6c6",
    "pos_5":  "#447c8e",
}

def get_hip_curve(traj_df: pd.DataFrame, marker: str, axis: str = "z", stat: str = "mean"):
    """
    Extracts a single hip trajectory curve from the long-format trajectories dataframe.
    Assumes columns: tracker, marker, percent_gait_cycle, value, axis, stat
    """
    sub = (
        traj_df[
            (traj_df["marker"] == marker)
            & (traj_df["axis"] == axis)
            & (traj_df["stat"] == stat)
        ]
        .sort_values("percent_gait_cycle")
    )
    if sub.empty:
        raise ValueError(f"No rows found for marker={marker}, axis={axis}, stat={stat}")

    percent = sub["percent_gait_cycle"].to_numpy()
    values = sub["value"].to_numpy().astype(float)
    return percent, values

def plot_hip_trajectories_grid(recordings: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    
    # Map (system, side) to subplot index
    # Top row: Qualisys, bottom row: MediaPipe
    # Left col: left hip, right col: right hip
    subplot_info = {
        ("qualisys", "left"):  (0, 0),
        ("qualisys", "right"): (0, 1),
        ("mediapipe_dlc", "left"):  (1, 0),
        ("mediapipe_dlc", "right"): (1, 1),
    }

    # For legend: keep one handle per condition
    legend_handles = {}
    
    for cond, base in recordings.items():
        color = COND_COLORS.get(cond, None)

        for tracker_name, tracker_dir in [("qualisys", "qualisys"),
                                          ("mediapipe_dlc", "mediapipe_dlc")]:
            traj_csv = base / "validation" / tracker_dir / "trajectories" / "trajectories_per_stride_summary_stats.csv"
            if not traj_csv.exists():
                print(f"⚠️ Missing file for {cond}, {tracker_name}: {traj_csv}")
                continue

            traj_df = pd.read_csv(traj_csv)

            # Left hip
            try:
                p_left, y_left = get_hip_curve(traj_df, marker="left_hip", axis="z", stat="mean")
                ax_row, ax_col = subplot_info[(tracker_name, "left")]
                ax = axes[ax_row, ax_col]
                line_left, = ax.plot(p_left, y_left, label=cond, color=color)
                if cond not in legend_handles:
                    legend_handles[cond] = line_left
            except Exception as e:
                print(f"Error getting left hip for {cond}, {tracker_name}: {e}")

            # Right hip
            try:
                p_right, y_right = get_hip_curve(traj_df, marker="right_hip", axis="z", stat="mean")
                ax_row, ax_col = subplot_info[(tracker_name, "right")]
                ax = axes[ax_row, ax_col]
                ax.plot(p_right, y_right, label=cond, color=color)
            except Exception as e:
                print(f"Error getting right hip for {cond}, {tracker_name}: {e}")

    # Titles and labels
    axes[0, 0].set_title("Qualisys – Left hip (z)")
    axes[0, 1].set_title("Qualisys – Right hip (z)")
    axes[1, 0].set_title("MediaPipe – Left hip (z)")
    axes[1, 1].set_title("MediaPipe – Right hip (z)")

    for row in range(2):
        for col in range(2):
            axes[row, col].set_xlabel("% gait")
            axes[row, col].set_ylabel("Vertical position (z)")

    # Shared legend outside the grid
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            ncol=len(legend_handles),
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.tight_layout()
    plt.show()

# # Call it
plot_hip_trajectories_grid(recordings)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Your recordings dictionary
recordings = {
    "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
    "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
    "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
    "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
}

# Optional consistent colors for conditions
COND_COLORS = {
    "neg_5":  "#94342b",
    "neg_25": "#d39182",
    "neutral": "#524F4F",
    "pos_25": "#7bb6c6",
    "pos_5":  "#447c8e",
}

def get_obliquity_curve(traj_df: pd.DataFrame, stat: str = "mean"):
    """
    Pelvic obliquity curve from long-format trajectories df:
    obliquity(t) = z_left_hip(t) - z_right_hip(t)
    """
    L = (
        traj_df[
            (traj_df["marker"] == "left_hip")
            & (traj_df["axis"] == "z")
            & (traj_df["stat"] == stat)
        ]
        .sort_values("percent_gait_cycle")
    )
    R = (
        traj_df[
            (traj_df["marker"] == "right_hip")
            & (traj_df["axis"] == "z")
            & (traj_df["stat"] == stat)
        ]
        .sort_values("percent_gait_cycle")
    )

    if L.empty or R.empty:
        raise ValueError("Missing left_hip or right_hip z rows for obliquity computation")

    percent = L["percent_gait_cycle"].to_numpy()
    obliq = L["value"].to_numpy().astype(float) - R["value"].to_numpy().astype(float)
    return percent, obliq


def plot_obliquity_curves(recordings: dict):
    """
    Makes a 1x2 figure:
      Left  panel: Qualisys pelvic obliquity curves by condition
      Right panel: MediaPipe pelvic obliquity curves by condition
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    system_axes = {
        "qualisys": axes[0],
        "mediapipe_dlc": axes[1],
    }

    # For a compact summary of magnitudes
    summary = { "qualisys": {}, "mediapipe_dlc": {} }

    for cond, base in recordings.items():
        color = COND_COLORS.get(cond, None)

        for system, subdir in [("qualisys", "qualisys"),
                               ("mediapipe_dlc", "mediapipe_dlc")]:
            traj_csv = base / "validation" / subdir / "trajectories" / "trajectories_per_stride_summary_stats.csv"
            if not traj_csv.exists():
                print(f"⚠️ Missing {system} trajectories for {cond}: {traj_csv}")
                continue

            traj_df = pd.read_csv(traj_csv)

            try:
                percent, obliq = get_obliquity_curve(traj_df)
            except Exception as e:
                print(f"Error computing obliquity for {cond}, {system}: {e}")
                continue

            ax = system_axes[system]
            ax.plot(percent, obliq, label=cond, color=color)

            # Store a simple magnitude metric
            summary[system][cond] = np.mean(np.abs(obliq))

    # Format plots
    axes[0].set_title("Qualisys – Pelvic Obliquity\n(left_hip z – right_hip z)")
    axes[1].set_title("MediaPipe – Pelvic Obliquity\n(left_hip z – right_hip z)")

    for ax in axes:
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("% gait")
        ax.set_ylabel("Obliquity (same units as z)")
        ax.grid(False)

    # Shared legend above
    # Grab handles from one axis (they all share labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.show()

    # Print


# plot_obliquity_curves(recordings)


# -----------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Your recordings dictionary
# recordings = {
#     "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
#     "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
#     "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
#     "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
#     "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
# }

# # Optional color mapping
# COND_COLORS = {
#     "neg_5":  "#94342b",
#     "neg_25": "#d39182",
#     "neutral": "#524F4F",
#     "pos_25": "#7bb6c6",
#     "pos_5":  "#447c8e",
# }


# def get_angle_curve(df, joint="hip", side="left", component="flex_ext", stat="mean"):
#     """
#     Extracts a hip flexion-extension curve from long-format joint angle CSV.
#     Columns assumed:
#       joint, side, component, percent_gait_cycle, stat, value
#     """
#     sub = (
#         df[
#             (df["joint"] == joint)
#             & (df["side"] == side)
#             & (df["component"] == component)
#             & (df["stat"] == stat)
#         ]
#         .sort_values("percent_gait_cycle")
#     )

#     if sub.empty:
#         raise ValueError(
#             f"No rows for joint={joint}, side={side}, component={component}, stat={stat}"
#         )

#     percent = sub["percent_gait_cycle"].to_numpy()
#     values = sub["value"].to_numpy().astype(float)
#     return percent, values


# def plot_hip_fe_grid(recordings: dict):
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

#     subplot_info = {
#         ("qualisys", "left"):  (0, 0),
#         ("qualisys", "right"): (0, 1),
#         ("mediapipe_dlc", "left"):  (1, 0),
#         ("mediapipe_dlc", "right"): (1, 1),
#     }

#     legend_handles = {}

#     for cond, base in recordings.items():
#         color = COND_COLORS.get(cond, None)

#         # Loop through both tracking systems
#         for tracker, tracker_dir in [
#             ("qualisys", "qualisys"),
#             ("mediapipe_dlc", "mediapipe_dlc"),
#         ]:

#             angle_csv = base / "validation" / tracker_dir / "joint_angles" / "joint_angles_per_stride_summary_stats.csv"
#             if not angle_csv.exists():
#                 print(f"⚠️ Missing: {angle_csv}")
#                 continue

#             ang_df = pd.read_csv(angle_csv)

#             # Left hip FE
#             try:
#                 p_left, fe_left = get_angle_curve(ang_df, side="left")
#                 row, col = subplot_info[(tracker, "left")]
#                 ax = axes[row, col]
#                 line, = ax.plot(p_left, fe_left, label=cond, color=color)

#                 if cond not in legend_handles:
#                     legend_handles[cond] = line
#             except Exception as e:
#                 print(f"Left hip error ({cond}, {tracker}): {e}")

#             # Right hip FE
#             try:
#                 p_right, fe_right = get_angle_curve(ang_df, side="right")
#                 row, col = subplot_info[(tracker, "right")]
#                 ax = axes[row, col]
#                 ax.plot(p_right, fe_right, label=cond, color=color)
#             except Exception as e:
#                 print(f"Right hip error ({cond}, {tracker}): {e}")

#     # Titles
#     axes[0, 0].set_title("Qualisys – Left Hip Flex/Ext")
#     axes[0, 1].set_title("Qualisys – Right Hip Flex/Ext")
#     axes[1, 0].set_title("MediaPipe – Left Hip Flex/Ext")
#     axes[1, 1].set_title("MediaPipe – Right Hip Flex/Ext")

#     # Axes labels
#     for r in range(2):
#         for c in range(2):
#             axes[r, c].set_xlabel("% gait")
#             axes[r, c].set_ylabel("Angle (deg)")

#     # Shared legend
#     fig.legend(
#         legend_handles.values(),
#         legend_handles.keys(),
#         loc="upper center",
#         ncol=len(legend_handles),
#         bbox_to_anchor=(0.5, 1.02),
#     )

#     plt.tight_layout()
#     plt.show()


# # Run it
# plot_hip_fe_grid(recordings)
