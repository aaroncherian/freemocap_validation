import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# Your condition folders
conditions = {
    "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
}

# Order for plotting
ordered_conditions = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]

# Where gait CSVs live relative to each recording folder
qualisys_rel = Path("validation/qualisys/gait_parameters/qualisys_gait_summary_stats.csv")
fmc_rel = Path("validation/mediapipe_dlc/gait_parameters/gait_summary_stats.csv")

# Collect data
rows = []

for cond, folder in conditions.items():
    folder = Path(folder)

    q_file = folder / qualisys_rel
    fmc_file = folder / fmc_rel

    q_df = pd.read_csv(q_file)
    f_df = pd.read_csv(fmc_file)

    # --- SIMPLE VERSION ---
    # Assume each summary CSV contains one row of means for stride parameters.
    # Add columns to track condition and system.
    q_df["condition"] = cond
    q_df["system"] = "qualisys"

    f_df["condition"] = cond
    f_df["system"] = "freeMoCap"

    rows.append(q_df)
    rows.append(f_df)

# Combine everything into one table
df = pd.concat(rows, ignore_index=True)

print(df.head())

plt.rcParams["figure.figsize"] = (8, 4)

def plot_metric(metric: str, side: str):
    """
    metric: one of ['stride_duration', 'stance_duration', 'swing_duration',
                    'stance_pct', 'swing_pct']
    side:   'left' or 'right'
    """

    sub = df[(df["metric"] == metric) & (df["side"] == side)].copy()

    # enforce condition order
    sub["condition"] = pd.Categorical(sub["condition"],
                                      categories=ordered_conditions,
                                      ordered=True)
    sub = sub.sort_values("condition")

    plt.figure()
    for system in sub["system"].unique():
        s = sub[sub["system"] == system]
        plt.plot(
            s["condition"],
            s["mean"],
            marker="o",
            linestyle="-",
            label=system,
        )

    plt.title(f"{metric} ({side} leg) across ankle flexion conditions")
    plt.xlabel("Condition")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# # durations
# plot_metric("stride_duration", "right")
# plot_metric("stance_duration", "right")
# plot_metric("swing_duration", "right")

# # stance / swing percentages
# plot_metric("stance_pct", "right")
# plot_metric("swing_pct", "right")

# # same for left side
# plot_metric("stride_duration", "left")

def plot_all_right_side_metrics(df: pd.DataFrame):
    # Choose the order of metrics you care about
    metric_order = [
        "stride_duration",
        "stance_duration",
        "swing_duration",
        "stance_pct",
        "swing_pct",
    ]

    # Filter to right side only
    right_df = df[df["side"] == "right"].copy()
    right_df["condition"] = pd.Categorical(
        right_df["condition"],
        categories=ordered_conditions,
        ordered=True,
    )

    # Only keep metrics that actually exist in the data
    metric_order = [m for m in metric_order if m in right_df["metric"].unique()]

    n_metrics = len(metric_order)

    # Layout: 2 rows x 3 cols (5 metrics → last axis turned off)
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6), sharex=True)
    axes = axes.flatten()

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        sub = right_df[right_df["metric"] == metric].sort_values("condition")

        for system in sub["system"].unique():
            s = sub[sub["system"] == system]
            ax.plot(
                s["condition"],
                s["mean"],
                marker="o",
                linestyle="-",
                label=system,
            )

        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Condition")
        ax.set_ylabel("mean")
        ax.grid(alpha=0.3)

    # Turn off any unused axes (e.g., the 6th one if we only have 5 metrics)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Make a single legend for the whole figure (from first used axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle("Right-side gait metrics across ankle flexion conditions", y=1.03)
    fig.tight_layout()
    plt.show()


# plot_all_right_side_metrics(df)



QUAL_COLOR = "#d62728"   # Qualisys red
FMC_COLOR  = "#1f77b4"   # FreeMoCap blue

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

QUAL_COLOR = "#d62728"   # Qualisys red
FMC_COLOR  = "#1f77b4"   # FreeMoCap blue


def plot_right_side_deltas(df, ordered_conditions):
    # ---- normalize system labels ----
    df = df.copy()
    df["system"] = df["system"].str.lower()
    df["system"] = df["system"].replace({
        "mediapipe_dlc": "freemocap",
        "freemocap_dlc": "freemocap",
        "fmc": "freemocap",
        "qualisys": "qualisys",
        "qtm": "qualisys",
    })

    metric_order = [
        "stride_duration",
        "stance_duration",
        "swing_duration",
        "stance_pct",
        "swing_pct",
    ]

    right = df[df["side"] == "right"].copy()
    right["condition"] = pd.Categorical(
        right["condition"],
        categories=ordered_conditions,
        ordered=True,
    )

    # ---- compute deltas from neutral ----
    deltas = []
    for metric in metric_order:
        sub = right[right["metric"] == metric]
        neutral_map = (
            sub[sub["condition"] == "neutral"]
            .set_index("system")["mean"]
            .to_dict()
        )

        for _, row in sub.iterrows():
            deltas.append({
                "system": row["system"],
                "metric": metric,
                "condition": row["condition"],
                "delta": row["mean"] - neutral_map[row["system"]],
                "std": row["std"],
                "n_valid": row["n_valid"],
            })

    delta_df = pd.DataFrame(deltas)
    delta_df["condition"] = pd.Categorical(
        delta_df["condition"],
        categories=ordered_conditions,
        ordered=True,
    )

    system_colors = {"qualisys": QUAL_COLOR, "freemocap": FMC_COLOR}

    # ---- nice labels for conditions (angles) ----
    cond_label_map = {
        "neg_5_6": r"−5.6°",
        "neg_2_8": r"−2.8°",
        "neutral": r"0°",
        "pos_2_8": r"+2.8°",
        "pos_5_6": r"+5.6°",
    }
    xtick_labels = [cond_label_map.get(c, c) for c in ordered_conditions]

    # ---- figure layout ----
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.flatten()
    panel_labels = ["A", "B", "C", "D", "E"]

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        sub = delta_df[delta_df["metric"] == metric].sort_values("condition")

        for system in ["qualisys", "freemocap"]:
            s = sub[sub["system"] == system]
            if s.empty:
                continue

            ci = 1.96 * (s["std"] / np.sqrt(s["n_valid"]))

            ax.errorbar(
                s["condition"], s["delta"],
                yerr=ci,
                color=system_colors[system],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=6,
                capsize=4,
                alpha=0.9,
                label=system if i == 0 else None,
            )

        # zero line
        ax.axhline(0, color="0.3", linewidth=1, alpha=0.4)

        # title with nicer text
        pretty_name = metric.replace("_", " ")
        ax.set_title(pretty_name)

        # y-axis label with units
        if "pct" in metric:
            ax.set_ylabel("Δ from neutral (% gait cycle)")
        else:
            ax.set_ylabel("Δ from neutral (s)")

        # grid + minimal spines
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # panel label (A, B, C, …)
        ax.text(
            0.02, 0.95, panel_labels[i],
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold",
        )

    # turn off unused 6th axis
    axes[-1].axis("off")

    # x-axis ticks/labels only on bottom row
    for ax in axes[3:5]:
        ax.set_xlabel("Ankle flexion angle")
        ax.set_xticks(range(len(ordered_conditions)))
        ax.set_xticklabels(xtick_labels)

    # one legend for entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower right",
        ncol=2,
        frameon=False,
        title="System",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(
        "Right-side gait parameter changes relative to neutral alignment",
        y=0.98, fontsize=14,
    )
    plt.show()

plot_right_side_deltas(df, ordered_conditions)