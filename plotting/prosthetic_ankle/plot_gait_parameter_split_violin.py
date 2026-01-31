from pathlib import Path
import pandas as pd

conditions = {
    "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
}

ordered_conditions = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]

def load_all_gait_metrics(conditions: dict) -> pd.DataFrame:
    rows = []
    for cond, folder in conditions.items():
        folder = Path(folder)

        fmc_path = folder / "validation/mediapipe_dlc/gait_parameters/gait_metrics.csv"
        q_path   = folder / "validation/qualisys/gait_parameters/qualisys_gait_metrics.csv"

        if fmc_path.exists():
            fmc = pd.read_csv(fmc_path)
            fmc["condition"] = cond
            rows.append(fmc)

        if q_path.exists():
            q = pd.read_csv(q_path)
            q["condition"] = cond
            rows.append(q)

    dfm = pd.concat(rows, ignore_index=True)

    # normalize system labels
    dfm["system"] = dfm["system"].str.lower().replace({
        "mediapipe_dlc": "freemocap",
        "freemocap_dlc": "freemocap",
        "fmc": "freemocap",
        "qualisys": "qualisys",
        "qtm": "qualisys",
    })

    dfm["condition"] = pd.Categorical(
        dfm["condition"], categories=ordered_conditions, ordered=True
    )
    return dfm

gait_metrics_df = load_all_gait_metrics(conditions)
print(gait_metrics_df.head())


import matplotlib.pyplot as plt
import seaborn as sns

QUAL_COLOR = "#d62728"
FMC_COLOR  = "#1f77b4"

def plot_right_side_violin_panels(dfm: pd.DataFrame, ordered_conditions):
    # right side only
    right = dfm[dfm["side"] == "right"].copy()

    metric_order = [
        "stride_duration",
        "stance_duration",
        "swing_duration",
        "stance_pct",
        "swing_pct",
    ]

    # nice x tick labels (angles)
    cond_label_map = {
        "neg_5_6": r"−5.6°",
        "neg_2_8": r"−2.8°",
        "neutral": r"0°",
        "pos_2_8": r"+2.8°",
        "pos_5_6": r"+5.6°",
    }
    xtick_labels = [cond_label_map.get(c, c) for c in ordered_conditions]

    sns.set(style="whitegrid")
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

    palette = {"qualisys": QUAL_COLOR, "freemocap": FMC_COLOR}

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        sub = right[right["metric"] == metric]

        if sub.empty:
            ax.axis("off")
            continue

        sns.violinplot(
            data=sub,
            x="condition",
            y="value",
            hue="system",
            split=True,
            palette=palette,
            cut=0,
            inner="quartile",
            linewidth=1,
            ax=ax,
        )

        pretty = metric.replace("_", " ")
        ax.set_title(pretty)

        if "pct" in metric:
            ax.set_ylabel("Value (% gait cycle)")
        else:
            ax.set_ylabel("Value (s)")

        ax.set_xlabel("Ankle flexion angle")
        ax.set_xticks(range(len(ordered_conditions)))
        ax.set_xticklabels(xtick_labels)

        # clean up legend: only keep on first axis
        if i == 0:
            ax.legend(title="System", frameon=False, loc="upper left")
        else:
            ax.get_legend().remove()

        # panel label
        ax.text(
            0.02, 0.95, panel_labels[i],
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold",
        )

    # turn off unused axis
    axes[-1].axis("off")

    fig.tight_layout()
    fig.suptitle("Right-side per-stride gait metrics across ankle flexion conditions",
                 y=1.02, fontsize=14)
    plt.show()
plot_right_side_violin_panels(gait_metrics_df, ordered_conditions)