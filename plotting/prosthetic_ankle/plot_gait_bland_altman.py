from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

QUALISYS_REL = Path("validation/qualisys/gait_parameters/qualisys_gait_metrics.csv")
FMC_REL      = Path("validation/mediapipe_dlc/gait_parameters/gait_metrics.csv")

def load_gait_long_df(
    conditions: dict,
    qualisys_rel: Path = QUALISYS_REL,
    fmc_rel: Path = FMC_REL,
) -> pd.DataFrame:
    """
    Returns one long dataframe with columns:
    condition, system, side, metric, event_index, value
    """
    rows = []

    for condition, folder in conditions.items():
        folder = Path(folder)

        for fpath in [folder / qualisys_rel, folder / fmc_rel]:
            if not fpath.exists():
                raise FileNotFoundError(f"Missing CSV: {fpath}")

            tmp = pd.read_csv(fpath)
            # minimal sanity check
            required = {"system", "side", "metric", "event_index", "value"}
            missing = required - set(tmp.columns)
            if missing:
                raise ValueError(f"{fpath} missing columns: {missing}")

            tmp["condition"] = condition
            rows.append(tmp)

    df_long = pd.concat(rows, ignore_index=True)

    # consistent dtypes
    df_long["event_index"] = df_long["event_index"].astype(int)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    return df_long


def get_paired_stride_df(
    df_long: pd.DataFrame,
    metric: str,
    side: str = "right",
    qualisys_system: str = "qualisys",
    fmc_system: str | None = None,   # if None, we auto-detect the non-qualisys system
) -> pd.DataFrame:
    """
    Returns stride-level paired dataframe aligned by condition+side+metric+event_index:
    condition, side, metric, event_index, qualisys, freemocap, mean, diff
    where diff = freemocap - qualisys
    """
    d = df_long[(df_long["metric"] == metric) & (df_long["side"] == side)].copy()

    systems = sorted(d["system"].dropna().unique().tolist())
    if qualisys_system not in systems:
        raise ValueError(f"Expected qualisys_system='{qualisys_system}', found systems={systems}")

    if fmc_system is None:
        # pick the "other" system name
        others = [s for s in systems if s != qualisys_system]
        if len(others) != 1:
            raise ValueError(f"Could not auto-detect FMC system. systems={systems}. Pass fmc_system=... explicitly.")
        fmc_system = others[0]

    q = d[d["system"] == qualisys_system].rename(columns={"value": "qualisys"})[
        ["condition", "side", "metric", "event_index", "qualisys"]
    ]
    f = d[d["system"] == fmc_system].rename(columns={"value": "freemocap"})[
        ["condition", "side", "metric", "event_index", "freemocap"]
    ]

    paired = q.merge(f, on=["condition", "side", "metric", "event_index"], how="inner")
    paired["mean"] = (paired["qualisys"] + paired["freemocap"]) / 2.0
    paired["diff"] = paired["freemocap"] - paired["qualisys"]

    return paired

def bland_altman_stats(df: pd.DataFrame) -> dict:
    bias = df["diff"].mean()
    sd = df["diff"].std(ddof=1)
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd

    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "n": len(df),
    }

def plot_bland_altman(df, title, ax=None, jitter_x=0.0005, jitter_y=0.0005, seed=0):
    stats = bland_altman_stats(df)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    x = df["mean"].to_numpy()
    y = df["diff"].to_numpy()

    rng = np.random.default_rng(seed)   # reproducible
    xj = x + rng.normal(0, jitter_x, size=len(x))
    yj = y + rng.normal(0, jitter_y, size=len(y))

    ax.scatter(xj, yj, s=12, alpha=0.35)

    ax.axhline(stats["bias"], color="black", linestyle="--", label=f"Bias = {stats['bias']:.3f}")
    ax.axhline(stats["loa_upper"], color="red", linestyle="--", label=f"+1.96 SD = {stats['loa_upper']:.3f}")
    ax.axhline(stats["loa_lower"], color="red", linestyle="--", label=f"-1.96 SD = {stats['loa_lower']:.3f}")

    ax.set_xlabel("Mean of systems")
    ax.set_ylabel("FreeMoCap − Qualisys")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax, stats


conditions = {
    "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
}

df_long = load_gait_long_df(conditions)

# see what the FMC system string is called in YOUR csvs
print(df_long["system"].unique())

paired_stance = get_paired_stride_df(df_long, metric="stance_duration", side="right")
paired_swing  = get_paired_stride_df(df_long, metric="swing_duration", side="right")
paired_pct    = get_paired_stride_df(df_long, metric="stance_pct", side="right")  # if that’s the exact metric name

conditions = paired_pct["condition"].unique()

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(conditions),
    figsize=(4 * len(conditions), 4),
    sharey=True,
)

for ax, cond in zip(axes, conditions):
    d = paired_pct[paired_pct["condition"] == cond]
    plot_bland_altman(
        d,
        title=cond,
        ax=ax,
    )

fig.suptitle("Bland–Altman: Stance Duration (Right)", y=1.05)
plt.tight_layout()
plt.show()


# def plot_ba_pooled_by_condition(df, title):
#     # df has columns: mean, diff, condition
#     stats = bland_altman_stats(df)

#     fig, ax = plt.subplots(figsize=(6.5, 5))

#     for cond, d in df.groupby("condition", sort=False):
#         ax.scatter(d["mean"], d["diff"], s=12, alpha=0.35, label=cond)

#     ax.axhline(stats["bias"], color="black", linestyle="--", label=f"Bias = {stats['bias']:.3f}")
#     ax.axhline(stats["loa_upper"], color="red", linestyle="--", label=f"+1.96 SD = {stats['loa_upper']:.3f}")
#     ax.axhline(stats["loa_lower"], color="red", linestyle="--", label=f"-1.96 SD = {stats['loa_lower']:.3f}")

#     ax.set_xlabel("Mean of systems")
#     ax.set_ylabel("FreeMoCap − Qualisys")
#     ax.set_title(title)

#     # keep the legend compact
#     ax.legend(frameon=False, ncol=2, fontsize=8)
#     plt.tight_layout()
#     return fig, ax, stats

# plot_ba_pooled_by_condition(paired_stance, "Bland–Altman: Stance % (Right), all conditions")
# plt.show()


f = 2