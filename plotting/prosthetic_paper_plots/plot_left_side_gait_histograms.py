from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

QUALISYS_REL = Path("validation/qualisys/gait_parameters/qualisys_gait_metrics.csv")
MP_DLC_REL   = Path("validation/mediapipe_dlc/gait_parameters/gait_metrics.csv")
RTMPOSE_REL  = Path("validation/rtmpose/gait_parameters/gait_metrics.csv")


def load_gait_dataframe_multi(
    folder_list: list[Path],
    qualisys_rel: Path = QUALISYS_REL,
    fmc_system_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """
    Loads ONLY sessions that contain ALL required systems.
    If any system is missing, the entire session is skipped.
    """
    if fmc_system_paths is None:
        fmc_system_paths = {
            "mediapipe_dlc": MP_DLC_REL,
            "rtmpose": RTMPOSE_REL,
        }

    rows = []
    skipped = []

    for folder in folder_list:
        required_paths = {
            "qualisys": folder / qualisys_rel,
            **{k: folder / v for k, v in fmc_system_paths.items()},
        }

        missing = [k for k, p in required_paths.items() if not p.exists()]
        if missing:
            skipped.append((folder.stem, missing))
            continue

        # load qualisys
        qtmp = pd.read_csv(required_paths["qualisys"])
        qtmp["trial_name"] = folder.stem
        rows.append(qtmp)

        # load fmc systems, force system name (important if both say "freemocap")
        for sys_name, fpath in required_paths.items():
            if sys_name == "qualisys":
                continue
            tmp = pd.read_csv(fpath)
            tmp["system"] = sys_name
            tmp["trial_name"] = folder.stem
            rows.append(tmp)

    if not rows:
        raise RuntimeError("No valid sessions found with complete system coverage.")

    if skipped:
        print("Skipped sessions (missing required systems):")
        for name, systems in skipped:
            print(f"  {name}: missing {systems}")

    df = pd.concat(rows, ignore_index=True)

    required_cols = {"system", "side", "metric", "event_index", "value", "trial_name"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Combined dataframe missing columns: {missing_cols}")

    df["event_index"] = df["event_index"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def get_paired_stride_df(
    df: pd.DataFrame,
    metric: str,
    side: str = "left",
    reference_system: str = "qualisys",
    fmc_system: str = "mediapipe_dlc",
) -> pd.DataFrame:
    d = df[(df["metric"] == metric) & (df["side"] == side)].copy()

    systems = sorted(d["system"].dropna().unique().tolist())
    if reference_system not in systems:
        raise ValueError(f"Reference system '{reference_system}' not found in data systems: {systems}")
    if fmc_system not in systems:
        raise ValueError(f"FMC system '{fmc_system}' not found in data systems: {systems}")

    ref = d[d["system"] == reference_system].rename(columns={"value": reference_system})[
        ["trial_name", "side", "metric", "event_index", reference_system]
    ]
    fmc = d[d["system"] == fmc_system].rename(columns={"value": "freemocap"})[
        ["trial_name", "side", "metric", "event_index", "freemocap"]
    ]

    paired = ref.merge(fmc, on=["trial_name", "side", "metric", "event_index"], how="inner")
    paired["diff"] = paired["freemocap"] - paired[reference_system]
    return paired


def plot_histogram_overlay(
    dfs: dict[str, pd.DataFrame],
    colors: dict[str, str],
    plot_title: str,
    ax=None,
    max_frames: int = 50,
    show_ylabel: bool = True,
    fps: int = 30,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    frame_dt = 1 / fps
    max_abs = max_frames * frame_dt
    ax.set_xlim(-max_abs, max_abs)

    max_ms = max_abs * 1000
    major_ticks_ms = np.arange(-2000, 2000, 200)
    major_ticks_ms = major_ticks_ms[np.abs(major_ticks_ms) <= max_ms]
    major_ticks = major_ticks_ms / 1000

    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1000:+.0f}" if v != 0 else "0"))
    ax.tick_params(axis="x", labelrotation=0)

    bin_width = frame_dt
    edges = np.arange(-max_abs - bin_width / 2, max_abs + bin_width, bin_width)

    stats_lines = []
    for label, dfi in dfs.items():
        x = dfi["diff"].to_numpy()
        x = x[np.isfinite(x)]

        ax.hist(x, bins=edges, edgecolor=None, color=colors[label], alpha=0.45, label=label)
        ax.hist(x, bins=edges, histtype="step", linewidth=1.2, color=colors[label])

        bias = np.mean(x) * 1000
        sd = np.std(x, ddof=1) * 1000
        stats_lines.append(f"{label}: $\\mu$={bias:+.0f} ms, $\\sigma$={sd:.0f} ms")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)
    ax.set_title(plot_title, fontweight="bold")

    if show_ylabel:
        ax.set_ylabel("Count", fontweight="bold")

    ax.set_xlabel("Error (ms)", fontweight="bold")
    ax.text(
        0.03, 0.95,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    return ax


# -------------------------
# RUN
# -------------------------
path_to_recordings = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")

list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]

list_of_valid_folders = []
for p in list_of_folders:
    if (p / "validation").is_dir():
        list_of_valid_folders.append(p)
    else:
        print(f"Skipping {p}")

gait_param_df = load_gait_dataframe_multi(
    list_of_valid_folders,
    fmc_system_paths={
        "mediapipe_dlc": MP_DLC_REL,
        "rtmpose": RTMPOSE_REL,
    },
)

SIDE = "left"

paired_stance_mp = get_paired_stride_df(gait_param_df, metric="stance_duration", side=SIDE, fmc_system="mediapipe_dlc")
paired_stance_rt = get_paired_stride_df(gait_param_df, metric="stance_duration", side=SIDE, fmc_system="rtmpose")

paired_swing_mp  = get_paired_stride_df(gait_param_df, metric="swing_duration",  side=SIDE, fmc_system="mediapipe_dlc")
paired_swing_rt  = get_paired_stride_df(gait_param_df, metric="swing_duration",  side=SIDE, fmc_system="rtmpose")

colors = {
    "mediapipe_dlc": "#1f77b4",
    "rtmpose": "#ff7f0e",
}

fig_hist, axes_hist = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)

plot_histogram_overlay(
    dfs={"mediapipe_dlc": paired_stance_mp, "rtmpose": paired_stance_rt},
    colors=colors,
    plot_title="Stance Duration",
    ax=axes_hist[0],
    show_ylabel=True,
)

plot_histogram_overlay(
    dfs={"mediapipe_dlc": paired_swing_mp, "rtmpose": paired_swing_rt},
    colors=colors,
    plot_title="Swing Duration",
    ax=axes_hist[1],
    show_ylabel=False,
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()
