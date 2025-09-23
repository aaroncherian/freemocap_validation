#!/usr/bin/env python3
"""
Aggregate per-joint RMSE across MULTIPLE recordings (trials) and plot mean ± std.

Folder layout for each recording (example):
<recording_root>/validation/mediapipe/metrics/
    speed_0_5/3d_xyz_position_rmse.csv
    speed_1_0/3d_xyz_position_rmse.csv
    ...

This script:
- Takes a LIST of metrics directories (one per recording/trial).
- Aggregates RMSE across trials by (joint, axis, speed).
- Saves a tidy CSV with columns: recording, joint, axis, speed, RMSE (raw rows)
  and a summary CSV with: joint, axis, speed, mean, std, n
- Produces two figures (Lower, Upper), each with X/Y/Z horizontally:
  each joint = mean line across trials with a ±1 SD shaded band; handles missing speeds.

Edit the variables in __main__ to your paths & file name.
"""

from pathlib import Path
import re
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- IO & Aggregation ---------------------------

def parse_speed_from_folder(name: str) -> float:
    m = re.match(r"^speed_(\d+)_(\d+)$", name)
    if not m:
        return float("nan")
    return float(f"{m.group(1)}.{m.group(2)}")


def find_speed_folders(metrics_dir: Path) -> Iterable[Tuple[float, Path]]:
    for p in sorted(metrics_dir.iterdir()):
        if p.is_dir() and p.name.startswith("speed_"):
            s = parse_speed_from_folder(p.name)
            if not np.isnan(s):
                yield s, p


def recording_name_from_metrics_dir(metrics_dir: Path) -> str:
    """
    metrics_dir is .../<recording>/validation/mediapipe/metrics
    """
    try:
        return metrics_dir.parent.parent.parent.name
    except Exception:
        return metrics_dir.name


def load_one_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # standardize
    df.columns = [c.strip().lower() for c in df.columns]
    axis_map = {"x":"x","x_err":"x","x_error":"x",
                "y":"y","y_err":"y","y_error":"y",
                "z":"z","z_err":"z","z_error":"z"}
    df["axis"] = df["coordinate"].str.strip().str.lower().map(axis_map)
    df = df.dropna(subset=["axis"]).copy()

    if "dimension" in df.columns:
        mask = df["dimension"].str.contains("per joint", case=False, na=False)
        if mask.any():
            df = df.loc[mask].copy()

    df["RMSE"] = pd.to_numeric(df["rmse"], errors="coerce")
    df = df.dropna(subset=["RMSE"])
    df = df.rename(columns={"keypoint": "joint"})
    return df[["joint","axis","RMSE"]]


def aggregate_from_metrics_dirs(metrics_dirs: List[Path], csv_name: str) -> pd.DataFrame:
    """
    Return long df with cols: recording, joint, axis, speed, RMSE
    """
    rows = []
    for mdir in metrics_dirs:
        rec_name = recording_name_from_metrics_dir(mdir)
        for speed, sdir in find_speed_folders(mdir):
            csv_path = sdir / csv_name
            if not csv_path.exists():
                print(f"[WARN] {rec_name} missing {csv_path.name} at speed {speed}; skipping.")
                continue
            try:
                sub = load_one_csv(csv_path)
            except Exception as e:
                print(f"[WARN] Failed to parse {csv_path}: {e}")
                continue
            sub = sub.assign(recording=rec_name, speed=float(speed))
            rows.append(sub)

    if not rows:
        raise RuntimeError("No data rows found from provided metrics directories. Check paths and csv_name.")
    df = pd.concat(rows, ignore_index=True)
    # enforce category order for axis
    df["axis"] = pd.Categorical(df["axis"], categories=["x","y","z"], ordered=True)
    return df[["recording","joint","axis","speed","RMSE"]]


# --------------------------- Grouping & Plotting ---------------------------

def categorize_joint(joint_name: str) -> str:
    j = joint_name.lower()
    upper_kw = ["shoulder","elbow","wrist","hand","head","neck","clav","thorax","upperarm","forearm","spine","chest"]
    lower_kw = ["hip","knee","ankle","heel","toe","foot","pelvis","thigh","shank","leg"]
    if any(k in j for k in upper_kw):
        return "upper"
    if any(k in j for k in lower_kw):
        return "lower"
    return "other"


def _base_joint_name(j: str) -> str:
    j = j.lower().strip()
    for prefix in ["left_", "right_", "l_", "r_"]:
        if j.startswith(prefix):
            return j[len(prefix):]
    return j


def _lr_side(j: str) -> str:
    jl = j.lower()
    if jl.startswith(("left_", "l_")):
        return "L"
    if jl.startswith(("right_", "r_")):
        return "R"
    return ""


def summarize_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by joint/axis/speed across recordings and compute mean, std, n.
    """
    g = (df.groupby(["joint","axis","speed"])["RMSE"]
           .agg(mean="mean", std="std", n="count")
           .reset_index())
    return g


def plot_group_mean_std(summary: pd.DataFrame, out_png: Path, group_name: str,
                        include_only: List[str] | None = None):
    """
    One figure: 3 horizontal subplots (X/Y/Z).
    Each subplot plots per-JOINT mean across trials, with ±1 SD fill (if n>=2).
    Handles missing speeds naturally (curves may have gaps).
    """
    # assign group
    summary = summary.copy()
    summary["group"] = summary["joint"].map(categorize_joint)
    summary = summary[summary["group"] == group_name]
    if include_only:
        summary = summary[summary["joint"].map(_base_joint_name).isin([b.lower() for b in include_only])]

    if summary.empty:
        print(f"[WARN] No joints to plot for group '{group_name}'.")
        return

    speeds = sorted(summary["speed"].unique())
    joints = sorted(summary["joint"].unique())
    bases = sorted({_base_joint_name(j) for j in joints})

    # consistent colors by base joint, linestyle by side
    cmap = plt.get_cmap("tab10")
    base_to_color = {b: cmap(i % 10) for i, b in enumerate(bases)}

    # shared y-limits (per group, across axes)
    y_min = float(summary["mean"].min(skipna=True))
    y_max = float(summary["mean"].max(skipna=True))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, axis in zip(axes, ["x","y","z"]):
        dfa = summary[summary["axis"] == axis]
        for joint in joints:
            s = dfa[dfa["joint"] == joint].sort_values("speed")
            if s.empty:
                continue
            base = _base_joint_name(joint)
            side = _lr_side(joint)
            ls = "-" if side in ("","L") else "--"
            marker = "o" if side in ("","L") else "s"
            color = base_to_color.get(base, None)

            ax.plot(s["speed"], s["mean"], linestyle=ls, marker=marker,
                    linewidth=2.2, alpha=0.95, color=color)

            # std band only if n>=2; allow NaN std
            if (s["n"] >= 2).any():
                ylo = s["mean"] - s["std"]
                yhi = s["mean"] + s["std"]
                ax.fill_between(s["speed"], ylo, yhi, alpha=0.15, linewidth=0, color=color)

            # end label
            try:
                x_last = s["speed"].iloc[-1]
                y_last = s["mean"].iloc[-1]
                ax.text(x_last + 0.02, y_last, joint, fontsize=8, va="center", color=color)
            except Exception:
                pass

        ax.set_title(f"{group_name.capitalize()} — {axis.upper()}")
        ax.set_xlabel("Speed (m/s)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("RMSE (mean ± SD)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# --------------------------- Main ---------------------------

if __name__ == "__main__":
    # >>> EDIT THESE: list your recordings' metrics directories here <<<
    RECORDING_METRICS_DIRS = [
        Path(r"D:\2025_09_03_OKK\freemocap\2025-09-03_14-56-30_GMT-4_okk_treadmill_1\validation\mediapipe\metrics"),
        Path(r"D:\2025_09_03_OKK\freemocap\2025-09-03_15-04-04_GMT-4_okk_treadmill_2\validation\mediapipe\metrics"),
    ]
    save_path = Path(r"D:\2025_09_03_OKK\freemocap")
    CSV_NAME = "3d_xyz_position_rmse.csv"

    # Where to write outputs (a sibling folder next to the first metrics dir by default)
    OUTDIR = save_path / "summary_across_trials"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) Load & combine all raw rows
    raw = aggregate_from_metrics_dirs(RECORDING_METRICS_DIRS, CSV_NAME)
    raw.to_csv(OUTDIR / "raw_rmse_rows_all_trials.csv", index=False)

    # 2) Summarize across trials (mean, std, n)
    summary = summarize_mean_std(raw)
    summary.to_csv(OUTDIR / "rmse_summary_mean_std_by_joint_axis_speed.csv", index=False)

    # 3) Plots (mean ± std) for groups
    plot_group_mean_std(summary, OUTDIR / "lower_body_mean_std.png", "lower")
    plot_group_mean_std(summary, OUTDIR / "upper_body_mean_std.png", "upper")

    print(f"Done. CSVs and figures written to: {OUTDIR}")
