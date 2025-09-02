
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_speed_from_folder(name: str) -> float:
    m = re.match(r"^speed_(\d+)_(\d+)$", name)
    if not m:
        return float("nan")
    whole, frac = m.groups()
    return float(f"{whole}.{frac}")


def find_speed_folders(metrics_dir: Path):
    for p in sorted(metrics_dir.iterdir()):
        if p.is_dir() and p.name.startswith("speed_"):
            speed = parse_speed_from_folder(p.name)
            if not np.isnan(speed):
                yield speed, p


def load_one_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    axis_map = {
        "x": "x", "x_error": "x", "x_err": "x",
        "y": "y", "y_error": "y", "y_err": "y",
        "z": "z", "z_error": "z", "z_err": "z",
    }
    df["axis"] = df["coordinate"].str.strip().str.lower().map(axis_map)
    df = df.dropna(subset=["axis"]).copy()
    if "dimension" in df.columns:
        mask = df["dimension"].str.contains("per joint", case=False, na=False)
        if mask.any():
            df = df.loc[mask].copy()
    df["RMSE"] = pd.to_numeric(df["rmse"], errors="coerce")
    df = df.dropna(subset=["RMSE"])
    df = df.rename(columns={"keypoint": "joint"})
    return df[["joint", "axis", "RMSE"]]


def aggregate_metrics(metrics_dir: Path, csv_name: str) -> pd.DataFrame:
    rows = []
    for speed, folder in find_speed_folders(metrics_dir):
        csv_path = folder / csv_name
        if not csv_path.exists():
            print(f"[WARN] Missing CSV at {csv_path}, skipping.")
            continue
        try:
            df = load_one_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Failed to parse {csv_path}: {e}")
            continue
        df = df.assign(speed=float(speed))
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No CSVs found under {metrics_dir} for '{csv_name}'")
    out = pd.concat(rows, ignore_index=True)
    out["axis"] = pd.Categorical(out["axis"], categories=["x","y","z"], ordered=True)
    out = out.sort_values(["joint","axis","speed"]).reset_index(drop=True)
    return out


def plot_per_joint_lines(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    joints = df["joint"].unique()
    for joint in joints:
        sub = df[df["joint"] == joint]
        plt.figure()
        for axis in ["x","y","z"]:
            ss = sub[sub["axis"]==axis].sort_values("speed")
            plt.plot(ss["speed"], ss["RMSE"], marker="o", label=axis.upper())
        plt.xlabel("Treadmill speed (m/s)")
        plt.ylabel("RMSE")
        plt.title(f"{joint}: RMSE vs Speed")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = outdir / f"rmse_trend_{joint.replace(' ','_')}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_axis_heatmaps(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    joints = df["joint"].unique()
    speeds = sorted(df["speed"].unique())

    # Build pivots and compute global color limits
    pivots = {}
    vmin, vmax = None, None
    for axis in ["x","y","z"]:
        sub = df[df["axis"] == axis]
        pivot = sub.pivot_table(index="joint", columns="speed", values="RMSE", aggfunc="mean")
        pivot = pivot.reindex(index=joints, columns=speeds)
        pivots[axis] = pivot
        cur_min = float(pivot.min().min()) if not pivot.empty else float("inf")
        cur_max = float(pivot.max().max()) if not pivot.empty else float("-inf")
        vmin = cur_min if vmin is None else min(vmin, cur_min)
        vmax = cur_max if vmax is None else max(vmax, cur_max)

    # Create a single figure with 3 side-by-side heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(joints) * 0.35)), sharey=True)
    for ax, axis in zip(axes, ["x","y","z"]):
        pivot = pivots[axis]
        im = ax.imshow(pivot.values, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{axis.upper()} axis")
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([str(s) for s in speeds])
        if axis == "x":
            ax.set_yticks(range(len(joints)))
            ax.set_yticklabels(joints)
            ax.set_ylabel("Joint")
        else:
            ax.set_yticks([])
        ax.set_xlabel("Speed (m/s)")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", label="RMSE")
    fig.suptitle("RMSE Heatmaps Across Axes", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(outdir / "rmse_heatmaps_all_axes.png", dpi=200)
    plt.close()

def categorize_joint(joint_name: str) -> str:
    """Heuristic categorization into 'upper', 'lower', or 'other' based on joint name."""
    j = joint_name.lower()
    upper_kw = ["shoulder", "elbow", "wrist", "hand", "head", "neck", "clav", "thorax", "upperarm", "forearm", "spine", "chest"]
    lower_kw = ["hip", "knee", "ankle", "heel", "toe", "foot", "pelvis", "thigh", "shank", "leg"]
    if any(k in j for k in upper_kw):
        return "upper"
    if any(k in j for k in lower_kw):
        return "lower"
    return "other"


def plot_group_trends(df: pd.DataFrame, out_png: Path, group_name: str):
    """
    One figure with 3 horizontal subplots (X/Y/Z). Each subplot: lines per JOINT in the group.
    All subplots share the same Y axis limits for comparability.
    """
    sub = df.copy()
    sub["group"] = sub["joint"].map(categorize_joint)
    sub = sub[sub["group"] == group_name]
    if sub.empty:
        print(f"[WARN] No joints found for group '{group_name}'. Skipping {out_png.name}.")
        return

    speeds = sorted(sub["speed"].unique())
    joints = sorted(sub["joint"].unique())

    # y-limits shared across axes for this group
    y_min = float(sub["RMSE"].min())
    y_max = float(sub["RMSE"].max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, axis in zip(axes, ["x","y","z"]):
        dfa = sub[sub["axis"] == axis]
        for joint in joints:
            line = dfa[dfa["joint"] == joint].sort_values("speed")
            if not line.empty:
                ax.plot(line["speed"], line["RMSE"], marker="o", label=joint)
        ax.set_title(f"{group_name.capitalize()} — {axis.upper()}")
        ax.set_xlabel("Speed (m/s)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("RMSE")
    # Single legend outside the right edge
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"RMSE vs Speed — {group_name.capitalize()} (shared Y across axes)")
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(out_png, dpi=200)
    plt.close()

if __name__ == "__main__":
    # EDIT THESE VARIABLES:
    recording_path = Path(r'D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1')
    recording_metrics_dir = recording_path/'validation'/'mediapipe'/'metrics'

    csv_name = "3d_xyz_position_rmse.csv"
    outdir = recording_metrics_dir / "summary_plots"

    outdir.mkdir(parents=True, exist_ok=True)
    df = aggregate_metrics(recording_metrics_dir, csv_name)
    df.to_csv(outdir / "aggregated_rmse_by_speed.csv", index=False)
    plot_per_joint_lines(df, outdir / "per_joint_trends")
    plot_axis_heatmaps(df, outdir / "heatmaps")
    # New grouped trend plots (two total):
    plot_group_trends(df, (outdir / "upper_body_trends.png"), "upper")
    plot_group_trends(df, (outdir / "lower_body_trends.png"), "lower")
    print(f"Done. Results in {outdir}")