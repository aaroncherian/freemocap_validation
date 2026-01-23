from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COORD_TO_AXIS = {"x_error": "X", "y_error": "Y", "z_error": "Z"}
AXIS_ORDER = ["X", "Y", "Z"]


# -----------------------------
# IO / parsing
# -----------------------------
def _read_position_rmse_csv(csv_path: Path) -> pd.DataFrame:
    """
    Expected columns (like your example):
      - keypoint
      - coordinate: x_error / y_error / z_error / All
      - RMSE
    """
    df = pd.read_csv(csv_path)

    required = {"keypoint", "coordinate", "RMSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    df = df[df["coordinate"].isin(COORD_TO_AXIS)].copy()  # drop "All" etc
    df["axis"] = df["coordinate"].map(COORD_TO_AXIS)
    df["rmse"] = df["RMSE"].astype(float)
    return df[["keypoint", "axis", "rmse"]]


def load_all_position_rmses(recording_session: Path, trackers: list[str]) -> pd.DataFrame:
    """
    recording_session/validation/<tracker>/rmse/position/<condition>/position_rmse.csv

    Returns tidy df:
      condition | tracker | keypoint | axis | rmse
    """
    rows = []
    for tracker in trackers:
        base = recording_session / "validation" / tracker / "rmse" / "position"
        if not base.exists():
            print(f"[WARN] Missing path: {base}")
            continue

        for cond_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            csv_path = cond_dir / "position_rmse.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing file: {csv_path}")
                continue

            df = _read_position_rmse_csv(csv_path)
            df["condition"] = cond_dir.name
            df["tracker"] = tracker
            rows.append(df)

    if not rows:
        raise RuntimeError("No RMSE CSVs found with the expected folder/file layout.")

    out = pd.concat(rows, ignore_index=True)
    out["axis"] = pd.Categorical(out["axis"], categories=AXIS_ORDER, ordered=True)
    return out


# -----------------------------
# Ordering helpers
# -----------------------------
def _anatomicalish_order(joints: list[str]) -> list[str]:
    """
    Simple heuristic. Tweak to your keypoint naming.
    """
    groups = [
        ("head", ["head", "nose", "eye", "ear"]),
        ("torso", ["neck", "thorax", "chest", "spine", "pelvis", "hip", "shoulder"]),
        ("arms", ["elbow", "wrist", "hand"]),
        ("legs", ["knee", "ankle", "heel", "foot", "toe", "index"]),
    ]

    def score(j: str):
        jl = j.lower()
        g_idx = 99
        sub = 99
        for i, (_, keys) in enumerate(groups):
            for k in keys:
                if k in jl:
                    g_idx = min(g_idx, i)
                    sub = min(sub, keys.index(k))
        # Left before right (optional)
        side = 0 if jl.startswith("left") else (1 if jl.startswith("right") else 2)
        return (g_idx, side, sub, jl)

    return sorted(joints, key=score)


def compute_joint_order(
    rmse_df: pd.DataFrame,
    condition: str,
    trackers: list[str],
    order_mode: str = "mean",  # "mean" | "spread" | "alpha" | "anatomical"
    drop_all: bool = True,
) -> list[str]:
    d = rmse_df[rmse_df["condition"] == condition].copy()
    if drop_all:
        d = d[d["keypoint"].str.lower() != "all"]

    joints = sorted(d["keypoint"].unique())
    if order_mode == "alpha":
        return joints
    if order_mode == "anatomical":
        return _anatomicalish_order(joints)

    # pivot: joint x (tracker, axis)
    piv = d.pivot_table(index="keypoint", columns=["tracker", "axis"], values="rmse", aggfunc="mean")
    # make sure we only consider requested trackers
    piv = piv.reindex(columns=pd.MultiIndex.from_product([trackers, AXIS_ORDER]), copy=False)

    if order_mode == "mean":
        s = piv.mean(axis=1, skipna=True)  # mean over trackers+axes
        return list(s.sort_values(ascending=False).index)

    if order_mode == "spread":
        # spread computed per-axis then averaged (more stable than raw max-min over all columns)
        spreads = []
        for ax in AXIS_ORDER:
            sub = piv.xs(ax, axis=1, level=1)  # joint x tracker
            spreads.append((sub.max(axis=1, skipna=True) - sub.min(axis=1, skipna=True)))
        s = pd.concat(spreads, axis=1).mean(axis=1, skipna=True)
        return list(s.sort_values(ascending=False).index)

    raise ValueError(f"Unknown order_mode={order_mode}")


# -----------------------------
# Plot: absolute RMSE heatmaps with dropdown across conditions
# -----------------------------
def plot_rmse_heatmaps_dropdown(
    rmse_df: pd.DataFrame,
    trackers: list[str],
    *,
    order_mode: str = "mean",
    scale_mode: str = "global",  # "global" | "per_axis"
    drop_all: bool = True,
    height: int = 1200,
    width: int = 2200,
    title: str = "Position RMSE by joint and tracker",
) -> go.Figure:
    conditions = sorted(rmse_df["condition"].unique())

    # Build one set of traces per condition (3 heatmaps: X,Y,Z)
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        column_titles=[f"{ax} RMSE" for ax in AXIS_ORDER],
    )

    all_traces = []
    for cond in conditions:
        joint_order = compute_joint_order(rmse_df, cond, trackers, order_mode=order_mode, drop_all=drop_all)

        # pivots per axis
        pivots = {}
        for ax in AXIS_ORDER:
            d = rmse_df[(rmse_df["condition"] == cond) & (rmse_df["axis"] == ax)].copy()
            if drop_all:
                d = d[d["keypoint"].str.lower() != "all"]
            piv = d.pivot_table(index="keypoint", columns="tracker", values="rmse", aggfunc="mean")
            piv = piv.reindex(index=joint_order, columns=trackers)
            pivots[ax] = piv

        if scale_mode == "global":
            zmin = min(np.nanmin(pivots[ax].values) for ax in AXIS_ORDER)
            zmax = max(np.nanmax(pivots[ax].values) for ax in AXIS_ORDER)
            zmins = {ax: zmin for ax in AXIS_ORDER}
            zmaxs = {ax: zmax for ax in AXIS_ORDER}
        elif scale_mode == "per_axis":
            zmins = {ax: np.nanmin(pivots[ax].values) for ax in AXIS_ORDER}
            zmaxs = {ax: np.nanmax(pivots[ax].values) for ax in AXIS_ORDER}
        else:
            raise ValueError(f"Unknown scale_mode={scale_mode}")

        for i, ax in enumerate(AXIS_ORDER):
            piv = pivots[ax]
            heat = go.Heatmap(
                z=piv.values,
                x=list(piv.columns),
                y=list(piv.index),
                zmin=float(zmins[ax]),
                zmax=float(zmaxs[ax]),
                colorbar=dict(title="RMSE") if (ax == "Z") else None,
                hovertemplate="Condition: %{customdata[0]}<br>Joint: %{y}<br>Tracker: %{x}<br>RMSE: %{z:.3f}<extra></extra>",
                customdata=np.full((piv.shape[0], piv.shape[1], 1), cond),
                visible=(cond == conditions[0]),
            )
            all_traces.append(heat)

    # Add traces to fig in order: cond0 X,Y,Z then cond1 X,Y,Z ...
    trace_idx = 0
    for cond in conditions:
        for c in range(1, 4):
            fig.add_trace(all_traces[trace_idx], row=1, col=c)
            trace_idx += 1

    # Dropdown: toggle visibility blocks of 3 traces
    buttons = []
    for ci, cond in enumerate(conditions):
        vis = [False] * (len(conditions) * 3)
        start = ci * 3
        vis[start:start + 3] = [True, True, True]
        buttons.append(
            dict(
                label=cond,
                method="update",
                args=[{"visible": vis},
                      {"title": f"{title} — condition: {cond} (order={order_mode}, scale={scale_mode})"}],
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=height,
        width=width,
        title=f"{title} — condition: {conditions[0]} (order={order_mode}, scale={scale_mode})",
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.12, xanchor="left", yanchor="top")],
        margin=dict(t=120),
    )
    fig.update_xaxes(title_text="Tracker", row=1, col=1)
    fig.update_xaxes(title_text="Tracker", row=1, col=2)
    fig.update_xaxes(title_text="Tracker", row=1, col=3)
    fig.update_yaxes(title_text="Joint", autorange="reversed", row=1, col=1)

    return fig


# -----------------------------
# Plot: rank heatmaps (robust “agreement on hard joints” view)
# -----------------------------
def plot_rank_heatmaps_dropdown(
    rmse_df: pd.DataFrame,
    trackers: list[str],
    *,
    order_mode: str = "spread",  # ranking view benefits from spread ordering
    drop_all: bool = True,
    height: int = 1200,
    width: int = 2200,
    title: str = "Ranked joint difficulty (by RMSE) — lower is better",
) -> go.Figure:
    """
    Rank within each tracker, per axis, per condition:
      rank 1 = best (lowest RMSE), larger rank = worse.

    Useful when you care about *which joints* are hard rather than absolute units.
    """
    conditions = sorted(rmse_df["condition"].unique())

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        column_titles=[f"{ax} rank" for ax in AXIS_ORDER],
    )

    all_traces = []
    for cond in conditions:
        joint_order = compute_joint_order(rmse_df, cond, trackers, order_mode=order_mode, drop_all=drop_all)

        for ax in AXIS_ORDER:
            d = rmse_df[(rmse_df["condition"] == cond) & (rmse_df["axis"] == ax)].copy()
            if drop_all:
                d = d[d["keypoint"].str.lower() != "all"]

            piv = d.pivot_table(index="keypoint", columns="tracker", values="rmse", aggfunc="mean")
            piv = piv.reindex(index=joint_order, columns=trackers)

            # rank within each tracker (column) across joints
            ranks = piv.rank(axis=0, method="average", ascending=True)  # 1 = lowest RMSE
            zmin, zmax = 1.0, float(np.nanmax(ranks.values))

            heat = go.Heatmap(
                z=ranks.values,
                x=list(ranks.columns),
                y=list(ranks.index),
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title="Rank") if (ax == "Z") else None,
                hovertemplate="Condition: %{customdata[0]}<br>Joint: %{y}<br>Tracker: %{x}<br>Rank: %{z:.1f}<br>RMSE: %{customdata[1]:.3f}<extra></extra>",
                # include RMSE in hover as customdata
                customdata=np.stack(
                    [np.full(ranks.shape, cond), piv.values],
                    axis=2
                ),
                visible=(cond == conditions[0]),
            )
            all_traces.append(heat)

    # add traces
    trace_idx = 0
    for cond in conditions:
        for c in range(1, 4):
            fig.add_trace(all_traces[trace_idx], row=1, col=c)
            trace_idx += 1

    # dropdown
    buttons = []
    for ci, cond in enumerate(conditions):
        vis = [False] * (len(conditions) * 3)
        start = ci * 3
        vis[start:start + 3] = [True, True, True]
        buttons.append(
            dict(
                label=cond,
                method="update",
                args=[{"visible": vis},
                      {"title": f"{title} — condition: {cond} (order={order_mode})"}],
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=height,
        width=width,
        title=f"{title} — condition: {conditions[0]} (order={order_mode})",
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.12, xanchor="left", yanchor="top")],
        margin=dict(t=120),
    )
    fig.update_xaxes(title_text="Tracker", row=1, col=1)
    fig.update_xaxes(title_text="Tracker", row=1, col=2)
    fig.update_xaxes(title_text="Tracker", row=1, col=3)
    fig.update_yaxes(title_text="Joint", autorange="reversed", row=1, col=1)

    return fig


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    recording_session = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1")

    # Qualisys is the reference used to *compute* RMSE, not a tracker here.
    trackers = ["mediapipe", "rtmpose", "vitpose"]

    rmse_df = load_all_position_rmses(recording_session, trackers)

    # Absolute RMSE view (dropdown across conditions)
    fig_abs = plot_rmse_heatmaps_dropdown(
        rmse_df,
        trackers=trackers,
        order_mode="spread",     # try: "mean" vs "spread" vs "anatomical"
        scale_mode="global",     # try: "per_axis"
        drop_all=True,
        height=1200,
        width=2200,
        title="Position RMSE by joint and tracker",
    )
    fig_abs.show()

    # Rank view (dropdown across conditions)
    fig_rank = plot_rank_heatmaps_dropdown(
        rmse_df,
        trackers=trackers,
        order_mode="spread",
        drop_all=True,
        height=1200,
        width=2200,
        title="Ranked joint difficulty (by RMSE) — lower is better",
    )
    fig_rank.show()
