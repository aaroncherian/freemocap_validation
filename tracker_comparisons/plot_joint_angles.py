from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Loading
# -----------------------------
def load_joint_angle_stride_summary_stats(
    recording_session: Path,
    trackers: list[str],
    filename: str = "joint_angles_per_stride_summary_stats.csv",
) -> pd.DataFrame:
    """
    Crawls:
      recording_session/validation/<tracker>/joint_angles/<condition>/<filename>

    Returns tidy df:
      condition | tracker | joint | side | component | percent_gait_cycle | stat | value
    """
    rows = []
    for tracker in trackers:
        base = recording_session / "validation" / tracker / "joint_angles"
        if not base.exists():
            print(f"[WARN] Missing path: {base}")
            continue

        for cond_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            csv_path = cond_dir / filename
            if not csv_path.exists():
                print(f"[WARN] Missing file: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            df["condition"] = cond_dir.name
            df["tracker"] = tracker  # enforce tracker from folder even if csv contains it
            rows.append(df)

    if not rows:
        raise RuntimeError("No joint angle summary stat CSVs found with the expected folder/file layout.")

    out = pd.concat(rows, ignore_index=True)

    # basic normalization
    out["percent_gait_cycle"] = out["percent_gait_cycle"].astype(int)
    out["value"] = out["value"].astype(float)
    return out


DEFAULT_TRACKER_COLORS = {
    "qualisys": "black",
    "mediapipe": "#2ca02c",
    "rtmpose": "#d62728",
    "vitpose_25": "#1f77b4",
    "vitpose_wholebody": "#ff7f0e",
}

JOINT_LABELS = {
    "hip": "Hip",
    "knee": "Knee",
    "ankle": "Ankle",
}

COMPONENT_LABELS = {
    "flex_ext": "Flex/Ext",
    "abd_add": "Abd/Add",
    "int_ext": "Int/Ext",
    "dorsi_plantar": "Dorsi/Plantar",
    "inv_ev": "Inv/Ev",
}

def plot_mean_joint_angles_canonical_grid(
    df_all: pd.DataFrame,
    *,
    side: str = "left",
    condition: str = "overall",
    joints: list[str] = ("ankle", "knee", "hip"),
    # canonical component sets per joint (tweak to taste)
    components_by_joint: dict[str, list[str]] | None = None,
    trackers_order: list[str] | None = None,
    tracker_colors: dict[str, str] | None = None,
    height: int = 1200,
    width: int = 2200,
) -> go.Figure:
    """
    Explicit subplot grid with no empty facet gaps:
      rows = joints
      cols = union of components across joints (or explicitly provided)
    """
    tracker_colors = tracker_colors or DEFAULT_TRACKER_COLORS

    d = df_all.copy()
    d = d[d["stat"].str.lower() == "mean"]
    d = d[d["side"].str.lower() == side.lower()]
    d = d[d["condition"] == condition]
    if d.empty:
        raise ValueError(f"No data after filtering: side={side}, condition={condition}")

    if trackers_order is None:
        trackers_order = sorted(d["tracker"].unique())

    if components_by_joint is None:
        components_by_joint = {
            "hip":   ["flex_ext", "abd_add", "int_ext"],
            "knee":  ["flex_ext"],  # usually only flex/ext is meaningful
            "ankle": ["dorsi_plantar", "inv_ev", "int_ext"],
        }

    # keep only requested joints
    d = d[d["joint"].isin(joints)].copy()

    # choose columns: keep union, but preserve a nice order
    preferred_col_order = ["flex_ext", "abd_add", "int_ext", "dorsi_plantar", "inv_ev"]
    col_components = []
    for comp in preferred_col_order:
        if any(comp in components_by_joint.get(j, []) for j in joints):
            col_components.append(comp)

    # subplot titles (col titles only; row titles handled by y-axis annotation)
    col_titles = [COMPONENT_LABELS.get(c, c) for c in col_components]

    fig = make_subplots(
        rows=len(joints),
        cols=len(col_components),
        shared_xaxes=True,
        shared_yaxes=False,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
        column_titles=[f"{t} (deg)" for t in col_titles],
        row_titles=[JOINT_LABELS.get(j, j) for j in joints],
    )

    # Add traces
    for r, joint in enumerate(joints, start=1):
        allowed_components = set(components_by_joint.get(joint, []))

        for c, comp in enumerate(col_components, start=1):
            if comp not in allowed_components:
                # leave the panel blank but *don’t* look like a missing facet:
                # we’ll hide axes below
                continue

            dd = d[(d["joint"] == joint) & (d["component"] == comp)]
            if dd.empty:
                continue

            for tracker in trackers_order:
                ddd = dd[dd["tracker"] == tracker]
                if ddd.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=ddd["percent_gait_cycle"],
                        y=ddd["value"],
                        mode="lines",
                        name=tracker,
                        legendgroup=tracker,
                        showlegend=(r == 1 and c == 1),  # legend once
                        line=dict(color=tracker_colors.get(tracker, None), width=2),
                        hovertemplate=(
                            f"Condition: {condition}<br>"
                            f"Side: {side}<br>"
                            f"Joint: {joint}<br>"
                            f"Component: {comp}<br>"
                            "Tracker: %{fullData.name}<br>"
                            "% gait: %{x}<br>"
                            "Angle: %{y:.2f} deg<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )

            fig.update_xaxes(title_text="% gait cycle" if r == len(joints) else None, row=r, col=c)
            fig.update_yaxes(title_text="Angle (deg)" if c == 1 else None, row=r, col=c)

    # Hide axes for panels that are not meaningful for that joint
    for r, joint in enumerate(joints, start=1):
        allowed_components = set(components_by_joint.get(joint, []))
        for c, comp in enumerate(col_components, start=1):
            if comp not in allowed_components:
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        height=height,
        width=width,
        title=f"Mean joint angles by tracker — {side} — condition: {condition}",
        legend_title_text="Tracker",
        margin=dict(t=110),
    )
    return fig



# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    recording_session = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1")

    trackers = ["qualisys", "mediapipe", "rtmpose", "vitpose_25", "vitpose_wholebody"]

    tracker_colors = {
        "qualisys": "black",
        "mediapipe": "#2ca02c",
        "rtmpose": "#d62728",
        "vitpose_25": "#1f77b4",
        "vitpose_wholebody": "#ff7f0e",
    }

    df_angles = load_joint_angle_stride_summary_stats(recording_session, trackers)

    # Single condition plot
    # fig = plot_mean_joint_angles_by_tracker(
    #     df_angles,
    #     condition="overall",
    #     side="left",
    #     trackers_order=trackers,
    #     tracker_colors=tracker_colors,
    # )
    # fig.show()

    # Dropdown across all conditions
    fig = plot_mean_joint_angles_canonical_grid(
        df_angles,
        side="left",
        condition="overall",
        joints=["ankle", "knee", "hip"],
        trackers_order=["qualisys", "mediapipe", "rtmpose", "vitpose_25", "vitpose_wholebody"],
        height=1200,
        width=2200,
    )
    fig.show()
