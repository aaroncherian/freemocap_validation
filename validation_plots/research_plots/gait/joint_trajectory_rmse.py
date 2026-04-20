import re
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------
# Config
# ------------------------
TRACKERS = ["mediapipe", "rtmpose", "vitpose", "qualisys"]
REFERENCE_SYSTEM = "qualisys"

JOINT_ORDER = ["hip", "knee", "ankle", "foot_index"]
AXES = ["x", "y", "z"]
AXIS_PRETTY = {"x": "ML", "y": "AP", "z": "Vertical"}

DB_PATH = "validation.db"

# Where to save slide-ready CSVs (edit)
OUT_DIR = Path(r"D:\validation\gait\trajectories")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# ------------------------
# Helpers
# ------------------------
def speed_key(cond: str) -> float:
    m = re.search(r"speed_(\d+)[_\.](\d+)", str(cond))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"speed_(\d+)", str(cond))
    if m2:
        return float(m2.group(1))
    return float("inf")

def parse_speed_float(cond: str) -> float:
    s = (
        str(cond)
        .replace("speed_", "")
        .replace("_", ".")
    )
    try:
        return float(s)
    except Exception:
        return np.nan

def calculate_rmse(tracker: pd.Series, reference: pd.Series) -> float:
    t = tracker.to_numpy(dtype=float)
    r = reference.to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(r)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((t[m] - r[m]) ** 2)))

def trajectory_rmse_slide_table(rmse_df: pd.DataFrame, joint_name: str, axis: str) -> pd.DataFrame:
    """
    Returns a slide-ready table:
      rows = Tracker
      cols = speed (m/s)
      values = RMSE (mm)
    """
    out = (
        rmse_df[(rmse_df["joint"] == joint_name) & (rmse_df["axis"] == axis)]
        .pivot_table(index="tracker", columns="speed", values="rmse", aggfunc="first")
        .sort_index(axis=1)
    )

    out = out.round(1)
    out.index = out.index.str.capitalize()
    out.columns = [f"{c:g}" for c in out.columns]  # 0.5, 1.0, etc.

    # Put tracker names into first column header like your joint-angle tables
    out.insert(0, "Speed (m/s)", out.index)
    out = out.set_index("Speed (m/s)")
    return out

def get_data_from_dataframe(database_path):
    conn = sqlite3.connect(database_path)
    query = """
    SELECT t.participant_code,
        t.trial_name,
        a.path,
        a.component_name,
        a.condition,
        a.tracker
    FROM artifacts a
    JOIN trials t ON a.trial_id = t.id
    WHERE t.trial_type = "treadmill"
    AND a.category = "trajectories_per_stride"
    AND a.tracker IN ("mediapipe", "rtmpose", "vitpose", "qualisys")
    AND a.file_exists = 1
    AND a.condition LIKE "speed_%"
    AND a.component_name LIKE "%summary_stats"
    ORDER BY t.trial_name, a.path
    """
    path_df = pd.read_sql_query(query, conn)

    dfs = []
    for _, row in path_df.iterrows():
        sub = pd.read_csv(row["path"])
        sub["participant_code"] = row["participant_code"]
        sub["trial_name"] = row["trial_name"]
        sub["tracker"] = (row["tracker"] or "").lower()
        sub["condition"] = row["condition"] or "none"
        dfs.append(sub)

    if not dfs:
        raise RuntimeError("No trajectory summary_stats CSVs found from the query.")

    return pd.concat(dfs, ignore_index=True,)

def combine_left_and_right_side(df: pd.DataFrame, joints_to_use:list[str]):
    m = df["marker"].astype(str).str.strip().str.lower()
    df["side"] = np.select(
        [m.str.startswith("left_"), m.str.startswith("right_")],
        ["left", "right"],
        default="unknown",
    )
    df["joint"] = m.str.replace(r"^(left_|right_)", "", regex=True)

    df = df[
        df["joint"].isin(joints_to_use)
        & (df["stat"].astype(str).str.lower() == "mean")
    ].copy()

    df["value_mirrored"] = df["value"].astype(float)
    left_ml = (df["axis"] == "x") & (df["side"] == "left")
    df.loc[left_ml, "value_mirrored"] *= -1

    return (
        df.groupby(
            ["condition", "tracker", "participant_code", "trial_name",
             "joint", "axis", "percent_gait_cycle"],
            as_index=False,
        )
        .agg(trial_mean_value=("value_mirrored", "mean"))
    )


def calculate_total_mean_and_std_rmse(df: pd.DataFrame, tracker_list = list[str], reference_system = str):
    df_trial_long = df.pivot(
        index = ['condition', 'participant_code', 'trial_name', 'joint', 'axis', 'percent_gait_cycle'],
        columns = 'tracker',
        values = 'trial_mean_value'
    ).reset_index()

    fmc_trackers = [tracker for tracker in tracker_list if tracker != reference_system]
    df_trial_melted = df_trial_long.melt(
        id_vars = ['condition', 'participant_code', 'trial_name', 'joint', 'axis', 'percent_gait_cycle', reference_system],
        value_vars = fmc_trackers,
        var_name = "tracker",
        value_name = "mean_trajectory"
    )

    df_grouped = df_trial_melted.groupby(["condition", "trial_name", "axis", "tracker", "joint"])

    
    rows = []
    for (condition, trial_name, axis, tracker, joint), trial_group in df_grouped:
        rmse = np.sqrt(np.mean((trial_group['mean_trajectory'] - trial_group[reference_system])**2))
        row = {"condition": condition,
               "trial_name": trial_name,
               "axis": axis,
               "tracker": tracker,
               "joint": joint,
               "rmse": rmse,}
        rows.append(row)
    trial_level_mean_rmse = pd.DataFrame(rows)
    
    return (trial_level_mean_rmse
                      .groupby(["condition", "axis", "tracker", "joint"])
                      .agg( 
                          mean = ("rmse", "mean"),
                          std = ("rmse", "std"),
                      )).reset_index()
    
def generate_typst_trajectory_rmse_table(rmse_df: pd.DataFrame, axis: str) -> str:
        rmse_df = rmse_df.copy()
        rmse_df["speed"] = rmse_df["condition"].map(parse_speed_float)
        
        speeds = sorted(rmse_df["speed"].dropna().unique())
        speed_labels = [f"{s:g}" for s in speeds]
        
        trackers_ordered = [t for t in ["mediapipe", "rtmpose", "vitpose"] if t in rmse_df["tracker"].unique()]
        n_trackers = len(trackers_ordered)
        
        n_speed_cols = len(speeds)
        col_spec = f"(1fr, 1.5fr, {'1.2fr, ' * (n_speed_cols - 1)}1.2fr)"
        align_spec = f"(left, left, {'center, ' * (n_speed_cols - 1)}center)"
        
        axis_label = AXIS_PRETTY.get(axis, axis.upper())
        label = f"tbl-traj-rmse-{axis}"
        caption = (
            f"Trajectory RMSE per pose estimation backend across speeds — {axis_label} (mm). "
            f"Values represent mean ± SD RMSE across all participants and trials compared to the marker-based reference."
        )
        
        subset_df = rmse_df[rmse_df["axis"] == axis]
        
        header_cells = ["[*Joint*]", "[*Backend*]"]
        for sl in speed_labels:
            header_cells.append(f"[*{sl} m/s*]")
        
        body_lines = []
        for joint in JOINT_ORDER:
            subset = subset_df[subset_df["joint"] == joint]
            if subset.empty:
                continue
            
            joint_label = JOINT_DISPLAY.get(joint, joint.title())
            
            for i, tracker in enumerate(trackers_ordered):
                row_data = subset[subset["tracker"] == tracker]
                display_name = TRACKER_DISPLAY.get(tracker, tracker.title())
                
                if i == 0:
                    body_lines.append(f"      table.cell(rowspan: {n_trackers}, align: horizon)[{joint_label}],")
                
                body_lines.append(f"      [{display_name}],")
                for spd in speeds:
                    val = row_data.loc[row_data["speed"] == spd]
                    if len(val) > 0:
                        m = val["mean"].iloc[0]
                        s = val["std"].iloc[0]
                        if np.isfinite(m) and np.isfinite(s):
                            body_lines.append(f"      [{m:.1f} ± {s:.1f}],")
                        elif np.isfinite(m):
                            body_lines.append(f"      [{m:.1f}],")
                        else:
                            body_lines.append(f"      [--],")
                    else:
                        body_lines.append(f"      [--],")
            
            body_lines.append("      table.hline(stroke: 0.5pt),")
        
        lines = []
        lines.append("#figure(")
        lines.append("  {")
        lines.append("    set text(size: 9pt)")
        lines.append("    table(")
        lines.append(f"      columns: {col_spec},")
        lines.append(f"      align: {align_spec},")
        lines.append("      stroke: none,")
        lines.append("      table.hline(stroke: 1pt),")
        lines.append("      table.header(")
        for cell in header_cells:
            lines.append(f"        {cell},")
        lines.append("      ),")
        lines.append("      table.hline(stroke: 0.5pt),")
        lines.extend(body_lines)
        lines.append("      table.hline(stroke: 1pt),")
        lines.append("    )")
        lines.append("  },")
        lines.append(f"  caption: [{caption}],")
        lines.append(f") <{label}>")
        
        return "\n".join(lines) + "\n"

# ------------------------
# Plotly trajectory RMSE figures for presentation
# Append below your existing script
# ------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _prepare_rmse_plot_df(rmse_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = rmse_df.copy()
    plot_df["speed"] = plot_df["condition"].map(parse_speed_float)
    plot_df = plot_df.sort_values(["axis", "joint", "tracker", "speed"]).reset_index(drop=True)
    plot_df["std"] = plot_df["std"].fillna(0)
    return plot_df


def plot_trajectory_rmse_recommended_plotly(rmse_df: pd.DataFrame):
    """
    Recommended layout:
      rows = axis
      cols = joint
      traces = tracker
    Best for talking through error trends.
    """
    plot_df = _prepare_rmse_plot_df(rmse_df)

    tracker_order = [t for t in ["mediapipe", "rtmpose", "vitpose"] if t in plot_df["tracker"].unique()]
    joint_order = [j for j in JOINT_ORDER if j in plot_df["joint"].unique()]
    axis_order = [a for a in AXES if a in plot_df["axis"].unique()]

    fig = make_subplots(
        rows=len(axis_order),
        cols=len(joint_order),
        subplot_titles=[JOINT_DISPLAY.get(j, j.title()) for _a in axis_order for j in joint_order],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for r, axis in enumerate(axis_order, start=1):
        for c, joint in enumerate(joint_order, start=1):
            sub = plot_df[(plot_df["axis"] == axis) & (plot_df["joint"] == joint)]

            for tracker in tracker_order:
                t = sub[sub["tracker"] == tracker].sort_values("speed")
                if t.empty:
                    continue

                label = TRACKER_DISPLAY.get(tracker, tracker.title())
                showlegend = (r == 1 and c == 1)

                fig.add_trace(
                    go.Scatter(
                        x=t["speed"],
                        y=t["mean"],
                        mode="lines+markers",
                        name=label,
                        legendgroup=label,
                        showlegend=showlegend,
                        error_y=dict(
                            type="data",
                            array=t["std"],
                            visible=True,
                            thickness=1,
                            width=3,
                        ),
                        hovertemplate=(
                            f"Tracker: {label}<br>"
                            f"Axis: {AXIS_PRETTY.get(axis, axis.upper())}<br>"
                            f"Joint: {JOINT_DISPLAY.get(joint, joint.title())}<br>"
                            "Speed: %{x} m/s<br>"
                            "RMSE: %{y:.2f} mm<br>"
                            "SD: %{customdata:.2f} mm"
                            "<extra></extra>"
                        ),
                        customdata=t["std"],
                    ),
                    row=r,
                    col=c,
                )

            if r == len(axis_order):
                fig.update_xaxes(title_text="Speed (m/s)", row=r, col=c)

            if c == 1:
                fig.update_yaxes(
                    title_text=f"{AXIS_PRETTY.get(axis, axis.upper())}<br>RMSE (mm)",
                    row=r,
                    col=c,
                )

    fig.update_layout(
        title="Trajectory RMSE trends by axis and joint",
        template="plotly_white",
        height=950,
        width=1400,
        legend_title_text="Tracker",
    )

    fig.show()


def plot_trajectory_rmse_tracker_first_plotly(rmse_df: pd.DataFrame):
    """
    Alternate layout:
      rows = tracker
      cols = joint
      traces = axis
    Better for comparing each tracker's error profile.
    """
    plot_df = _prepare_rmse_plot_df(rmse_df)

    tracker_order = [t for t in ["mediapipe", "rtmpose", "vitpose"] if t in plot_df["tracker"].unique()]
    joint_order = [j for j in JOINT_ORDER if j in plot_df["joint"].unique()]
    axis_order = [a for a in AXES if a in plot_df["axis"].unique()]

    fig = make_subplots(
        rows=len(tracker_order),
        cols=len(joint_order),
        subplot_titles=[JOINT_DISPLAY.get(j, j.title()) for _t in tracker_order for j in joint_order],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for r, tracker in enumerate(tracker_order, start=1):
        for c, joint in enumerate(joint_order, start=1):
            sub = plot_df[(plot_df["tracker"] == tracker) & (plot_df["joint"] == joint)]

            for axis in axis_order:
                t = sub[sub["axis"] == axis].sort_values("speed")
                if t.empty:
                    continue

                label = AXIS_PRETTY.get(axis, axis.upper())
                showlegend = (r == 1 and c == 1)

                fig.add_trace(
                    go.Scatter(
                        x=t["speed"],
                        y=t["mean"],
                        mode="lines+markers",
                        name=label,
                        legendgroup=label,
                        showlegend=showlegend,
                        error_y=dict(
                            type="data",
                            array=t["std"],
                            visible=True,
                            thickness=1,
                            width=3,
                        ),
                        hovertemplate=(
                            f"Tracker: {TRACKER_DISPLAY.get(tracker, tracker.title())}<br>"
                            f"Axis: {label}<br>"
                            f"Joint: {JOINT_DISPLAY.get(joint, joint.title())}<br>"
                            "Speed: %{x} m/s<br>"
                            "RMSE: %{y:.2f} mm<br>"
                            "SD: %{customdata:.2f} mm"
                            "<extra></extra>"
                        ),
                        customdata=t["std"],
                    ),
                    row=r,
                    col=c,
                )

            if r == len(tracker_order):
                fig.update_xaxes(title_text="Speed (m/s)", row=r, col=c)

            if c == 1:
                fig.update_yaxes(
                    title_text=f"{TRACKER_DISPLAY.get(tracker, tracker.title())}<br>RMSE (mm)",
                    row=r,
                    col=c,
                )

    fig.update_layout(
        title="Trajectory RMSE trends by tracker and joint",
        template="plotly_white",
        height=950,
        width=1400,
        legend_title_text="Axis",
    )

    fig.show()




if __name__ == "__main__":
    # ------------------------
    TRACKERS = ["mediapipe", "rtmpose", "vitpose", "qualisys"]
    REFERENCE_SYSTEM = "qualisys"
    DB_PATH = "validation.db"
    JOINT_ORDER = ["hip", "knee", "ankle", "foot_index"]
    AXES = ["x", "y", "z"]

    
    TRACKER_DISPLAY = {
        "mediapipe": "MediaPipe",
        "rtmpose": "RTMPose",
        "vitpose": "ViTPose",
    }

    JOINT_DISPLAY = {
        "hip": "Hip",
        "knee": "Knee",
        "ankle": "Ankle",
        "foot_index": "Toe",
    }

    TYPST_OUT_DIR = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\gait\tables")
    TYPST_OUT_DIR.mkdir(exist_ok=True, parents=True)

    database_data = get_data_from_dataframe(DB_PATH)

    df_trial_lr_mean = combine_left_and_right_side(database_data, JOINT_ORDER)

    df_total_means_and_stds = calculate_total_mean_and_std_rmse(
        df_trial_lr_mean,
        tracker_list=TRACKERS,
        reference_system=REFERENCE_SYSTEM,
    )

    # ------------------------
    # Call these after df_total_means_and_stds is created
    # ------------------------
    plot_trajectory_rmse_recommended_plotly(df_total_means_and_stds)
    plot_trajectory_rmse_tracker_first_plotly(df_total_means_and_stds)

    for axis in AXES:
        typst_content = generate_typst_trajectory_rmse_table(df_total_means_and_stds, axis)
        typst_path = TYPST_OUT_DIR / f"trajectory_rmse_{axis}.typ"
        typst_path.write_text(typst_content, encoding="utf-8")
        print(f"Saved: {typst_path}")