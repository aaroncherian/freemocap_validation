import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Database connection and data loading ----
conn = sqlite3.connect("validation.db")

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
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.condition = "speed_0_5" 
    AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""

path_df = pd.read_sql_query(query, conn)
dfs = []

for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    condition = row.get("condition") or ""
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_csv(path)
    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker
    sub_df["condition"] = condition if condition else "none"
    dfs.append(sub_df)

combined_df = pd.concat(dfs, ignore_index=True)

# Create pivot table and calculate errors
pivot = combined_df.pivot_table(
    index=["participant_code", "trial_name", "marker", "axis", "percent_gait_cycle"],
    columns="tracker", 
    values="value"
).reset_index()

pivot["error"] = pivot["mediapipe"] - pivot["qualisys"]

# Calculate mean and std error across trials
error_waveforms = (
    pivot.groupby(["marker", "axis", "percent_gait_cycle"], as_index=False)
         .agg(mean_error=("error", "mean"), std_error=("error", "std"))
)

# ---- Configuration ----
JOINTS = ["HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
axes = ["x", "y", "z"]  
sides = ["left", "right"]

# Visual styling
LINE_WIDTH = 2.5

# Colors for each axis - main lines
AXIS_COLORS = {
    "x": "#d62728",  # Red
    "y": "#2ca02c",  # Green
    "z": "#1f77b4"   # Blue
}

# Lighter colors for error bands
RIBBON_COLORS = {
    "x": "#ffcccb",  # Light red/pink
    "y": "#90ee90",  # Light green
    "z": "#add8e6"   # Light blue
}

RIBBON_ALPHA = 0.4  # Moderate opacity for the light colors

# Process data
def split_marker(marker_name):
    """Extract side and joint from marker name"""
    m = marker_name.lower()
    if m.startswith("left_"):
        side = "left"
        joint = m.replace("left_", "").upper()
    elif m.startswith("right_"):
        side = "right"
        joint = m.replace("right_", "").upper()
    else:
        side = "unknown"
        joint = m.upper()
    return pd.Series({"side": side, "joint": joint})

err = error_waveforms.copy()
err[["side", "joint"]] = err["marker"].apply(split_marker)

# Filter to only include joints and sides we want
err = err[err["joint"].isin(JOINTS) & err["side"].isin(sides) & err["axis"].isin(axes)].copy()

# ---- Create figure ----
n_rows = len(JOINTS) * 2  # Left and Right for each joint
n_cols = len(axes)

fig = make_subplots(
    rows=n_rows, 
    cols=n_cols,
    shared_xaxes=True,
    vertical_spacing=0.025,  # Small gap between all rows
    horizontal_spacing=0.09,  # More room between columns
    column_titles=[ax.upper() for ax in axes],
)

# Helper function for axis key
def ax_key(kind, row, col):
    idx = (row - 1) * n_cols + col
    return f"{kind}axis" + ("" if idx == 1 else str(idx))

# Collect y-range per (joint, axis) to sync left/right
y_minmax = {(j, a): [np.inf, -np.inf] for j in JOINTS for a in axes}

# ---- Add traces ----
row_idx = 0
for j_idx, joint in enumerate(JOINTS):
    for side in sides:
        row_idx += 1
        marker_name = f"{side}_{joint.lower()}"
        
        for c, axis in enumerate(axes, start=1):
            # Get data for this marker/axis combination
            df_ma = err[(err["marker"] == marker_name) & (err["axis"] == axis)]
            
            # Get the colors for this axis
            axis_color = AXIS_COLORS[axis]
            ribbon_color = RIBBON_COLORS[axis]
            
            if not df_ma.empty:
                df_ma = df_ma.sort_values("percent_gait_cycle")
                x = df_ma["percent_gait_cycle"].values
                mean = df_ma["mean_error"].values
                std = df_ma["std_error"].values
                
                # Update y-range
                lower = mean - std
                upper = mean + std
                y_minmax[(joint, axis)][0] = min(y_minmax[(joint, axis)][0], np.nanmin(lower))
                y_minmax[(joint, axis)][1] = max(y_minmax[(joint, axis)][1], np.nanmax(upper))
                
                # Add confidence ribbon (two traces with fill)
                fig.add_trace(
                    go.Scatter(
                        x=x, y=upper,
                        mode='lines',
                        line=dict(color=ribbon_color, width=0),
                        hoverinfo='skip',
                        opacity=RIBBON_ALPHA,
                        showlegend=False
                    ),
                    row=row_idx, col=c
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=x, y=lower,
                        mode='lines',
                        line=dict(color=ribbon_color, width=0),
                        fill='tonexty',
                        fillcolor=ribbon_color,
                        hoverinfo='skip',
                        opacity=RIBBON_ALPHA,
                        showlegend=False
                    ),
                    row=row_idx, col=c
                )
                
                # Add mean line
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean,
                        mode='lines',
                        line=dict(color=axis_color, width=LINE_WIDTH),
                        hovertemplate="Gait cycle: %{x:.0f}%<br>Error: %{y:.1f} mm<extra></extra>",
                        showlegend=False
                    ),
                    row=row_idx, col=c
                )

# ---- Sync y-ranges for left/right per joint/axis ----
for j_idx, joint in enumerate(JOINTS):
    left_row = j_idx * 2 + 1
    right_row = j_idx * 2 + 2
    for c, axis in enumerate(axes, start=1):
        lo, hi = y_minmax[(joint, axis)]
        if np.isfinite(lo) and np.isfinite(hi):
            pad = (hi - lo) * 0.1 if hi > lo else 5
            rng = [lo - pad, hi + pad]
            fig.update_yaxes(range=rng, row=left_row, col=c)
            fig.update_yaxes(range=rng, row=right_row, col=c)
            
            # Add zero line
            fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.2)", 
                         row=left_row, col=c)
            fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.2)", 
                         row=right_row, col=c)

# ---- Ticks and labels ----
tickvals = list(range(0, 101, 20))

# X-axis labels only on bottom row
for c in range(1, n_cols + 1):
    fig.update_xaxes(
        showticklabels=True,
        tickmode='array', 
        tickvals=tickvals, 
        ticks='outside',
        title=dict(text="Percent gait cycle", standoff=6),
        row=n_rows, col=c
    )

# Show y-ticks on all subplots
for r in range(1, n_rows + 1):
    for c in range(1, n_cols + 1):
        fig.update_yaxes(showticklabels=True, ticks='outside', automargin=True, row=r, col=c)

# ---- Alternating row backgrounds (Left vs Right) ----
fig.update_layout(margin=dict(l=120, r=40, t=90, b=90))

for r in range(1, n_rows + 1):
    y0, y1 = fig.layout[ax_key("y", r, 1)].domain
    fill = "rgba(0,0,0,0.03)" if (r % 2 == 1) else "rgba(0,0,0,0.015)"
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0.0, x1=1.0, y0=y0, y1=y1,
        layer="below", line=dict(width=0), fillcolor=fill
    )

# ---- Left/Right labels beside every subplot ----
side_text = {"left": "Left", "right": "Right"}
for r in range(1, n_rows + 1):
    is_left_row = (r % 2 == 1)
    side = "left" if is_left_row else "right"
    for c in range(1, n_cols + 1):
        xdom = fig.layout[ax_key("x", r, c)].domain
        ydom = fig.layout[ax_key("y", r, c)].domain
        x_pos = xdom[0] - 0.04  # Small gutter before each subplot
        y_pos = 0.5 * (ydom[0] + ydom[1])
        fig.add_annotation(
            x=x_pos, xref="paper",
            y=y_pos, yref="paper",
            text=side_text[side],
            textangle=-90,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=13, color="#444")
        )

# ---- Joint titles positioned ABOVE each joint pair ----
for j_idx, joint in enumerate(JOINTS):
    top_row = j_idx * 2 + 1
    
    # Position above the top row of each joint pair
    y_top = fig.layout[ax_key("y", top_row, 1)].domain[1]
    
    # Place title above the joint pair with consistent padding
    y_position = y_top + 0.008  # Standard padding for all joints
    
    fig.add_annotation(
        x=0.5, xref="paper",
        y=y_position, yref="paper",
        text=f"<b>{joint}</b>",
        showarrow=False,
        xanchor="center", yanchor="bottom",
        font=dict(size=13, color="#000"),
        bgcolor="rgba(255,255,255,0.96)", 
        borderpad=2
    )

# ---- Divider lines between joints ----
for j_idx in range(len(JOINTS) - 1):  # Skip after the last one
    bot_row = j_idx * 2 + 2
    y_bot = fig.layout[ax_key("y", bot_row, 1)].domain[0]
    fig.add_shape(
        type="line",
        x0=0.0, x1=1.0, xref="paper",
        y0=y_bot - 0.012, y1=y_bot - 0.012, yref="paper",
        line=dict(width=1, color="rgba(0,0,0,0.65)")
    )

# ---- Single Y-axis label on first column ----
y_top = fig.layout[ax_key("y", 1, 1)].domain[1]
y_bot = fig.layout[ax_key("y", n_rows, 1)].domain[0]
y_center = (y_top + y_bot) / 2
x_left = fig.layout[ax_key("x", 1, 1)].domain[0]
fig.add_annotation(
    x=x_left - 0.070, xref="paper",
    y=y_center, yref="paper",
    text="<b>Error (mm)</b>",
    textangle=-90,
    showarrow=False,
    xanchor="center", yanchor="middle",
    font=dict(size=13, color="#000")
)

# ---- Final layout ----
fig.update_layout(
    height=max(440, 120 * n_rows),
    width=1120,
    template="plotly_white",
    title={
        'text': "<b>Trajectory Error (FreeMoCap − Qualisys)</b><br><sub>Mean ± SD across all trials</sub>",
        'y': 0.98,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
)

# Move column titles (X, Y, Z) higher - they're the first 3 annotations
for i in range(3):
    fig.layout.annotations[i]['y'] = 1.02  # Move them above the plot area

fig.update_xaxes(showgrid=True, zeroline=False)
fig.update_yaxes(showgrid=True, zeroline=False)

# Print diagnostic info
print(f"\nData summary:")
print(f"Total markers in error_waveforms: {error_waveforms['marker'].nunique()}")
print(f"Markers after filtering: {err['marker'].nunique()}")
print(f"Unique joints: {err['joint'].unique().tolist()}")
print(f"Unique sides: {err['side'].unique().tolist()}")
print(f"Unique axes: {err['axis'].unique().tolist()}")

fig.show()