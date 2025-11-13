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
AXES = ["x", "y", "z"]  # lowercase to match your data
SIDES = ["Left", "Right"]

# Visual styling
COLOR = "#1f77b4"  # Blue color
LINE_WIDTH = 2.5
RIBBON_ALPHA = 0.2

# Layout spacing - reduced for larger plots
H_SPACING = 0.04
V_SPACING = 0.06
ROW_HEIGHT = 180  # Increased from 140
FIG_WIDTH = 1400  # Increased from 1200
FIG_HEIGHT = ROW_HEIGHT * len(JOINTS) * 2 + 150

def hex_to_rgba(hex_color, alpha):
    """Convert hex color to rgba string"""
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f"rgba({r},{g},{b},{alpha})"

# ---- Process data ----
def split_marker(marker_name):
    """Extract side and joint from marker name"""
    m = marker_name.lower()
    if m.startswith("left_"):
        side = "Left"
        joint = m.replace("left_", "").upper()
    elif m.startswith("right_"):
        side = "Right"
        joint = m.replace("right_", "").upper()
    else:
        side = "Unknown"
        joint = m.upper()
    return pd.Series({"side": side, "joint": joint})

# Process error_waveforms
err = error_waveforms.copy()
err[["side", "joint"]] = err["marker"].apply(split_marker)

# Filter to only include joints and sides we want
err = err[err["joint"].isin(JOINTS) & err["side"].isin(SIDES) & err["axis"].isin(AXES)].copy()

# Enforce categorical ordering
err["joint"] = pd.Categorical(err["joint"], JOINTS, ordered=True)
err["side"] = pd.Categorical(err["side"], SIDES, ordered=True)
err["axis"] = pd.Categorical(err["axis"], AXES, ordered=True)

# ---- Create figure ----
n_rows = len(JOINTS) * 2  # Left and Right for each joint
n_cols = len(AXES)

# Create custom row titles - now empty since we'll add joint names as annotations
row_titles = ["" for _ in range(n_rows)]  # All empty
V_SPACING = 0.10     # was ~0.03–0.06; more vertical space between rows
H_SPACING = 0.08
fig = make_subplots(
    rows=n_rows, 
    cols=n_cols,
    horizontal_spacing=H_SPACING,
    vertical_spacing=V_SPACING,
    column_titles=[f"<b>{ax.upper()}</b>" for ax in AXES],  # Uppercase for display
    row_titles=row_titles,  # All empty now
    specs=[[{} for _ in range(n_cols)] for _ in range(n_rows)]
)
fig.update_layout(height=120 * n_rows + 420)
# ---- Calculate consistent y-ranges per joint ----
joint_ranges = {}
for joint in JOINTS:
    joint_data = err[err["joint"] == joint]
    if len(joint_data) == 0:
        joint_ranges[joint] = [-50, 50]  # Default range if no data
        continue
    
    mean_vals = joint_data["mean_error"].values
    std_vals = joint_data["std_error"].values
    
    y_min = (mean_vals - std_vals).min()
    y_max = (mean_vals + std_vals).max()
    
    # Add padding
    padding = 0.15 * (y_max - y_min) if y_max > y_min else 5
    joint_ranges[joint] = (y_min - padding, y_max + padding)

# ---- Helper functions for subplot positioning ----
def get_row(joint, side):
    """Get row index for given joint and side"""
    joint_idx = JOINTS.index(joint)
    return joint_idx * 2 + (1 if side == "Left" else 2)

def get_col(axis):
    """Get column index for given axis"""
    return AXES.index(axis.lower()) + 1

# ---- Add traces ----
trace_count = 0
for (joint, side, axis), group_data in err.groupby(["joint", "side", "axis"], sort=False):
    group_data = group_data.sort_values("percent_gait_cycle")
    
    if len(group_data) == 0:
        continue
    
    x = group_data["percent_gait_cycle"].values
    mean = group_data["mean_error"].values
    std = group_data["std_error"].values
    
    row = get_row(joint, side)
    col = get_col(axis)
    
    # Add confidence ribbon
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself",
            line=dict(width=0),
            fillcolor=hex_to_rgba(COLOR, RIBBON_ALPHA),
            hoverinfo="skip",
            showlegend=False
        ),
        row=row, col=col
    )
    
    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(width=LINE_WIDTH, color=COLOR),
            hovertemplate="Gait cycle: %{x:.0f}%<br>Error: %{y:.1f} mm<extra></extra>",
            showlegend=False
        ),
        row=row, col=col
    )
    
    trace_count += 2

print(f"Added {trace_count} traces to the plot")

# ---- Add reference lines and format axes ----
for row in range(1, n_rows + 1):
    joint_idx = (row - 1) // 2
    joint = JOINTS[joint_idx]
    side = "Left" if row % 2 == 1 else "Right"
    
    for col in range(1, n_cols + 1):
        # Add zero line
        fig.add_hline(
            y=0, 
            line_width=1, 
            line_color="rgba(0,0,0,0.2)", 
            row=row, 
            col=col
        )
        
        # Update y-axis
        y_range = joint_ranges.get(joint, [-50, 50])
        fig.update_yaxes(
            range=y_range,
            gridcolor="rgba(0,0,0,0.05)",
            row=row, 
            col=col
        )
        
        # Add side label to y-axis
        if col == 1:  # Only for first column
            fig.update_yaxes(
                title_text=f"<b>{side}</b>",
                title_font=dict(size=10),
                row=row, 
                col=col
            )
        
        # Update x-axis
        fig.update_xaxes(
            range=[0, 100],
            gridcolor="rgba(0,0,0,0.05)",
            row=row, 
            col=col
        )
        
        # Only show x-axis title for bottom row
        if row == n_rows:
            fig.update_xaxes(
                title_text="Gait Cycle (%)",
                title_font=dict(size=10),
                row=row, 
                col=col
            )

# ---- Add divider lines between joints ----
for i in range(1, len(JOINTS)):
    y_position = 1 - (i * 2 / n_rows)  # Calculate position in paper coordinates
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0.05,
        x1=0.95,
        y0=y_position,
        y1=y_position,
        line=dict(color="rgba(0,0,0,0.15)", width=1)
    )

# ---- Update overall layout ----
fig.update_layout(
    title={
        'text': "<b>Trajectory Error (FreeMoCap − Qualisys)</b><br><sub>Mean ± SD across all trials</sub>",
        'y': 0.99,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    template="plotly_white",
    height=FIG_HEIGHT,
    width=FIG_WIDTH,
    margin=dict(l=80, r=30, t=120, b=80),  # Adjusted margins
    font=dict(size=12),  # Slightly larger base font
    showlegend=False
)

# Add overall y-axis label
fig.add_annotation(
    text="<b>Error (mm)</b>",
    xref="paper",
    yref="paper",
    x=-0.06,
    y=0.5,
    showarrow=False,
    textangle=-90,
    font=dict(size=12)
)

# ---- Add joint titles above each joint section ----
# These need to be added after the figure layout is established
for idx, joint in enumerate(JOINTS):
    # Each joint takes up 2 rows (Left and Right)
    # We want to position the title above the middle column of each joint's section
    
    # Calculate y position based on the joint index
    # The figure is divided into n_rows, and each joint gets 2 rows
    top_of_joint = 1 - (idx * 2 / n_rows)  # Top of this joint's section
    bottom_of_joint = 1 - ((idx + 1) * 2 / n_rows)  # Bottom of this joint's section
    
    # Position the title slightly above the top of the joint's rows
    y_position = top_of_joint + 0.02
    
    # Don't add title if it would go above the figure
    if y_position < 0.97:
        fig.add_annotation(
            text=f"<b>{joint}</b>",
            xref="paper",
            yref="paper",
            x=0.5,  # Center horizontally
            y=y_position,
            showarrow=False,
            font=dict(size=12),
            xanchor="center",
            yanchor="bottom"
        )

# Print diagnostic info
print(f"\nData summary:")
print(f"Total markers in error_waveforms: {error_waveforms['marker'].nunique()}")
print(f"Markers after filtering: {err['marker'].nunique()}")
print(f"Unique joints: {err['joint'].unique().tolist()}")
print(f"Unique sides: {err['side'].unique().tolist()}")
print(f"Unique axes: {err['axis'].unique().tolist()}")

fig.show()