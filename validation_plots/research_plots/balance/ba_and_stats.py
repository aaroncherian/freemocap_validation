import sqlite3
from pathlib import Path
from plotly.colors import sample_colorscale

import numpy as np
import pandas as pd
import pingouin as pg
from plotly.subplots import make_subplots
import plotly.graph_objects as go


DB_PATH = "validation.db"

CONDITION_ORDER = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

TRACKERS = ["qualisys", "mediapipe"]


# -------------------------
# Helpers
# -------------------------
def load_path_length_json(json_path: str) -> pd.DataFrame:
    """
    Loads your COM path length JSON artifact into a tidy dataframe with:
      condition, path_length, frame_interval (if present)
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {json_path}")

    # If your JSON is a dict-like structure, read it as-is.
    # pd.read_json(path) usually works; this is just explicit and debuggable.
    raw = pd.read_json(p)

    # Your earlier rename suggests the JSON columns look like these:
    # "Frame Intervals" and "Path Lengths:"
    # But sometimes these vary slightly; make it tolerant.
    col_map = {
        "Frame Intervals": "frame_interval",
        "Frame Interval": "frame_interval",
        "Path Lengths:": "path_length",
        "Path Lengths": "path_length",
        "Path Length": "path_length",
    }
    raw = raw.rename(columns=col_map)

    # If conditions are stored as the index, move them to a column.
    raw = raw.reset_index().rename(columns={"index": "condition"})

    # Keep only what we need
    keep = [c for c in ["condition", "path_length", "frame_interval"] if c in raw.columns]
    out = raw[keep].copy()

    # Ensure numeric
    out["path_length"] = pd.to_numeric(out["path_length"], errors="coerce")

    return out


def to_long_for_icc(wide_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """
    Convert paired columns (qualisys, mediapipe) into pingouin ICC long format.
    """
    long = pd.concat(
        [
            wide_df.assign(rater="qualisys", rating=wide_df["qualisys"]),
            wide_df.assign(rater="mediapipe", rating=wide_df["mediapipe"]),
        ],
        ignore_index=True,
    )
    long["target"] = long[target_cols].astype(str).agg("|".join, axis=1)
    return long[["target", "rater", "rating"]]


def compute_icc_absolute_agreement(wide_df: pd.DataFrame, target_cols: list[str], label: str) -> dict:
    """
    ICC absolute agreement:
      - ICC2  : single-measure, absolute agreement
      - ICC2k : average-measure, absolute agreement

    Note: Pingouin labels include: ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k.
    """
    long = to_long_for_icc(wide_df, target_cols=target_cols)

    icc_table = pg.intraclass_corr(
        data=long,
        targets="target",
        raters="rater",
        ratings="rating",
    )

    row_2  = icc_table.loc[icc_table["Type"] == "ICC2"].iloc[0]
    row_2k = icc_table.loc[icc_table["Type"] == "ICC2k"].iloc[0]

    return {
        "subset": label,
        "n_targets": long["target"].nunique(),
        "ICC2": row_2["ICC"],
        "ICC2_CI95_low": row_2["CI95%"][0],
        "ICC2_CI95_high": row_2["CI95%"][1],
        "ICC2k": row_2k["ICC"],
        "ICC2k_CI95_low": row_2k["CI95%"][0],
        "ICC2k_CI95_high": row_2k["CI95%"][1],
    }


# -------------------------
# 1) Load from DB
# -------------------------
conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    t.participant_code,
    t.trial_name,
    a.path,
    a.condition,
    a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE
    t.trial_type = "balance"
    AND a.category = "com_analysis"
    AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
    AND a.file_exists = 1
    AND a.component_name LIKE '%path_length_com%'
ORDER BY
    t.trial_name, a.path;
"""
artifact_df = pd.read_sql_query(query, conn)
conn.close()

dfs = []
for _, row in artifact_df.iterrows():
    sub = load_path_length_json(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"]
    sub["tracker"] = row["tracker"]
    dfs.append(sub)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df["condition"] = pd.Categorical(
    combined_df["condition"], categories=CONDITION_ORDER, ordered=True
)

# -------------------------
# 2) Wide format for paired comparisons
# -------------------------
wide_df = (
    combined_df
    .pivot_table(
        index=["participant_code", "trial_name", "condition"],
        columns="tracker",
        values="path_length",
        aggfunc="first",
    )
    .reset_index()
    .dropna(subset=["qualisys", "mediapipe"])
)

# -------------------------
# 3) ICC (absolute agreement)
# -------------------------
icc_results = []
icc_results.append(
    compute_icc_absolute_agreement(
        wide_df,
        target_cols=["participant_code", "trial_name", "condition"],
        label="All targets (participant×trial×condition)",
    )
)

# Per-condition ICC with participant as target (often easier to defend)
for cond in CONDITION_ORDER:
    sub = wide_df[wide_df["condition"] == cond].copy()
    if sub["participant_code"].nunique() >= 3:
        # If you have multiple trials per participant, average them first:
        sub_avg = (
            sub.groupby(["participant_code", "condition"], as_index=False)[["qualisys", "mediapipe"]]
            .mean()
        )
        icc_results.append(
            compute_icc_absolute_agreement(
                sub_avg,
                target_cols=["participant_code", "condition"],
                label=f"{cond} (participant-averaged)",
            )
        )

icc_df = pd.DataFrame(icc_results)
print("\n=== ICC (Absolute agreement) ===")
print(icc_df.to_string(index=False, float_format="%.3f"))

# -------------------------
# 4A) Paired scatter (raw agreement)
# -------------------------
trackers = ["mediapipe", "rtmpose", "vitpose"]  # change to yours
titles = ["MediaPipe", "RTMPose", "ViTPose"]

fig = make_subplots(
    rows=1,
    cols=len(trackers),
    subplot_titles=titles,
    shared_yaxes=True,
)

# Global axis limits
all_vals = combined_df["path_length"].values
global_min = np.nanmin(all_vals)
global_max = np.nanmax(all_vals)
pad = (global_max - global_min) * 0.05
axis_range = [global_min - pad, global_max + pad]

for i, tracker in enumerate(trackers, start=1):
    
    sub = (
        combined_df[combined_df["tracker"].isin(["qualisys", tracker])]
        .pivot_table(
            index=["participant_code", "trial_name", "condition"],
            columns="tracker",
            values="path_length",
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    
    fig.add_trace(
        go.Scatter(
            x=sub["qualisys"],
            y=sub[tracker],
            mode="markers",
            marker=dict(size=8, opacity=0.75),
            showlegend=False,
        ),
        row=1,
        col=i,
    )
    
    # Identity line
    fig.add_trace(
        go.Scatter(
            x=axis_range,
            y=axis_range,
            mode="lines",
            line=dict(dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=i,
    )

    fig.update_xaxes(range=axis_range, row=1, col=i)
    fig.update_yaxes(range=axis_range, row=1, col=i)

fig.update_layout(
    template="simple_white",
    height=450,
    width=1300,
    title="Paired Scatter: COM Path Length Agreement by Tracker",
)

fig.update_yaxes(title_text="Tracker (mm)", row=1, col=1)
fig.update_xaxes(title_text="Qualisys (mm)")

fig.show()

# ============================================================
# 4. Bland-Altman plots — per condition + pooled (FIXED)
#   - adds bias + LoA lines
#   - forces shared, symmetric y-limits across panels
#   - makes the figure more horizontal
# ============================================================

wide_df = wide_df.copy()
wide_df["ba_mean"] = (wide_df["qualisys"] + wide_df["mediapipe"]) / 2
wide_df["ba_diff"] = wide_df["mediapipe"] - wide_df["qualisys"]

ba_groups = ["All conditions"] + CONDITION_ORDER
n_panels = len(ba_groups)

fig = make_subplots(
    rows=1,
    cols=n_panels,
    subplot_titles=ba_groups,
    shared_yaxes=True,
    horizontal_spacing=0.06,
)

def axis_ref(col: int):
    """Plotly axis refs: col=1 -> ('x','y'), col=2 -> ('x2','y2'), ..."""
    if col == 1:
        return "x", "y"
    return f"x{col}", f"y{col}"

def add_ba_panel(fig, sub: pd.DataFrame, col: int):
    """Add one Bland–Altman panel with bias + LoA lines."""
    bias = sub["ba_diff"].mean()
    sd = sub["ba_diff"].std(ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # X span for horizontal lines
    x_min = float(sub["ba_mean"].min())
    x_max = float(sub["ba_mean"].max())
    x_pad = (x_max - x_min) * 0.08
    if not np.isfinite(x_pad) or x_pad == 0:
        x_pad = 0.01
    x0, x1 = x_min - x_pad, x_max + x_pad

    xref, yref = axis_ref(col)

    # Points
    fig.add_trace(
        go.Scatter(
            x=sub["ba_mean"],
            y=sub["ba_diff"],
            mode="markers",
            marker=dict(size=8, opacity=0.75),
            showlegend=False,
        ),
        row=1, col=col,
    )

    # Bias line (black dash)
    fig.add_shape(
        type="line",
        x0=x0, x1=x1, y0=bias, y1=bias,
        xref=xref, yref=yref,
        line=dict(color="black", width=2, dash="dash"),
    )

    # LoA lines (red dot)
    for loa in (loa_upper, loa_lower):
        fig.add_shape(
            type="line",
            x0=x0, x1=x1, y0=loa, y1=loa,
            xref=xref, yref=yref,
            line=dict(color="red", width=1.5, dash="dot"),
        )

    return dict(bias=bias, sd=sd, loa_lower=loa_lower, loa_upper=loa_upper)

# Build panels + collect stats
ba_stats = []
for i, group in enumerate(ba_groups, start=1):
    sub = wide_df if group == "All conditions" else wide_df[wide_df["condition"] == group]
    if len(sub) == 0:
        continue
    stats = add_ba_panel(fig, sub, i)
    stats.update({"condition": group, "n": len(sub)})
    ba_stats.append(stats)

ba_stats_df = pd.DataFrame(ba_stats)

# --- Force shared, symmetric y-limits across ALL panels ---
# Use pooled LoA envelope, then pad a bit.
y_lo = float(ba_stats_df["loa_lower"].min())
y_hi = float(ba_stats_df["loa_upper"].max())
y_pad = 0.06 * (y_hi - y_lo) if np.isfinite(y_hi - y_lo) and (y_hi - y_lo) > 0 else 0.01
y_lo -= y_pad
y_hi += y_pad
# Make symmetric about 0 (looks cleaner)
m = max(abs(y_lo), abs(y_hi))
y_range = [-m, m]

fig.update_yaxes(title_text="MediaPipe − Qualisys (mm)", row=1, col=1, range=y_range)
for c in range(1, n_panels + 1):
    fig.update_xaxes(title_text="Mean of systems (mm)", row=1, col=c)

# Make it more horizontal + less “tall”
fig.update_layout(
    template="simple_white",
    title_text="Bland–Altman: COM path length (MediaPipe vs Qualisys)",
    height=320,
    width=1500,
    margin=dict(t=70, b=60, l=80, r=40),
)

# fig.show()

print("\n=== Bland–Altman Summary (Path Length) ===")
print(
    ba_stats_df[["condition", "n", "bias", "sd", "loa_lower", "loa_upper"]]
    .to_string(index=False, float_format="%.4f")
)

df = wide_df.copy()

df["ba_mean"] = (df["qualisys"] + df["mediapipe"]) / 2
df["ba_diff"] = df["mediapipe"] - df["qualisys"]

# --- Compute pooled bias + LoA ---
bias = df["ba_diff"].mean()
sd = df["ba_diff"].std(ddof=1)
loa_upper = bias + 1.96 * sd
loa_lower = bias - 1.96 * sd

# --- Color map ---
ordered_conditions = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

# Generate evenly spaced colors from a sequential scale
colors = sample_colorscale(
    "Viridis",
    [0.15 + 0.6*(i/(len(ordered_conditions)-1)) for i in range(len(ordered_conditions))]
)
color_map = dict(zip(ordered_conditions, colors))

fig = go.Figure()

# Add points by condition
for cond, sub in df.groupby("condition"):
    fig.add_trace(
        go.Scatter(
            x=sub["ba_mean"],
            y=sub["ba_diff"],
            mode="markers",
            name=cond,
            marker=dict(
                size=9,
                opacity=0.8,
                color=color_map.get(cond),
            ),
            hovertemplate=(
                "Participant: %{customdata[0]}<br>"
                "Mean: %{x:.4f} mm<br>"
                "Diff: %{y:.4f} mm<extra></extra>"
            ),
            customdata=sub[["participant_code"]].values,
        )
    )

# X-range padding
x_min = float(df["ba_mean"].min())
x_max = float(df["ba_mean"].max())
x_pad = (x_max - x_min) * 0.08
x0, x1 = x_min - x_pad, x_max + x_pad

# Bias line
fig.add_shape(
    type="line",
    x0=x0, x1=x1,
    y0=bias, y1=bias,
    line=dict(color="black", width=2, dash="dash"),
)

# LoA lines
for y in [loa_upper, loa_lower]:
    fig.add_shape(
        type="line",
        x0=x0, x1=x1,
        y0=y, y1=y,
        line=dict(color="red", width=1.5, dash="dot"),
    )

# --- Force symmetric y-axis ---
m = max(abs(loa_upper), abs(loa_lower))
y_range = [-m * 1.15, m * 1.15]

fig.update_layout(
    template="simple_white",
    title="Bland–Altman: COM Path Length (MediaPipe vs Qualisys)",
    xaxis_title="Mean of systems (mm)",
    yaxis_title="MediaPipe − Qualisys (mm)",
    height=450,
    width=850,
    margin=dict(t=80, b=80, l=80, r=40),
    legend_title_text="Condition",
)

fig.update_yaxes(range=y_range)

fig.show()

print("\n=== Pooled Bland–Altman Summary ===")
print(f"Bias: {bias:.4f} mm")
print(f"LoA: [{loa_lower:.4f}, {loa_upper:.4f}] mm")

ba_stats_df = pd.DataFrame(ba_stats)
print("\n=== Bland–Altman summary ===")
print(ba_stats_df.to_string(index=False, float_format="%.4f"))