import pandas as pd
import numpy as np
import sqlite3
import pingouin as pg
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ============================================================
# 1. Load path-length data (same pattern as your other scripts)
# ============================================================
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
WHERE t.trial_type = "balance"
    AND a.category = "com_analysis"
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE '%path_length_com'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub_df = pd.read_json(row["path"])
    sub_df = sub_df.rename(columns={
        "Frame Intervals": "frame_interval",
        "Path Lengths:": "path_length",
    }).reset_index().rename(columns={"index": "condition"})

    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    dfs.append(sub_df)

combined_df = pd.concat(dfs, ignore_index=True)

condition_order = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]
combined_df["condition"] = pd.Categorical(
    combined_df["condition"], categories=condition_order, ordered=True
)

# ============================================================
# 2. Pivot to wide format: one row per participant × condition
# ============================================================
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

# ============================================================
# 3. ICC — overall and per condition
# ============================================================
def compute_icc(df, value_col_a="qualisys", value_col_b="mediapipe", label="overall"):
    """Reshape paired columns into long format and compute ICC(3,1)."""
    long = pd.concat([
        df.assign(rater="qualisys", rating=df[value_col_a])[["participant_code", "trial_name", "condition", "rater", "rating"]],
        df.assign(rater="mediapipe", rating=df[value_col_b])[["participant_code", "trial_name", "condition", "rater", "rating"]],
    ], ignore_index=True)

    # Target = unique observation (participant × trial × condition)
    long["target"] = (
        long["participant_code"] + "|" + long["trial_name"] + "|" + long["condition"].astype(str)
    )

    icc_table = pg.intraclass_corr(
        data=long,
        targets="target",
        raters="rater",
        ratings="rating",
    )
    # ICC3,1 = two-way mixed, single measures, consistency
    row = icc_table[icc_table["Type"] == "ICC2k"].iloc[0]  # or ICC3 for single
    row_single = icc_table[icc_table["Type"] == "ICC2"].iloc[0]

    return {
        "subset": label,
        "n": len(df),
        "ICC3_1": row_single["ICC"],
        "ICC3_1_CI95_low": row_single["CI95%"][0],
        "ICC3_1_CI95_high": row_single["CI95%"][1],
        "ICC3_k": row["ICC"],
        "ICC3_k_CI95_low": row["CI95%"][0],
        "ICC3_k_CI95_high": row["CI95%"][1],
    }


icc_results = [compute_icc(wide_df, label="All conditions")]

for cond in condition_order:
    sub = wide_df[wide_df["condition"] == cond]
    if len(sub) >= 3:
        icc_results.append(compute_icc(sub, label=cond))

icc_df = pd.DataFrame(icc_results)
print("\n=== ICC Results (Path Length) ===")
print(icc_df.to_string(index=False, float_format="%.3f"))

# ============================================================
# 4. Bland-Altman plots — per condition + pooled
# ============================================================
wide_df["ba_mean"] = (wide_df["qualisys"] + wide_df["mediapipe"]) / 2
wide_df["ba_diff"] = wide_df["mediapipe"] - wide_df["qualisys"]

ba_groups = ["All conditions"] + condition_order
n_panels = len(ba_groups)

fig = make_subplots(
    rows=1,
    cols=n_panels,
    subplot_titles=ba_groups,
    shared_yaxes=True,
)


def add_ba_panel(fig, sub, col, show_legend=False):
    """Add one Bland-Altman panel."""
    bias = sub["ba_diff"].mean()
    sd = sub["ba_diff"].std()
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    x_range = [sub["ba_mean"].min(), sub["ba_mean"].max()]
    x_pad = (x_range[1] - x_range[0]) * 0.1 or 0.01
    x_lo, x_hi = x_range[0] - x_pad, x_range[1] + x_pad

    # Data points
    fig.add_trace(
        go.Scatter(
            x=sub["ba_mean"],
            y=sub["ba_diff"],
            mode="markers",
            marker=dict(color="#1f77b4", size=8, opacity=0.7),
            name="Participant",
            showlegend=show_legend,
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Mean: %{x:.4f} mm<br>"
                "Diff: %{y:.4f} mm<extra></extra>"
            ),
            customdata=sub[["participant_code"]].values,
        ),
        row=1, col=col,
    )

    # Bias line
    fig.add_shape(
        type="line", x0=x_lo, x1=x_hi, y0=bias, y1=bias,
        line=dict(color="black", width=1.5, dash="dash"),
        xref=f"x{col}" if col > 1 else "x",
        yref=f"y{col}" if col > 1 else "y",
    )

    # LoA lines
    for loa in [loa_upper, loa_lower]:
        fig.add_shape(
            type="line", x0=x_lo, x1=x_hi, y0=loa, y1=loa,
            line=dict(color="red", width=1, dash="dot"),
            xref=f"x{col}" if col > 1 else "x",
            yref=f"y{col}" if col > 1 else "y",
        )

    # Annotate bias and LoA values on the right edge
    xref = f"x{col}" if col > 1 else "x"
    yref = f"y{col}" if col > 1 else "y"
    anno_x = x_hi
    for val, label, color in [
        (bias, f"Bias: {bias:.4f}", "black"),
        (loa_upper, f"+1.96 SD: {loa_upper:.4f}", "red"),
        (loa_lower, f"−1.96 SD: {loa_lower:.4f}", "red"),
    ]:
        fig.add_annotation(
            x=anno_x, y=val, text=label,
            xref=xref, yref=yref,
            showarrow=False, font=dict(size=9, color=color),
            xanchor="left", yanchor="bottom",
        )

    return bias, sd, loa_lower, loa_upper


ba_stats = []
for i, group in enumerate(ba_groups):
    col = i + 1
    if group == "All conditions":
        sub = wide_df
    else:
        sub = wide_df[wide_df["condition"] == group]

    bias, sd, loa_lo, loa_hi = add_ba_panel(fig, sub, col, show_legend=(col == 1))
    ba_stats.append({
        "condition": group,
        "n": len(sub),
        "bias": bias,
        "sd_diff": sd,
        "loa_lower": loa_lo,
        "loa_upper": loa_hi,
    })

fig.update_yaxes(title_text="MediaPipe − Qualisys (mm)", row=1, col=1)
for c in range(1, n_panels + 1):
    fig.update_xaxes(title_text="Mean of systems (mm)", row=1, col=c)

fig.update_layout(
    height=450,
    width=1200,
    template="simple_white",
    title_text="Bland-Altman: COM Path Length (MediaPipe vs. Qualisys)",
    margin=dict(t=80, b=80, r=120),
    showlegend=False,
)
fig.update_annotations(font_size=11)
fig.show()

# Print BA summary table
ba_stats_df = pd.DataFrame(ba_stats)
print("\n=== Bland-Altman Summary (Path Length) ===")
print(ba_stats_df.to_string(index=False, float_format="%.4f"))

# ============================================================
# 5. (Optional) Export
# ============================================================
# fig.write_image("bland_altman_path_length.png", width=1200, height=450, scale=2)
# fig.write_image("bland_altman_path_length.pdf", width=1200, height=450, scale=2)
# icc_df.to_csv("icc_path_length.csv", index=False)
# ba_stats_df.to_csv("bland_altman_stats_path_length.csv", index=False)