import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import chi2

# =========================
# Paper-ready figure params
# =========================
DPI = 300
FIG_W_IN = 7
FIG_H_IN = 2.4
FIG_W_PX = int(FIG_W_IN * DPI)
FIG_H_PX = int(FIG_H_IN * DPI)

EXPORT_BASENAME = "com_statokinesiogram_mean_centered"
root_path = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\balance\figures")
root_path.mkdir(exist_ok=True, parents=True)

participant_to_use = "ATC"
trial_name = "2025-11-04_15-18-21_GMT-5_atc_nih_2"
tracker_to_use = "vitpose"

# -------------------
# Helpers
# -------------------
# def compute_confidence_ellipse(x, y, n_points=200, conf_scale=2.4477):
#     """
#     95% confidence ellipse for 2D data.
#     conf_scale = sqrt(chi2.ppf(0.95, df=2)) ≈ 2.4477
#     """
#     xy = np.vstack([x, y])
#     cov = np.cov(xy)

#     eigvals, eigvecs = np.linalg.eigh(cov)
#     order = eigvals.argsort()[::-1]
#     eigvals = eigvals[order]
#     eigvecs = eigvecs[:, order]

#     theta = np.linspace(0, 2 * np.pi, n_points)
#     circle = np.vstack([np.cos(theta), np.sin(theta)])

#     ellipse = eigvecs @ np.diag(conf_scale * np.sqrt(eigvals)) @ circle
#     return ellipse[0], ellipse[1]

def compute_confidence_ellipse(x, y):
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    chi2_val = chi2.ppf(0.95, df=2)

    a = np.sqrt(eigenvalues[0] * chi2_val)
    b = np.sqrt(eigenvalues[1] * chi2_val)
    area = np.pi * a * b

    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a * np.cos(t)
    ellipse_y = b * np.sin(t)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    ellipse_points = R @ np.array([ellipse_x, ellipse_y])

    return ellipse_points, a, b, theta, area

def compute_ellipse_area(x, y, conf_scale=2.4477):
    """95% confidence ellipse area in mm²."""
    cov = np.cov(x, y)
    eigvals = np.linalg.eigvalsh(cov)
    a = conf_scale * np.sqrt(eigvals[1])  # larger
    b = conf_scale * np.sqrt(eigvals[0])  # smaller
    return np.pi * a * b

# -------------------
# Load data from DB
# -------------------
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
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%balance_positions'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)
conn.close()

# dfs = []
# for _, row in path_df.iterrows():
#     sub_df = pd.read_csv(row["path"])
#     sub_df["participant_code"] = row["participant_code"]
#     sub_df["trial_name"] = row["trial_name"]
#     sub_df["condition"] = row.get("condition") or ""
#     sub_df["tracker"] = row["tracker"]
#     dfs.append(sub_df)

# final_df = pd.concat(dfs, ignore_index=True)

# participant_df = final_df.query(
#     "participant_code == @participant_to_use and trial_name == @trial_name and tracker == @tracker_to_use"
# ).copy()

conditions = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

short_titles = [
    "EO / Solid",
    "EC / Solid",
    "EO / Foam",
    "EC / Foam",
]

# # -------------------
# # Build centered data
# # -------------------
# plot_data = {}

# for condition in conditions:
#     x_col = f"{condition}_x"
#     y_col = f"{condition}_y"

#     x_raw = participant_df[x_col].to_numpy()
#     y_raw = participant_df[y_col].to_numpy()

#     # mean-center for statokinesiogram
#     x_center = np.mean(x_raw)
#     y_center = np.mean(y_raw)

#     x = x_raw - x_center
#     y = y_raw - y_center

#     start_x = x[0]
#     start_y = y[0]

#     ellipse_x, ellipse_y = compute_confidence_ellipse(x, y)

#     plot_data[condition] = {
#         "x": x,
#         "y": y,
#         "start_x": start_x,
#         "start_y": start_y,
#         "ellipse_x": ellipse_x,
#         "ellipse_y": ellipse_y,
#     }

# # Shared symmetric axis limits across all panels
# all_x = np.concatenate(
#     [plot_data[c]["x"] for c in conditions] +
#     [plot_data[c]["ellipse_x"] for c in conditions]
# )
# all_y = np.concatenate(
#     [plot_data[c]["y"] for c in conditions] +
#     [plot_data[c]["ellipse_y"] for c in conditions]
# )

# max_extent = np.max(np.abs(np.concatenate([all_x, all_y])))
# axis_limit = np.ceil(max_extent + 1)

# for condition in conditions:
#     x = participant_df[f"{condition}_x"].to_numpy()
#     y = participant_df[f"{condition}_y"].to_numpy()

#     x_c = x - np.mean(x)
#     y_c = y - np.mean(y)

#     path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
#     ml_sd = np.std(x_c)
#     ap_sd = np.std(y_c)

#     print(condition)
#     print(f"  ML SD: {ml_sd:.2f} mm")
#     print(f"  AP SD: {ap_sd:.2f} mm")
#     print(f"  Path length: {path_length:.2f} mm")

# # -------------------
# # Plot
# # -------------------
# fig = make_subplots(
#     rows=1,
#     cols=4,
#     subplot_titles=short_titles,
#     horizontal_spacing=0.06,
# )

# for i, condition in enumerate(conditions, start=1):
#     x = plot_data[condition]["x"]
#     y = plot_data[condition]["y"]
#     ex = plot_data[condition]["ellipse_x"]
#     ey = plot_data[condition]["ellipse_y"]
#     sx = plot_data[condition]["start_x"]
#     sy = plot_data[condition]["start_y"]

#     # COM path
#     fig.add_trace(
#         go.Scatter(
#             x=x,
#             y=y,
#             mode="lines",
#             line=dict(width=2),
#             showlegend=False,
#             hoverinfo="skip",
#         ),
#         row=1,
#         col=i,
#     )

#     # 95% ellipse
#     fig.add_trace(
#         go.Scatter(
#             x=ex,
#             y=ey,
#             mode="lines",
#             line=dict(width=2, dash="dash"),
#             showlegend=False,
#             hoverinfo="skip",
#         ),
#         row=1,
#         col=i,
#     )

#     # Mean center marker
#     fig.add_trace(
#         go.Scatter(
#             x=[0],
#             y=[0],
#             mode="markers",
#             marker=dict(symbol="square", size=7),
#             showlegend=False,
#             hoverinfo="skip",
#         ),
#         row=1,
#         col=i,
#     )

#     # Optional: start marker relative to mean center
#     fig.add_trace(
#         go.Scatter(
#             x=[sx],
#             y=[sy],
#             mode="markers",
#             marker=dict(symbol="circle", size=5),
#             showlegend=False,
#             hoverinfo="skip",
#         ),
#         row=1,
#         col=i,
#     )

#     # Crosshair at mean center
#     fig.add_hline(y=0, line_width=1, line_dash="dot", row=1, col=i)
#     fig.add_vline(x=0, line_width=1, line_dash="dot", row=1, col=i)

#     fig.update_xaxes(
#         title_text="ML displacement (mm)",
#         range=[-axis_limit, axis_limit],
#         zeroline=False,
#         row=1,
#         col=i,
#     )

#     fig.update_yaxes(
#         title_text="AP displacement (mm)" if i == 1 else None,
#         range=[-axis_limit, axis_limit],
#         scaleanchor=f"x{i}",
#         scaleratio=1,
#         zeroline=False,
#         row=1,
#         col=i,
#     )

# fig.update_layout(
#     width=FIG_W_PX,
#     height=FIG_H_PX,
#     template="simple_white",
#     margin=dict(l=20, r=20, t=40, b=20),
# )

# fig.show()

# fig.write_image(root_path / f"{EXPORT_BASENAME}.png", scale=1)
# fig.write_image(root_path / f"{EXPORT_BASENAME}.pdf")

dfs = []
for _, row in path_df.iterrows():
    sub_df = pd.read_csv(row["path"])
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    dfs.append(sub_df)

final_df = pd.concat(dfs, ignore_index=True)

results = []
for (participant, trial, tracker), grp in final_df.groupby(
    ["participant_code", "trial_name", "tracker"]
):
    for condition in conditions:
        x_col = f"{condition}_x"
        y_col = f"{condition}_y"
        if x_col not in grp.columns:
            continue

        x = grp[x_col].to_numpy()
        y = grp[y_col].to_numpy()

        # mean-center
        x = x - np.mean(x)
        y = y - np.mean(y)

        _, a, b, theta, area = compute_confidence_ellipse(x, y)

        results.append({
            "participant": participant,
            "trial": trial,
            "tracker": tracker,
            "condition": condition,
            "ellipse_area_mm2": area,
        })

area_df = pd.DataFrame(results)

summary = (
    area_df
    .groupby(["tracker", "condition", "participant"])["ellipse_area_mm2"]
    .mean()
    .reset_index()
    .groupby(["tracker", "condition"])["ellipse_area_mm2"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

print(summary.to_string(index=False))

# --- Path length data ---
query_pl = """
SELECT t.participant_code, 
       t.trial_name,
       a.path,
       a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
  AND a.category = "com_analysis"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.component_name LIKE '%path_length_com'
ORDER BY t.trial_name, a.path;
"""

conn = sqlite3.connect("validation.db")
pl_path_df = pd.read_sql_query(query_pl, conn)
conn.close()

pl_dfs = []
for _, row in pl_path_df.iterrows():
    sub_df = pd.read_json(row["path"])
    sub_df = sub_df.rename(columns={
        "Frame Intervals": "frame_interval",
        "Path Lengths:": "path_length",
    }).reset_index().rename(columns={"index": "condition"})

    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["tracker"] = row["tracker"]
    pl_dfs.append(sub_df)

pl_df = pd.concat(pl_dfs, ignore_index=True)

# Path length summary
pl_summary = (
    pl_df
    .groupby(["tracker", "condition", "participant_code"])["path_length"]
    .mean()
    .reset_index()
    .groupby(["tracker", "condition"])["path_length"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "pl_mean", "std": "pl_std"})
    .reset_index()
)

# Ellipse area summary (from your existing area_df)
ea_summary = (
    area_df
    .groupby(["tracker", "condition", "participant"])["ellipse_area_mm2"]
    .mean()
    .reset_index()
    .groupby(["tracker", "condition"])["ellipse_area_mm2"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "ea_mean", "std": "ea_std"})
    .reset_index()
)

# Merge
summary = pl_summary.merge(ea_summary, on=["tracker", "condition"])

# Format for display
summary["path_length"] = summary.apply(
    lambda r: f"{r['pl_mean']:.1f} ± {r['pl_std']:.1f}", axis=1
)
summary["ellipse_area"] = summary.apply(
    lambda r: f"{r['ea_mean']:.1f} ± {r['ea_std']:.1f}", axis=1
)

print(summary[["tracker", "condition", "path_length", "ellipse_area"]].to_string(index=False))

# =========================
# Ellipse area line plot
# =========================
from plotly.subplots import make_subplots
import plotly.graph_objects as go

condition_order = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

display_x_short = ["EO-S", "EC-S", "EO-F", "EC-F"]

TRACKERS = ["qualisys", "mediapipe", "rtmpose", "vitpose"]

sub_title = {
    "mediapipe": "FMC-MediaPipe",
    "qualisys": "Reference",
    "rtmpose":  "FMC-RTMPose",
    "vitpose":  "FMC-ViTPose",
}

col_for = {trk: i + 1 for i, trk in enumerate(TRACKERS)}

# Copy and standardize participant column name
ellipse_plot_df = area_df.copy().rename(columns={"participant": "participant_code"})
ellipse_plot_df["condition"] = pd.Categorical(
    ellipse_plot_df["condition"],
    categories=condition_order,
    ordered=True
)

fig = make_subplots(
    rows=1,
    cols=len(TRACKERS),
    shared_yaxes=True,
    subplot_titles=tuple(sub_title[t] for t in TRACKERS),
    horizontal_spacing=0.05,
)

# Individual trial lines
for tracker in TRACKERS:
    dft = ellipse_plot_df[ellipse_plot_df["tracker"] == tracker].copy()
    if dft.empty:
        continue

    dft["trial_id"] = dft["participant_code"] + " | " + dft["trial"]

    for trial_id, sub in dft.groupby("trial_id", sort=False):
        s = (
            sub.groupby("condition")["ellipse_area_mm2"]
               .mean()
               .reindex(condition_order)
        )

        fig.add_trace(
            go.Scatter(
                x=display_x_short,
                y=s.values,
                mode="lines+markers",
                line=dict(color="rgba(150,150,150,0.4)", width=2),
                marker=dict(size=5, color="rgba(150,150,150,0.5)"),
                showlegend=False,
                hovertemplate=(
                    f"{trial_id}<br>%{{x}}"
                    "<br>Ellipse area: %{y:.1f} mm²<extra></extra>"
                ),
            ),
            row=1,
            col=col_for[tracker],
        )

# Mean ± SD overlay
agg = (
    ellipse_plot_df
    .groupby(["tracker", "condition"])["ellipse_area_mm2"]
    .agg(["mean", "std"])
)

for tracker in TRACKERS:
    if tracker not in agg.index.get_level_values(0):
        continue

    sub = agg.loc[tracker].reindex(condition_order)
    means = sub["mean"].to_numpy()
    stds = sub["std"].to_numpy()

    fig.add_trace(
        go.Scatter(
            x=display_x_short,
            y=means,
            mode="lines+markers",
            line=dict(color="black", width=2.5),
            marker=dict(color="black", size=7),
            showlegend=False,
            error_y=dict(
                type="data",
                array=stds,
                visible=True,
                thickness=2.5,
                width=4,
            ),
            hovertemplate=(
                "%{x}<br>Mean: %{y:.1f} mm²"
                "<br>SD: %{customdata:.1f} mm²<extra></extra>"
            ),
            customdata=stds,
        ),
        row=1,
        col=col_for[tracker],
    )

fig.update_layout(
    height=500,
    width=1200,
    template="simple_white",
    margin=dict(l=100, r=20, t=30, b=70),
    font=dict(
        family="Arial",
        size=14,
    ),
)

for ann in fig.layout.annotations:
    ann.update(
        font=dict(family="Arial", size=28),
        xanchor="center",
    )

fig.update_yaxes(
    title_text="<b>Ellipse Area (mm²)</b>",
    title_font=dict(size=28),
    tickfont=dict(size=22),
    row=1,
    col=1,
)

for c in range(1, len(TRACKERS) + 1):
    fig.update_xaxes(
        tickfont=dict(size=22),
        title_text="",
        row=1,
        col=c,
    )

fig.show()

fig.write_image(root_path / "com_ellipse_area.svg", scale=3)
# fig.write_image(root_path / "com_ellipse_area.png", scale=3)
# fig.write_image(root_path / "com_ellipse_area.pdf")