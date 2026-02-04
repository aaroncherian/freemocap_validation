from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# FIGURE OPTIONS
# =========================
FIG_WIDTH_IN  = 2.75    # inches  (≈ single-column / two-panel width)
FIG_HEIGHT_IN = 1.25   # inches
DPI           = 300     # for static export only


FIG_WIDTH_PX  = int(FIG_WIDTH_IN  * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)

# ------------------------
# Data loading (unchanged)
# ------------------------

QUALISYS_REL = Path("validation/qualisys/gait_parameters/qualisys_gait_metrics.csv")
FMC_REL      = Path("validation/mediapipe_dlc/gait_parameters/gait_metrics.csv")

def load_gait_dataframe(
        folder_list: list,
        qualisys_rel: Path = QUALISYS_REL,
        fmc_rel: Path = FMC_REL
) -> pd.DataFrame:

    rows = []

    for folder in folder_list:
        for fpath in [folder/qualisys_rel, folder/fmc_rel]:
            if not fpath.exists():
                raise FileNotFoundError(f"Missing CSV: {fpath}")

            tmp = pd.read_csv(fpath)

            required = {"system", "side", "metric", "event_index", "value"}
            missing = required - set(tmp.columns)
            if missing:
                raise ValueError(f"{fpath} missing columns: {missing}")

            tmp["trial_name"] = folder.stem
            rows.append(tmp)

    df = pd.concat(rows, ignore_index=True)
    df["event_index"] = df["event_index"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def get_paired_stride_df(
        df: pd.DataFrame,
        metric: str,
        side: str = "right",
        reference_system: str = "qualisys",
        fmc_system: str | None = None
) -> pd.DataFrame:

    d = df[(df["metric"] == metric) & (df["side"] == side)].copy()

    systems = sorted(d["system"].dropna().unique().tolist())
    if reference_system not in systems:
        raise ValueError(f"Reference system '{reference_system}' not found in data systems: {systems}")

    if fmc_system is None:
        others = [s for s in systems if s != reference_system]
        if len(others) != 1:
            raise ValueError(f"Could not auto-detect fmc_system; found systems: {systems}")
        fmc_system = others[0]

    qual = d[d["system"] == reference_system].rename(columns={"value": reference_system})[
        ["trial_name", "side", "metric", "event_index", reference_system]
    ]

    fmc = d[d["system"] == fmc_system].rename(columns={"value": "freemocap"})[
        ["trial_name", "side", "metric", "event_index", "freemocap"]
    ]

    paired = qual.merge(
        fmc,
        on=["trial_name", "side", "metric", "event_index"],
        how="inner"
    )

    paired["diff"] = paired["freemocap"] - paired[reference_system]
    return paired


# ------------------------
# Plotly histogram version
# ------------------------

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex like #1f77b4, got: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def add_histogram_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    col: int,
    title: str,
    max_frames: int = 7,
    show_ylabel: bool = True,
    fps: float = 30.0,
    fill_color: str = "#1f77b4",
    edge_color: str = "#0a3f64",
    fill_alpha: float = 0.45,
) -> None:
    x = df["diff"].to_numpy()
    x = x[np.isfinite(x)]

    frame_dt = 1.0 / float(fps)
    max_abs = max_frames * frame_dt

    # Bin edges centered like your mpl approach
    bin_width = frame_dt
    edges = np.arange(-max_abs - bin_width / 2.0, max_abs + bin_width + 1e-12, bin_width)

    # Histogram counts
    counts, bin_edges = np.histogram(x, bins=edges)

    ymax = int(counts.max()) if counts.size else 1
    fig.update_yaxes(range=[0, ymax * 1.08], row=row, col=col)
    lefts = bin_edges[:-1]
    rights = bin_edges[1:]

    # Build a single filled "step" polygon so there are NO internal bar seams
    # x: L0, R0, L1, R1, ..., Ln, Rn
    # y: c0, c0, c1, c1, ..., cn, cn
    xs = np.column_stack([lefts, rights]).ravel()
    ys = np.column_stack([counts, counts]).ravel()

    # Close the polygon down to y=0 at ends
    xs_poly = np.concatenate(([xs[0]], xs, [xs[-1]]))
    ys_poly = np.concatenate(([0], ys, [0]))

    # Filled polygon
    fig.add_trace(
        go.Scatter(
            x=xs_poly,
            y=ys_poly,
            mode="lines",
            fill="tozeroy",
            line=dict(width=0),          # no line on the fill trace
            fillcolor=hex_to_rgba(fill_color, fill_alpha),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row, col=col
    )

    # Outline (single dark border around the whole histogram silhouette)
    fig.add_trace(
        go.Scatter(
            x=xs_poly,
            y=ys_poly,
            mode="lines",
            line=dict(color=edge_color, width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row, col=col
    )

    # Vertical reference at 0
    fig.add_vline(
        x=0.0,
        line=dict(color=edge_color, width=1.2, dash="dash"),
        opacity=0.4,
        row=row, col=col
    )

    # Title bold (subplot_titles created by make_subplots)
    fig.layout.annotations[(row - 1) * 2 + (col - 1)].update(text=f"<b>{title}</b>")

    # Axis range and ticks at 50ms intervals (within bounds)
    max_ms = max_abs * 1000.0
    major_ticks_ms = np.arange(-250, 251, 50, dtype=float)
    major_ticks_ms = major_ticks_ms[np.abs(major_ticks_ms) <= max_ms + 1e-9]
    tickvals = (major_ticks_ms / 1000.0).tolist()

    fig.update_xaxes(
        range=[-max_abs, max_abs],
        tickmode="array",
        tickvals=tickvals,
        ticktext=[("0" if ms == 0 else f"{ms:+.0f}") for ms in major_ticks_ms],
        row=row, col=col
    )

    fig.update_yaxes(title_text="<b>Count</b>" if show_ylabel else "", row=row, col=col)

    # µ ± sd annotation (ms)
    bias_ms = float(np.mean(x) * 1000.0) if x.size else float("nan")
    sd_ms   = float(np.std(x, ddof=1) * 1000.0) if x.size >= 2 else float("nan")
    label = f"μ = {bias_ms:+.0f} ± {sd_ms:.0f} ms" if np.isfinite(bias_ms) and np.isfinite(sd_ms) else "μ = n/a"

    # Axis-domain refs: x, x2, y, y2 ... (works for 1x2)
    axis_suffix = "" if (row == 1 and col == 1) else str((row - 1) * 2 + col)
    fig.add_annotation(
        x=0.03, y=0.93,
        xref=f"x{axis_suffix} domain",
        yref=f"y{axis_suffix} domain",
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=11),
    )

    # Light y-grid like mpl
    fig.update_yaxes(showgrid=False, gridcolor="rgba(0,0,0,0.08)", row=row, col=col)
    fig.update_xaxes(showgrid=False, row=row, col=col)


# ------------------------
# Run
# ------------------------

path_to_recordings = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]

list_of_valid_folders = []
for p in list_of_folders:
    if (p / "validation").is_dir():
        list_of_valid_folders.append(p)
    else:
        print(f"Skipping {p}")

gait_param_df = load_gait_dataframe(list_of_valid_folders)

paired_stance = get_paired_stride_df(
    df=gait_param_df,
    metric="stance_duration",
    side="right"
)

paired_swing = get_paired_stride_df(
    df=gait_param_df,
    metric="swing_duration",
    side="right"
)

fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Stance Duration", "Swing Duration"),
    horizontal_spacing=0.03
)

add_histogram_panel(fig, paired_stance, row=1, col=1, title="Stance Duration", show_ylabel=True)
add_histogram_panel(fig, paired_swing,  row=1, col=2, title="Swing Duration",  show_ylabel=False)

# Single centered x-axis label (Plotly supports a global x title)
fig.update_layout(
    width=FIG_WIDTH_PX,
    height=FIG_HEIGHT_PX,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=70),
)

fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

# x labels on BOTH
fig.update_xaxes(title_text="<b>Error (ms)</b>", row=1, col=1)
fig.update_xaxes(title_text="<b>Error (ms)</b>", row=1, col=2)

import plotly.io as pio
pio.kaleido.scope.mathjax = None
path_to_save = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
fig.write_image(path_to_save / "prosthetic_gait_error_histogram.pdf")
fig.show()

# OPTIONAL: export (requires kaleido: pip install -U kaleido)
# scale = DPI / 96  # plotly's "scale" is relative to ~96dpi
# fig.write_image("gait_error_hist.png", width=FIG_WIDTH_PX, height=FIG_HEIGHT_PX, scale=scale)
