from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# FIGURE OPTIONS (INCHES)
# =========================
FIG_WIDTH_IN  = 2
FIG_HEIGHT_IN = 1.5
DPI           = 300

FIG_WIDTH_PX  = int(FIG_WIDTH_IN  * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)

# ------------------------
# Data loading (single FMC system: RTMPOSE_DLC)
# ------------------------
QUALISYS_REL   = Path("validation/qualisys/gait_parameters/qualisys_gait_metrics.csv")
RTMPOSE_DLC_REL = Path("validation/rtmpose_dlc/gait_parameters/gait_metrics.csv")


def load_gait_dataframe_multi(
    folder_list: list[Path],
    qualisys_rel: Path = QUALISYS_REL,
    fmc_system_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """
    Load gait-metrics CSVs from multiple systems and concatenate into one long-form dataframe.

    Expected columns in each CSV (at minimum):
      - metric
      - side
      - event_index
      - value
    """
    if fmc_system_paths is None:
        fmc_system_paths = {"rtmpose_dlc": RTMPOSE_DLC_REL}

    rows: list[pd.DataFrame] = []
    skipped: list[tuple[str, list[str]]] = []

    for folder in folder_list:
        required_paths = {
            "qualisys": folder / qualisys_rel,
            **{k: folder / v for k, v in fmc_system_paths.items()},
        }

        missing = [k for k, p in required_paths.items() if not p.exists()]
        if missing:
            skipped.append((folder.stem, missing))
            continue

        # -----------------
        # Qualisys
        # -----------------
        qtmp = pd.read_csv(required_paths["qualisys"])
        qtmp["system"] = "qualisys"  # IMPORTANT (ensures pairing works)
        qtmp["trial_name"] = folder.stem
        rows.append(qtmp)

        # -----------------
        # FMC systems
        # -----------------
        for sys_name, fpath in required_paths.items():
            if sys_name == "qualisys":
                continue
            tmp = pd.read_csv(fpath)
            tmp["system"] = sys_name
            tmp["trial_name"] = folder.stem
            rows.append(tmp)

    if not rows:
        raise RuntimeError("No valid sessions found with complete system coverage.")

    if skipped:
        print("Skipped sessions (missing required systems):")
        for name, systems in skipped:
            print(f"  {name}: missing {systems}")

    df = pd.concat(rows, ignore_index=True)

    # types
    if "event_index" in df.columns:
        df["event_index"] = df["event_index"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def get_paired_stride_df(
    df: pd.DataFrame,
    metric: str,
    side: str,
    reference_system: str = "qualisys",
    fmc_system: str = "rtmpose_dlc",
) -> pd.DataFrame:
    """
    Return per-stride paired values for (reference_system vs fmc_system),
    matched on trial_name, side, metric, event_index.
    """
    d = df[(df["metric"] == metric) & (df["side"] == side)].copy()

    systems = sorted(d["system"].dropna().unique().tolist())
    if reference_system not in systems:
        raise ValueError(f"Reference system '{reference_system}' not found in data systems: {systems}")
    if fmc_system not in systems:
        raise ValueError(f"FMC system '{fmc_system}' not found in data systems: {systems}")

    ref = d[d["system"] == reference_system].rename(columns={"value": reference_system})[
        ["trial_name", "side", "metric", "event_index", reference_system]
    ]
    fmc = d[d["system"] == fmc_system].rename(columns={"value": "freemocap"})[
        ["trial_name", "side", "metric", "event_index", "freemocap"]
    ]

    paired = ref.merge(fmc, on=["trial_name", "side", "metric", "event_index"], how="inner")
    paired["diff"] = paired["freemocap"] - paired[reference_system]
    return paired


# ------------------------
# Plotly helpers
# ------------------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color like #1f77b4, got: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_histogram_overlay_panel(
    fig: go.Figure,
    dfs: dict[str, pd.DataFrame],
    colors: dict[str, str],
    *,
    row: int,
    col: int,
    ncols: int,
    title: str,
    max_frames: int = 50,
    show_ylabel: bool = True,
    fps: float = 30.0,
    alpha_fill: float = 0.4,
    outline_width: float = 1.4,
) -> None:
    frame_dt = 1.0 / float(fps)
    max_abs = max_frames * frame_dt

    # bins
    bin_width = frame_dt
    edges = np.arange(-max_abs - bin_width / 2.0, max_abs + bin_width + 1e-12, bin_width)

        
    # ticks (ms)
    # -------------------------
    # Neat x-axis ticks (ms) WITHOUT changing the axis range
    # -------------------------
    max_ms = max_abs * 1000.0  # uses your existing max_abs from max_frames/fps

    # Aim for ~5–7 labels across the whole width
    target_labels = 7
    raw_step = (2.0 * max_ms) / max(1, (target_labels - 1))

    nice_steps = np.array([50, 100, 200, 250, 500, 1000], dtype=float)
    tick_step = float(nice_steps[np.argmin(np.abs(nice_steps - raw_step))])

    max_ms_rounded = float(np.ceil(max_ms / tick_step) * tick_step)

    major_ticks_ms = np.arange(
        -max_ms_rounded,
         max_ms_rounded + tick_step,
         tick_step,
        dtype=float
    )

    tickvals = (major_ticks_ms / 1000.0).tolist()
    ticktext = [f"{int(ms)}" for ms in major_ticks_ms]


    ymax = 1

    for label, dfi in dfs.items():
        x = dfi["diff"].to_numpy()
        x = x[np.isfinite(x)]

        counts, bin_edges = np.histogram(x, bins=edges)
        ymax = max(ymax, int(counts.max()) if counts.size else 1)

        lefts = bin_edges[:-1]
        rights = bin_edges[1:]

        xs = np.column_stack([lefts, rights]).ravel()
        ys = np.column_stack([counts, counts]).ravel()

        xs_poly = np.concatenate(([xs[0]], xs, [xs[-1]]))
        ys_poly = np.concatenate(([0], ys, [0]))

        color = colors[label]
        fill_rgba = hex_to_rgba(color, alpha_fill)

        # fill
        fig.add_trace(
            go.Scatter(
                x=xs_poly,
                y=ys_poly,
                mode="lines",
                fill="tozeroy",
                line=dict(width=0),
                fillcolor=fill_rgba,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col
        )

        # outline
        fig.add_trace(
            go.Scatter(
                x=xs_poly,
                y=ys_poly,
                mode="lines",
                line=dict(color=color, width=outline_width),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col
        )

        # μ ± σ (ms) sanity print
        mu = float(np.mean(x) * 1000.0) if x.size else float("nan")
        sd = float(np.std(x, ddof=1) * 1000.0) if x.size >= 2 else float("nan")
        print(f"{title} | {label}: mu={mu:+.1f} ms, sd={sd:.1f} ms, n={x.size}")

    # headroom
    fig.update_yaxes(range=[0, 350], row=row, col=col)

    # zero ref line
    fig.add_vline(
        x=0.0,
        line=dict(color="black", width=1.0, dash="dash"),
        opacity=0.35,
        row=row, col=col
    )

    # bold subplot title
    fig.layout.annotations[(row - 1) * ncols + (col - 1)].update(text=f"<b>{title}</b>")

    # axes
    fig.update_xaxes(
        range=[-max_abs, max_abs],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=0,
        tickfont=dict(size=10),
        row=row, col=col
    )

    fig.update_yaxes(
        title_text="<b>Count</b>" if show_ylabel else "",
        tickfont=dict(size=16),
        row=row, col=col
    )

    fig.update_xaxes(title_text="<b>Error (ms)</b>", row=row, col=col)

    # no grids
    fig.update_yaxes(showgrid=False, gridcolor="rgba(0,0,0,0.10)", row=row, col=col)
    fig.update_xaxes(showgrid=False, row=row, col=col)


# ------------------------
# RUN
# ------------------------
path_to_recordings = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]

list_of_valid_folders: list[Path] = []
for p in list_of_folders:
    if (p / "validation").is_dir():
        list_of_valid_folders.append(p)
    else:
        print(f"Skipping {p}")

gait_param_df = load_gait_dataframe_multi(
    list_of_valid_folders,
    fmc_system_paths={"rtmpose_dlc": RTMPOSE_DLC_REL},
)

print("Systems present:", sorted(gait_param_df["system"].dropna().unique()))

# Paired diffs: RTMPOSE_DLC vs QUALISYS, left and right
paired_stance_left  = get_paired_stride_df(gait_param_df, metric="stance_duration", side="left",  fmc_system="rtmpose_dlc")
paired_stance_right = get_paired_stride_df(gait_param_df, metric="stance_duration", side="right", fmc_system="rtmpose_dlc")

paired_swing_left   = get_paired_stride_df(gait_param_df, metric="swing_duration",  side="left",  fmc_system="rtmpose_dlc")
paired_swing_right  = get_paired_stride_df(gait_param_df, metric="swing_duration",  side="right", fmc_system="rtmpose_dlc")

pooled_errors = np.concatenate([
    paired_stance_left["diff"].dropna().values,
    paired_stance_right["diff"].dropna().values,
    paired_swing_left["diff"].dropna().values,
    paired_swing_right["diff"].dropna().values,
])
cutoff_sec = np.nanpercentile(np.abs(pooled_errors), 99)

n_total = len(pooled_errors)
n_outside = np.sum(np.abs(pooled_errors) > cutoff_sec)
print(f"Display cutoff: ±{cutoff_sec*1000:.0f} ms")
print(f"Strides outside window: {n_outside}/{n_total} ({100*n_outside/n_total:.1f}%)")


LEFT_LEG_LABEL = "Non-prosthetic leg (FreeMoCap-RTMPose)"
RIGHT_LEG_LABEL = "Prosthetic leg (FreeMoCap-DLC)"
# Colors keyed by labels we plot
colors = {
    LEFT_LEG_LABEL: "#ff7f0e",
    RIGHT_LEG_LABEL: "#1f77b4",
}

fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Stance Duration", "Swing Duration"),
)

# legend entries (dummy traces) for a clean global legend
for label in [LEFT_LEG_LABEL, RIGHT_LEG_LABEL]:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=colors[label], width=3),
            name=label,
            showlegend=True,
        )
    )

add_histogram_overlay_panel(
    fig,
    dfs={
        LEFT_LEG_LABEL: paired_stance_left,
        RIGHT_LEG_LABEL: paired_stance_right,
    },
    colors=colors,
    row=1, col=1, ncols=2,
    title="Stance Duration",
    show_ylabel=True,
    max_frames=int(np.ceil(cutoff_sec * 30.0)) ,
)

add_histogram_overlay_panel(
    fig,
    dfs={
        LEFT_LEG_LABEL: paired_swing_left,
        RIGHT_LEG_LABEL: paired_swing_right,
    },
    colors=colors,
    row=1, col=2, ncols=2,
    title="Swing Duration",
    show_ylabel=False,
    max_frames=int(np.ceil(cutoff_sec * 30.0)) ,
)

fig.update_layout(
    width=FIG_WIDTH_PX,
    height=FIG_HEIGHT_PX,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=70),
    legend=dict(
        orientation="h",
        x=0.5,
        y=-0.15,
        xanchor="center",
        yanchor="top",
        font=dict(size=14),
    ),
)

# boxed axes
fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

fig.show()

# export
import plotly.io as pio
pio.kaleido.scope.mathjax = None
path_to_save = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
fig.write_image(path_to_save / "gait_error_histogram_left_right.pdf")
fig.write_image(path_to_save / "gait_error_histogram_left_right.png", scale=3)
