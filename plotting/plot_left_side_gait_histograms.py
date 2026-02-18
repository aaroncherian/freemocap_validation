from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# FIGURE OPTIONS (INCHES)
# =========================
FIG_WIDTH_IN  = 2.5
FIG_HEIGHT_IN = 1.25
DPI           = 300

FIG_WIDTH_PX  = int(FIG_WIDTH_IN  * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)


# ------------------------
# Data loading (multi-system)
# ------------------------

QUALISYS_REL = Path("validation/qualisys/gait_parameters/qualisys_gait_metrics.csv")
MP_DLC_REL   = Path("validation/mediapipe_dlc/gait_parameters/gait_metrics.csv")
RTMPOSE_REL  = Path("validation/rtmpose/gait_parameters/gait_metrics.csv")


def load_gait_dataframe_multi(
    folder_list: list[Path],
    qualisys_rel: Path = QUALISYS_REL,
    fmc_system_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    if fmc_system_paths is None:
        fmc_system_paths = {
            "mediapipe_dlc": MP_DLC_REL,
            "rtmpose": RTMPOSE_REL,
        }

    rows = []
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

        qtmp = pd.read_csv(required_paths["qualisys"])
        qtmp["trial_name"] = folder.stem
        rows.append(qtmp)

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
    df["event_index"] = df["event_index"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def get_paired_stride_df(
    df: pd.DataFrame,
    metric: str,
    side: str = "left",
    reference_system: str = "qualisys",
    fmc_system: str = "mediapipe_dlc",
) -> pd.DataFrame:
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

def _axis_suffix(row: int, col: int, ncols: int) -> str:
    """Plotly axis naming: x, x2, x3... for subplots."""
    idx = (row - 1) * ncols + col
    return "" if idx == 1 else str(idx)


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

    # ticks (match your screenshot scale)
    max_ms = max_abs * 1000.0
    major_ticks_ms = np.arange(-1600, 1601, 400, dtype=float)
    major_ticks_ms = major_ticks_ms[np.abs(major_ticks_ms) <= max_ms + 1e-9]
    tickvals = (major_ticks_ms / 1000.0).tolist()
    ticktext = [("0" if ms == 0 else f"{ms:+.0f}") for ms in major_ticks_ms]

    stats_lines: list[str] = []
    ymax = 1

    # draw each system as a single filled step silhouette + outline
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
                showlegend=False,   # IMPORTANT: no global legend
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

        # stats: μ ± σ
        mu = float(np.mean(x) * 1000.0) if x.size else float("nan")
        sd = float(np.std(x, ddof=1) * 1000.0) if x.size >= 2 else float("nan")

        print(title,label, mu, sd)
        if np.isfinite(mu) and np.isfinite(sd):
            stats_lines.append(f"{label}: μ = {mu:+.0f} ± {sd:.0f} ms")
        else:
            stats_lines.append(f"{label}: μ = n/a")

    # headroom so annotation doesn't collide with peak bars
    fig.update_yaxes(range=[0, ymax * 1.25], row=row, col=col)

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
        tickfont=dict(size=11),
        row=row, col=col
    )

    fig.update_yaxes(
        title_text="<b>Count</b>" if show_ylabel else "",
        tickfont=dict(size=11),
        row=row, col=col
    )

    fig.update_xaxes(title_text="<b>Error (ms)</b>", row=row, col=col)

    # stats annotation (white box, top-left)
    # suf = _axis_suffix(row, col, ncols)
    # fig.add_annotation(
    #     x=0.03, y=0.92,
    #     xref=f"x{suf} domain",
    #     yref=f"y{suf} domain",
    #     text="<br>".join(stats_lines),
    #     showarrow=False,
    #     xanchor="left",
    #     yanchor="top",
    #     font=dict(size=11),
    #     bgcolor="rgba(255,255,255,0.85)",  # keep white backing
    #     borderwidth=0,                    # <-- no outline
    # )
    fig.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        y=-0.10,              # push below axes
        xanchor="center",
        yanchor="top",
        font=dict(size=11),
    )
)
    # light horizontal grid
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
    fmc_system_paths={
        "mediapipe_dlc": MP_DLC_REL,
        "rtmpose": RTMPOSE_REL,
    },
)

SIDE = "left"

paired_stance_mp = get_paired_stride_df(gait_param_df, metric="stance_duration", side=SIDE, fmc_system="mediapipe_dlc")
paired_stance_rt = get_paired_stride_df(gait_param_df, metric="stance_duration", side=SIDE, fmc_system="rtmpose")

paired_swing_mp  = get_paired_stride_df(gait_param_df, metric="swing_duration",  side=SIDE, fmc_system="mediapipe_dlc")
paired_swing_rt  = get_paired_stride_df(gait_param_df, metric="swing_duration",  side=SIDE, fmc_system="rtmpose")

colors = {
    "mediapipe_dlc": "#1f77b4",
    "rtmpose": "#ff7f0e",
}

fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Stance Duration", "Swing Duration"),
)

fig.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color=colors["mediapipe_dlc"], width=3),
        name="MediaPipe",
        showlegend=True,
    )
)

fig.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color=colors["rtmpose"], width=3),
        name="RTMPose",
        showlegend=True,
    )
)

add_histogram_overlay_panel(
    fig,
    dfs={"mediapipe_dlc": paired_stance_mp, "rtmpose": paired_stance_rt},
    colors=colors,
    row=1, col=1, ncols=2,
    title="Stance Duration",
    show_ylabel=True,
    max_frames=50,
)

add_histogram_overlay_panel(
    fig,
    dfs={"mediapipe_dlc": paired_swing_mp, "rtmpose": paired_swing_rt},
    colors=colors,
    row=1, col=2, ncols=2,
    title="Swing Duration",
    show_ylabel=False,
    max_frames=50,
)

fig.update_layout(
    width=FIG_WIDTH_PX,
    height=FIG_HEIGHT_PX,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=70),
)

# boxed axes
fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

fig.show()

import plotly.io as pio
pio.kaleido.scope.mathjax = None
path_to_save = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
fig.write_image(path_to_save / "nonprosthetic_gait_error_histogram.pdf")