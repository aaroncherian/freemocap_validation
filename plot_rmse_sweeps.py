from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Configuration
# -----------------------------

TRACKERS = ["mediapipe", "rtmpose"]

TRACKER_STYLE = {
    "mediapipe": {
        "color": "#1f77b4",          # blue
        "fill": "rgba(31,119,180,0.18)",
    },
    "rtmpose": {
        "color": "#ff7f0e",          # orange
        "fill": "rgba(255,127,14,0.18)",
    },
}


# -----------------------------
# Utilities
# -----------------------------

def parse_lag_from_foldername(name: str) -> float | None:
    m = re.match(r"lag_(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def extract_metrics_from_position_rmse_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)

    def _get(dimension: str, coordinate: str) -> float:
        sub = df[
            (df["dimension"] == dimension)
            & (df["coordinate"] == coordinate)
            & (df["keypoint"] == "All")
        ]
        if sub.empty:
            raise ValueError(f"Missing {dimension=} {coordinate=} in {csv_path}")
        return float(sub["RMSE"].iloc[0])

    return {
        "overall_rmse": _get("Overall", "All"),
        "y_rmse": _get("Per Dimension", "y_error"),
        "z_rmse": _get("Per Dimension", "z_error"),
    }


def collect_tracker_sweep(recording_dir: Path, tracker: str) -> pd.DataFrame:
    sweeps_root = recording_dir / "validation" / "_sweeps" / tracker
    if not sweeps_root.exists():
        return pd.DataFrame()

    rows = []
    for lag_dir in sorted(sweeps_root.glob("lag_*")):
        lag = parse_lag_from_foldername(lag_dir.name)
        if lag is None:
            continue

        csv_path = (
            lag_dir
            / "validation"
            / tracker
            / "rmse"
            / "position"
            / "overall"
            / "position_rmse.csv"
        )
        if not csv_path.exists():
            continue

        metrics = extract_metrics_from_position_rmse_csv(csv_path)
        rows.append(
            {
                "recording": recording_dir.name,
                "tracker": tracker,
                "lag_frames": lag,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_across_recordings(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["tracker", "lag_frames"], as_index=False)
        .agg(
            overall_mean=("overall_rmse", "mean"),
            overall_std=("overall_rmse", "std"),
            y_mean=("y_rmse", "mean"),
            y_std=("y_rmse", "std"),
            z_mean=("z_rmse", "mean"),
            z_std=("z_rmse", "std"),
            n=("recording", "nunique"),
        )
        .sort_values(["tracker", "lag_frames"])
        .reset_index(drop=True)
    )


# -----------------------------
# Plot helpers
# -----------------------------

def add_mean_with_shaded_sd(
    fig: go.Figure,
    d: pd.DataFrame,
    *,
    tracker: str,
    mean_col: str,
    std_col: str,
    row: int,
):
    style = TRACKER_STYLE[tracker]
    color = style["color"]
    fill = style["fill"]

    x = d["lag_frames"].to_numpy()
    mean = d[mean_col].to_numpy()
    std = d[std_col].to_numpy()

    upper = mean + std
    lower = mean - std

    # Upper bound (invisible)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )

    # Lower bound with fill
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            fill="tonexty",
            fillcolor=fill,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            name=tracker,
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color),
            hovertemplate=(
                f"tracker={tracker}<br>"
                "lag=%{x:.3f} frames<br>"
                "mean=%{y:.3f}<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )


def apply_paper_style(fig: go.Figure):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14),
        margin=dict(l=70, r=30, t=80, b=60),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            yanchor="top",
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        zeroline=False,
        showline=True,
        mirror=True,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=True,
        zeroline=False,
        showline=True,
        mirror=True,
        ticks="outside",
    )


# -----------------------------
# Main
# -----------------------------

def main():
    recording_dirs = [
        Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1"),
        Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-52-16_GMT-4_jsm_treadmill_2"),
        Path(r"D:\2026_01_26_KK\2026-01-16_14-15-39_GMT-5_kk_treadmill_1"),
        Path(r"D:\2026_01_26_KK\2026-01-16_14-25-46_GMT-5_kk_treadmill_2"),
        Path(r"D:\2025_09_03_OKK\freemocap\2025-09-03_14-56-30_GMT-4_okk_treadmill_1"),
        Path(r"D:\2025_09_03_OKK\freemocap\2025-09-03_15-04-04_GMT-4_okk_treadmill_2"),
        Path(r"D:\2025-11-04_ATC\2025-11-04_15-33-01_GMT-5_atc_treadmill_1"),
        Path(r"D:\2025-11-04_ATC\2025-11-04_15-44-06_GMT-5_atc_treadmill_2"),
        Path(r"D:\2026-01-30-JTM\2026-01-30_11-21-06_GMT-5_JTM_treadmill_1"),
        Path(r"D:\2026-01-30-JTM\2026-01-30_11-32-56_GMT-5_JTM_treadmill_2")

    ]

    all_rows = []
    for rec in recording_dirs:
        for trk in TRACKERS:
            d = collect_tracker_sweep(rec, trk)
            if not d.empty:
                all_rows.append(d)

    if not all_rows:
        raise RuntimeError("No sweep RMSE data found.")

    df = summarize_across_recordings(pd.concat(all_rows, ignore_index=True))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Overall position RMSE (mean ± SD)",
            "Y-dimension position RMSE (mean ± SD)",
            "Z-dimension position RMSE (mean ± SD)",
        ],
    )

    for trk in TRACKERS:
        if trk not in df["tracker"].unique():
            continue

        d = df[df["tracker"] == trk]

        add_mean_with_shaded_sd(fig, d, tracker=trk, mean_col="overall_mean", std_col="overall_std", row=1)
        add_mean_with_shaded_sd(fig, d, tracker=trk, mean_col="y_mean", std_col="y_std", row=2)
        add_mean_with_shaded_sd(fig, d, tracker=trk, mean_col="z_mean", std_col="z_std", row=3)

        # hide duplicate legend entries
        fig.data[-2].showlegend = False
        fig.data[-1].showlegend = False

    fig.update_xaxes(title_text="Lag (frames)", row=3, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)
    fig.update_yaxes(title_text="RMSE", row=3, col=1)

    fig.update_layout(
        title="Position RMSE vs temporal lag (mean ± SD across recordings)",
        height=500,
        width=500,
    )

    apply_paper_style(fig)
    fig.show()


if __name__ == "__main__":
    main()
