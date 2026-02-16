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

TRACKERS = ["mediapipe", "rtmpose", "vitpose"]

TRACKER_STYLE = {
    "mediapipe": {
        "color": "#1f77b4",          # blue
        "fill": "rgba(31,119,180,0.18)",
    },
    "rtmpose": {
        "color": "#ff7f0e",          # orange
        "fill": "rgba(255,127,14,0.18)",
    },
    "vitpose": {
        "color": "#057702",          # green
        "fill": "rgba(255,127,14,0.18)",
    },
}

from dataclasses import dataclass

@dataclass
class BestLagResult:
    tracker: str
    metric: str
    discrete_best_lag: float
    discrete_best_mean: float
    quad_best_lag: float | None
    quad_best_mean: float | None
    bootstrap_mode_lag: float | None
    bootstrap_p05: float | None
    bootstrap_p95: float | None
    top_candidates: list[tuple[float, float]]  # (lag, mean)

def _group_mean_by_lag(df_raw: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    df_raw: per-recording rows with columns:
      ['recording','tracker','lag_frames', metric_col]
    Returns: lag_frames, mean, std, n_recordings
    """
    g = (
        df_raw.groupby("lag_frames", as_index=False)
        .agg(
            mean=(metric_col, "mean"),
            std=(metric_col, "std"),
            n=("recording", "nunique"),
        )
        .sort_values("lag_frames")
        .reset_index(drop=True)
    )
    return g

def _discrete_best(g: pd.DataFrame) -> tuple[float, float]:
    i = int(g["mean"].idxmin())
    return float(g.loc[i, "lag_frames"]), float(g.loc[i, "mean"])

def _quadratic_best(g: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    Fit mean = a*x^2 + b*x + c, return x* = -b/(2a) if convex.
    Clamped to [min_lag, max_lag].
    """
    x = g["lag_frames"].to_numpy(dtype=float)
    y = g["mean"].to_numpy(dtype=float)
    if len(x) < 3:
        return None, None

    a, b, c = np.polyfit(x, y, deg=2)
    if a <= 0:  # not convex -> no meaningful minimum
        return None, None

    x_star = -b / (2 * a)
    x_star = float(np.clip(x_star, x.min(), x.max()))
    y_star = float(a * x_star**2 + b * x_star + c)
    return x_star, y_star

def _bootstrap_best_lag(
    df_raw: pd.DataFrame,
    metric_col: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """
    Resample recordings with replacement, recompute mean RMSE per lag, take discrete best lag.
    Returns array of best lags (length n_boot).
    """
    rng = np.random.default_rng(seed)
    recordings = df_raw["recording"].unique()
    if len(recordings) < 2:
        return np.array([], dtype=float)

    bests = []
    for _ in range(n_boot):
        sampled = rng.choice(recordings, size=len(recordings), replace=True)
        boot = pd.concat([df_raw[df_raw["recording"] == r] for r in sampled], ignore_index=True)
        g = _group_mean_by_lag(boot, metric_col)
        best_lag, _ = _discrete_best(g)
        bests.append(best_lag)

    return np.array(bests, dtype=float)

def find_best_lag_candidates(
    df_all_raw: pd.DataFrame,
    tracker: str,
    metric_col: str = "overall_rmse",
    top_k: int = 3,
    n_boot: int = 2000,
) -> BestLagResult:
    """
    df_all_raw is the concatenation of all per-recording sweep rows (BEFORE summarize_across_recordings).
    """
    d = df_all_raw[df_all_raw["tracker"] == tracker].copy()
    if d.empty:
        raise ValueError(f"No data for tracker={tracker}")

    g = _group_mean_by_lag(d, metric_col)
    discrete_lag, discrete_mean = _discrete_best(g)

    quad_lag, quad_mean = _quadratic_best(g)

    # Top-k candidate discrete lags (useful if the curve is flat-ish)
    candidates = (
        g.sort_values("mean")[["lag_frames", "mean"]]
        .head(top_k)
        .to_records(index=False)
    )
    top_candidates = [(float(l), float(m)) for l, m in candidates]

    boot = _bootstrap_best_lag(d, metric_col, n_boot=n_boot, seed=0)
    if boot.size:
        # mode (most frequent lag) is a nice “candidate”
        vals, counts = np.unique(boot, return_counts=True)
        mode_lag = float(vals[np.argmax(counts)])
        p05 = float(np.quantile(boot, 0.05))
        p95 = float(np.quantile(boot, 0.95))
    else:
        mode_lag = p05 = p95 = None

    return BestLagResult(
        tracker=tracker,
        metric=metric_col,
        discrete_best_lag=discrete_lag,
        discrete_best_mean=discrete_mean,
        quad_best_lag=quad_lag,
        quad_best_mean=quad_mean,
        bootstrap_mode_lag=mode_lag,
        bootstrap_p05=p05,
        bootstrap_p95=p95,
        top_candidates=top_candidates,
    )


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
        Path(r"D:\validation\data\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1"),
        Path(r"D:\validation\data\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-52-16_GMT-4_jsm_treadmill_2"),
        Path(r"D:\validation\data\2026_01_26_KK\2026-01-16_14-15-39_GMT-5_kk_treadmill_1"),
        Path(r"D:\validation\data\2026_01_26_KK\2026-01-16_14-25-46_GMT-5_kk_treadmill_2"),
        Path(r"D:\validation\data\2025_09_03_OKK\freemocap\2025-09-03_14-56-30_GMT-4_okk_treadmill_1"),
        Path(r"D:\validation\data\2025_09_03_OKK\freemocap\2025-09-03_15-04-04_GMT-4_okk_treadmill_2"),
        Path(r"D:\validation\data\2025_11_04_ATC\2025-11-04_15-33-01_GMT-5_atc_treadmill_1"),
        Path(r"D:\validation\data\2025_11_04_ATC\2025-11-04_15-44-06_GMT-5_atc_treadmill_2"),
        Path(r"D:\validation\data\2026_01_30_JTM\2026-01-30_11-21-06_GMT-5_JTM_treadmill_1"),
        Path(r"D:\validation\data\2026_01_30_JTM\2026-01-30_11-32-56_GMT-5_JTM_treadmill_2")

    ]

    all_rows = []
    for rec in recording_dirs:
        for trk in TRACKERS:
            d = collect_tracker_sweep(rec, trk)
            if not d.empty:
                all_rows.append(d)

    if not all_rows:
        raise RuntimeError("No sweep RMSE data found.")
        
    df_raw = pd.concat(all_rows, ignore_index=True)

    for trk in TRACKERS:
        if trk not in df_raw["tracker"].unique():
            continue

        res = find_best_lag_candidates(
            df_raw,
            tracker=trk,
            metric_col="overall_rmse",   # or "y_rmse" / "z_rmse"
            top_k=4,
            n_boot=2000,
        )

        print("\n", "=" * 60)
        print(f"Tracker: {res.tracker} | Metric: {res.metric}")
        print(f"Discrete best lag: {res.discrete_best_lag:.3f} (mean RMSE={res.discrete_best_mean:.3f})")
        if res.quad_best_lag is not None:
            print(f"Quadratic best lag: {res.quad_best_lag:.3f} (pred RMSE={res.quad_best_mean:.3f})")
        if res.bootstrap_mode_lag is not None:
            print(f"Bootstrap mode lag: {res.bootstrap_mode_lag:.3f}")
            print(f"Bootstrap 90% interval: [{res.bootstrap_p05:.3f}, {res.bootstrap_p95:.3f}]")
        print("Top candidates (lag, mean RMSE):")
        for lag, mean in res.top_candidates:
            print(f"  {lag:.3f} : {mean:.3f}")



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
