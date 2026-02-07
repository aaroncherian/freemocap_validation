from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_lag_from_foldername(name: str) -> float | None:
    m = re.match(r"lag_(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def extract_metrics_from_position_rmse_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)

    def _get(dimension: str, coordinate: str, keypoint: str = "All") -> float:
        sub = df[(df["dimension"] == dimension) & (df["coordinate"] == coordinate) & (df["keypoint"] == keypoint)]
        if sub.empty:
            raise ValueError(
                f"Missing row: dimension={dimension}, coordinate={coordinate}, keypoint={keypoint} in {csv_path}"
            )
        return float(sub["RMSE"].iloc[0])

    overall = _get("Overall", "All", "All")
    y_rmse = _get("Per Dimension", "y_error", "All")
    z_rmse = _get("Per Dimension", "z_error", "All")
    return {"overall_rmse": overall, "y_rmse": y_rmse, "z_rmse": z_rmse}


def collect_tracker_sweep(recording_dir: Path, tracker: str) -> pd.DataFrame:
    sweeps_root = recording_dir / "validation" / "_sweeps" / tracker
    if not sweeps_root.exists():
        raise FileNotFoundError(f"Missing sweeps folder: {sweeps_root}")

    rows = []
    for lag_dir in sorted(sweeps_root.glob("lag_*")):
        lag = parse_lag_from_foldername(lag_dir.name)
        if lag is None:
            continue

        csv_path = (
            lag_dir / "validation" / tracker / "rmse" / "position" / "overall" / "position_rmse.csv"
        )
        if not csv_path.exists():
            continue

        metrics = extract_metrics_from_position_rmse_csv(csv_path)
        rows.append(
            {
                "tracker": tracker,
                "lag_frames": lag,
                "csv_path": str(csv_path),
                **metrics,
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"Found no position_rmse.csv files for tracker='{tracker}'. "
            f"Expected under: {sweeps_root}/lag_*/validation/{tracker}/rmse/position/overall/position_rmse.csv"
        )

    return pd.DataFrame(rows).sort_values(["tracker", "lag_frames"]).reset_index(drop=True)


def _paper_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14),
        margin=dict(l=70, r=30, t=70, b=60),
        legend=dict(
            title="",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            yanchor="top",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        title_standoff=10,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        title_standoff=10,
    )
    return fig


def main():
    # ---- EDIT THIS PATH ----
    recording_dir = Path(
        r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1"
    )
    # ------------------------

    trackers = ["mediapipe", "rtmpose"]

    dfs = []
    for t in trackers:
        try:
            dfs.append(collect_tracker_sweep(recording_dir, t))
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

    if not dfs:
        raise SystemExit("No sweep data found for any tracker.")

    df = pd.concat(dfs, ignore_index=True)

    # Save tidy summary
    out_csv = recording_dir / "validation" / "_sweeps" / "position_rmse_sweep_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote summary: {out_csv}")

    # Make one figure with 3 subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Overall position RMSE",
            "Y-dimension position RMSE",
            "Z-dimension position RMSE",
        ],
    )

    metrics = [
        ("overall_rmse", 1, "RMSE"),
        ("y_rmse", 2, "RMSE (Y)"),
        ("z_rmse", 3, "RMSE (Z)"),
    ]

    # consistent ordering
    tracker_order = [t for t in trackers if t in df["tracker"].unique()]

    for trk in tracker_order:
        d = df[df["tracker"] == trk].sort_values("lag_frames")
        for metric, row, ytitle in metrics:
            fig.add_trace(
                go.Scatter(
                    x=d["lag_frames"],
                    y=d[metric],
                    mode="lines+markers",
                    name=trk,
                    hovertemplate=(
                        "tracker=%{text}<br>"
                        "lag=%{x:.3f} frames<br>"
                        f"{metric}=%{{y:.3f}}<extra></extra>"
                    ),
                    text=[trk] * len(d),
                ),
                row=row,
                col=1,
            )
        # only one legend entry per tracker (keep first trace)
        # hide duplicates
        # traces are added in order: overall,y,z per tracker
        # keep overall visible in legend, hide the next two
        fig.data[-2].showlegend = False
        fig.data[-1].showlegend = False

    fig.update_xaxes(title_text="Lag (frames)", row=3, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)
    fig.update_yaxes(title_text="RMSE", row=3, col=1)

    fig.update_layout(
        title="Position RMSE vs temporal lag (sweep)",
        height=500,
        width = 500,
    )

    _paper_style(fig)
    fig.show()


if __name__ == "__main__":
    main()
