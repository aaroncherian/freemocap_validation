from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from skellymodels.managers.human import Human


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

recordings = {
    "neg_5": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"
    ),
    "neg_25": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"
    ),
    "neutral": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"
    ),
    "pos_25": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"
    ),
    "pos_5": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"
    ),
}

CONDITION_ORDER = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]
TRACKERS = ["rtmpose_dlc", "qualisys"]

SYSTEM_LABELS = {
    "rtmpose_dlc": "FMC-Hybrid",
    "qualisys": "Qualisys",
}

SYSTEM_STYLES = {
    "rtmpose_dlc": {"color": "#1f77b4", "symbol": "circle"},
    "qualisys": {"color": "#d62728", "symbol": "square"},
}

INCH_OFFSETS = {
    "neg_5": -0.5,
    "neg_25": -0.25,
    "neutral": 0.0,
    "pos_25": 0.25,
    "pos_5": 0.5,
}

INCH_TO_MM = 25.4
MM_OFFSETS = {condition: offset * INCH_TO_MM for condition, offset in INCH_OFFSETS.items()}

TICK_LABELS = {
    condition: f"{MM_OFFSETS[condition]:.2f} mm" if condition != "neutral" else "Neutral"
    for condition in CONDITION_ORDER
}

OUTPUT_DIR = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTPUT_DIR / "leg_length_direct_overlay.pdf"
OUT_PNG = OUTPUT_DIR / "leg_length_direct_overlay.png"
OUT_CSV = OUTPUT_DIR / "leg_length_direct_overlay_results.csv"


# -------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------

@dataclass
class LegResults:
    data: np.ndarray
    median: float
    mad: float


def leg_length_from_human(human: Human) -> LegResults:
    knee = human.body.xyz.as_dict["right_knee"]
    ankle = human.body.xyz.as_dict["right_ankle"]
    leg_lengths_mm = np.linalg.norm(knee - ankle, axis=1)

    median = float(np.nanmedian(leg_lengths_mm))
    mad = float(np.nanmedian(np.abs(leg_lengths_mm - median)))
    return LegResults(data=leg_lengths_mm, median=median, mad=mad)


def load_all_results() -> dict[str, dict[str, LegResults]]:
    results = {tracker: {} for tracker in TRACKERS}

    for condition, recording_path in recordings.items():
        print(f"\nLoading condition: {condition}")

        for tracker in TRACKERS:
            tracker_path = recording_path / "validation" / tracker

            if not tracker_path.exists():
                raise FileNotFoundError(f"Missing tracker folder:\n{tracker_path}")

            print(f"  {tracker}")
            human = Human.from_data(tracker_path)
            results[tracker][condition] = leg_length_from_human(human)

    return results


def build_summary_dataframe(results: dict[str, dict[str, LegResults]]) -> pd.DataFrame:
    rows = []

    for tracker in TRACKERS:
        neutral_median = results[tracker]["neutral"].median

        for condition in CONDITION_ORDER:
            result = results[tracker][condition]
            measured_delta = result.median - neutral_median
            expected_delta = MM_OFFSETS[condition]

            rows.append({
                "tracker": tracker,
                "condition": condition,
                "median_leg_length_mm": result.median,
                "mad_leg_length_mm": result.mad,
                "delta_from_neutral_mm": measured_delta,
                "expected_delta_mm": expected_delta,
                "deviation_from_expected_mm": measured_delta - expected_delta,
                "n_frames": len(result.data),
            })

    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df.sort_values(["tracker", "condition"]).reset_index(drop=True)


# -------------------------------------------------------------------
# FIGURE
# -------------------------------------------------------------------

def make_direct_overlay_figure(df: pd.DataFrame) -> go.Figure:
    DPI = 300
    FIG_W_IN = 3.5
    FIG_H_IN = 2.6
    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE_FONT = 14
    TICK_FONT = 12
    LEGEND_FONT = 11
    MARKER_SIZE = 8

    x_base = np.arange(len(CONDITION_ORDER))
    fig = go.Figure()

    # Exact same x positions: no horizontal jitter.
    for tracker in TRACKERS:
        tracker_df = (
            df[df["tracker"] == tracker]
            .set_index("condition")
            .loc[CONDITION_ORDER]
            .reset_index()
        )
        style = SYSTEM_STYLES[tracker]

        fig.add_trace(
            go.Scatter(
                x=x_base,
                y=tracker_df["delta_from_neutral_mm"],
                mode="markers",
                name=SYSTEM_LABELS[tracker],
                marker=dict(
                    size=MARKER_SIZE,
                    symbol=style["symbol"],
                    color=style["color"],
                    line=dict(width=0.7, color="black"),
                ),
                error_y=dict(
                    type="data",
                    array=tracker_df["mad_leg_length_mm"],
                    visible=True,
                    thickness=1.1,
                    width=3,
                    color=style["color"],
                ),
                customdata=np.column_stack([
                    tracker_df["expected_delta_mm"],
                    tracker_df["deviation_from_expected_mm"],
                    tracker_df["mad_leg_length_mm"],
                ]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Measured Δ: %{y:.2f} mm<br>"
                    "Expected Δ: %{customdata[0]:.2f} mm<br>"
                    "Deviation: %{customdata[1]:+.2f} mm<br>"
                    "MAD: %{customdata[2]:.2f} mm"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_base,
            y=[MM_OFFSETS[c] for c in CONDITION_ORDER],
            mode="lines+markers",
            name="Expected Δ (mm)",
            line=dict(color="#4d4d4d", dash="dash", width=1.8),
            marker=dict(
                size=7,
                symbol="circle",
                color="#4d4d4d",
                line=dict(width=0.6, color="black"),
            ),
            hovertemplate="Expected Δ: %{y:.2f} mm<extra></extra>",
        )
    )

    fig.update_layout(
        template="simple_white",
        width=W,
        height=H,
        font=dict(family="Arial", size=BASE_FONT, color="black"),
        margin=dict(l=65, r=15, t=15, b=65),
        xaxis=dict(
            title="<b>Pylon length (mm)</b>",
            tickmode="array",
            tickvals=x_base,
            ticktext=[TICK_LABELS[c] for c in CONDITION_ORDER],
            tickfont=dict(size=TICK_FONT),
            title_font=dict(size=BASE_FONT),
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="outside",
            ticklen=4,
        ),
        yaxis=dict(
            title="<b>Δ Median leg length (mm)</b>",
            tickfont=dict(size=TICK_FONT),
            title_font=dict(size=BASE_FONT),
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="outside",
            ticklen=4,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.02,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=LEGEND_FONT),
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )

    return fig


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------

if __name__ == "__main__":
    results = load_all_results()
    df_leg = build_summary_dataframe(results)
    df_leg.to_csv(OUT_CSV, index=False)

    fig = make_direct_overlay_figure(df_leg)
    fig.show()

    pio.kaleido.scope.mathjax = None
    fig.write_image(OUT_PDF)
    fig.write_image(OUT_PNG, scale=3)

    print("\nSaved:")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PNG}")
    print(f"  {OUT_CSV}")
