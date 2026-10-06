from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

recordings = {
    "neutral": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1"
    ),
    "neg_2_8": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1"
    ),
    "neg_5_6": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1"
    ),
    "pos_2_8": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1"
    ),
    "pos_5_6": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1"
    ),
}


# Main paper comparison
TRACKERS = [
    "qualisys",
    "rtmpose_dlc",
    # "rtmpose",
    # "mediapipe",
]


SYSTEM_LABELS = {
    "qualisys": "Qualisys",
    "rtmpose_dlc": "FMC-Hybrid",
    "rtmpose": "FMC-RTMPose",
    "mediapipe": "FMC-MediaPipe",
}


SYSTEM_STYLES = {
    "qualisys": {
        "color": "#4d4d4d",
        "symbol": "square",
    },
    "rtmpose_dlc": {
        "color": "#1f77b4",
        "symbol": "circle",
    },
    "rtmpose": {
        "color": "#d62728",
        "symbol": "diamond",
    },
    "mediapipe": {
        "color": "#e69f00",
        "symbol": "triangle-up",
    },
}


COND_ORDER = [
    "neg_5_6",
    "neg_2_8",
    "neutral",
    "pos_2_8",
    "pos_5_6",
]


COND_LABELS = {
    "neg_5_6": "−5.6°",
    "neg_2_8": "−2.8°",
    "neutral": "Neutral",
    "pos_2_8": "+2.8°",
    "pos_5_6": "+5.6°",
}


OUTPUT_DIR = Path(
    r"C:\Users\aaron\Documents\prosthetics_paper"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


OUT_PDF = OUTPUT_DIR / "toe_clearance_relative_to_neutral.pdf"
OUT_PNG = OUTPUT_DIR / "toe_clearance_relative_to_neutral.png"
OUT_SUMMARY_CSV = (
    OUTPUT_DIR / "toe_clearance_relative_to_neutral_summary.csv"
)
OUT_RMSE_CSV = (
    OUTPUT_DIR / "toe_clearance_multitracker_rmse_summary.csv"
)


# -------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------

def extract_minimum_toe_clearance_per_stride(
    path_to_csv: Path,
    marker: str = "right_foot_index",
) -> pd.DataFrame:
    """
    Extract minimum toe height for each stride.

    Minimum toe clearance is defined here as the minimum vertical toe
    position occurring between 70% and 95% of the gait cycle.
    """

    data = pd.read_csv(path_to_csv)

    toe = data.query(
        "marker == @marker"
    )[
        [
            "cycle",
            "percent_gait_cycle",
            "z",
        ]
    ].copy()

    rows = []

    for cycle, group in toe.groupby("cycle"):

        swing = group.query(
            "percent_gait_cycle > 70 "
            "and percent_gait_cycle < 95"
        ).copy()

        if swing.empty:
            continue

        min_idx = swing["z"].idxmin()
        min_row = swing.loc[min_idx]

        rows.append(
            {
                "cycle": cycle,
                "mtc_height": min_row["z"],
                "mtc_pct": min_row["percent_gait_cycle"],
            }
        )

    return pd.DataFrame(rows)


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build condition-level summary statistics and Qualisys comparison metrics.

    The condition summary contains the raw mean and SD of toe clearance.

    Relative-to-neutral values are calculated afterward by subtracting
    each tracker's own neutral-condition mean from its condition means.

    The SD remains the within-condition stride-to-stride SD because
    subtracting a constant neutral mean does not change the SD.
    """

    summary_rows = []
    rmse_rows = []

    for condition, recording in recordings.items():

        per_tracker = {}

        for tracker in TRACKERS:

            csv_path = (
                recording
                / "validation"
                / tracker
                / "trajectories"
                / "trajectories_per_stride.csv"
            )

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Missing: {csv_path}"
                )

            df = extract_minimum_toe_clearance_per_stride(
                csv_path
            )

            per_tracker[tracker] = df

            summary_rows.append(
                {
                    "condition": condition,
                    "tracker": tracker,
                    "mean_height": df[
                        "mtc_height"
                    ].mean(),
                    "std_height": df[
                        "mtc_height"
                    ].std(),
                    "mean_pct": df[
                        "mtc_pct"
                    ].mean(),
                    "std_pct": df[
                        "mtc_pct"
                    ].std(),
                    "n_strides": len(df),
                }
            )

        # -----------------------------------------------------------
        # Error relative to Qualisys for this condition
        # -----------------------------------------------------------

        reference = per_tracker["qualisys"]

        for tracker in TRACKERS:

            if tracker == "qualisys":
                continue

            paired = reference.merge(
                per_tracker[tracker],
                on="cycle",
                how="inner",
                suffixes=("_q", "_test"),
            )

            if paired.empty:
                print(
                    f"No matched cycles found for "
                    f"{condition} / {tracker}"
                )
                continue

            height_error = (
                paired["mtc_height_test"]
                - paired["mtc_height_q"]
            )

            pct_error = (
                paired["mtc_pct_test"]
                - paired["mtc_pct_q"]
            )

            rmse_rows.append(
                {
                    "tracker": tracker,
                    "condition": condition,
                    "n_strides": len(paired),

                    "qualisys_mean_height":
                        paired["mtc_height_q"].mean(),

                    "tracker_mean_height":
                        paired["mtc_height_test"].mean(),

                    "bias_height_mm":
                        height_error.mean(),

                    "mae_height_mm":
                        height_error.abs().mean(),

                    "rmse_height_mm":
                        np.sqrt(
                            np.mean(
                                height_error ** 2
                            )
                        ),

                    "bias_pct_gc":
                        pct_error.mean(),

                    "mae_pct_gc":
                        pct_error.abs().mean(),

                    "rmse_pct_gc":
                        np.sqrt(
                            np.mean(
                                pct_error ** 2
                            )
                        ),
                }
            )

    summary = pd.DataFrame(summary_rows)

    summary["condition"] = pd.Categorical(
        summary["condition"],
        categories=COND_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            [
                "tracker",
                "condition",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Express each tracker's condition means relative to its own
    # neutral-condition mean
    # ---------------------------------------------------------------

    neutral_means = (
        summary[
            summary["condition"] == "neutral"
        ][
            [
                "tracker",
                "mean_height",
            ]
        ]
        .rename(
            columns={
                "mean_height":
                    "neutral_mean_height"
            }
        )
    )

    summary = summary.merge(
        neutral_means,
        on="tracker",
        how="left",
    )

    summary["mean_height_relative"] = (
        summary["mean_height"]
        - summary["neutral_mean_height"]
    )

    # Neutral should be exactly zero except for floating-point noise.
    summary.loc[
        summary["condition"] == "neutral",
        "mean_height_relative",
    ] = 0.0


    rmse = pd.DataFrame(rmse_rows)

    if not rmse.empty:

        rmse["condition"] = pd.Categorical(
            rmse["condition"],
            categories=COND_ORDER,
            ordered=True,
        )

        rmse = (
            rmse
            .sort_values(
                [
                    "tracker",
                    "condition",
                ]
            )
            .reset_index(drop=True)
        )

    return summary, rmse


# -------------------------------------------------------------------
# FIGURE
# -------------------------------------------------------------------

def make_toe_clearance_relative_plot(
    summary: pd.DataFrame,
) -> go.Figure:

    # Keep same physical dimensions as the current figure
    FIG_W_IN = 1.8
    FIG_H_IN = 1.3
    DPI = 300

    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE_FONT = 15
    TICK_FONT = 14
    LEGEND_FONT = 14
    MARKER_SIZE = 7

    x_base = np.arange(
        len(COND_ORDER)
    )

    # Slight offset so Qualisys and FMC-Hybrid markers
    # do not sit directly on top of one another.
    # offsets = {
    #     "qualisys": -0.04,
    #     "rtmpose_dlc": 0.04,
    #     "rtmpose": 0.12,
    #     "mediapipe": -0.12,
    # }

    offsets = {
    "qualisys": 0.00,
    "rtmpose_dlc": 0.00,
    # "rtmpose": 0.12,
    # "mediapipe": -0.12,
}

    fig = go.Figure()

    for tracker in TRACKERS:

        tracker_df = (
            summary[
                summary["tracker"] == tracker
            ]
            .set_index("condition")
            .loc[COND_ORDER]
            .reset_index()
        )

        style = SYSTEM_STYLES[tracker]

        fig.add_trace(
            go.Scatter(
                x=(
                    x_base
                    + offsets[tracker]
                ),

                y=tracker_df[
                    "mean_height_relative"
                ],

                mode="markers+lines",

                name=SYSTEM_LABELS[
                    tracker
                ],

                marker=dict(
                    color=style["color"],
                    size=MARKER_SIZE,
                    symbol=style["symbol"],
                    line=dict(
                        width=0.5,
                        color="black",
                    ),
                ),

                opacity=0.8,

                line=dict(
                    width=1.5,
                    color=style["color"],
                ),

                # The plotted means are shifted relative to
                # neutral, but these error bars remain the
                # within-condition stride SD.
                error_y=dict(
                    type="data",
                    array=tracker_df[
                        "std_height"
                    ],
                    visible=True,
                    thickness=1.2,
                    width=4,
                    color=style["color"],
                ),

                customdata=np.column_stack(
                    [
                        tracker_df[
                            "condition"
                        ],

                        tracker_df[
                            "mean_height"
                        ],

                        tracker_df[
                            "neutral_mean_height"
                        ],

                        tracker_df[
                            "std_height"
                        ],

                        tracker_df[
                            "n_strides"
                        ],
                    ]
                ),

                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Condition: %{customdata[0]}<br>"
                    "Change from neutral: "
                    "%{y:.1f} mm<br>"
                    "Raw mean: "
                    "%{customdata[1]:.1f} mm<br>"
                    "Neutral mean: "
                    "%{customdata[2]:.1f} mm<br>"
                    "SD: "
                    "%{customdata[3]:.1f} mm<br>"
                    "Strides: "
                    "%{customdata[4]:.0f}"
                    "<extra></extra>"
                ),
            )
        )


    # Zero = no change from neutral
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
        line_color="gray",
    )


    fig.update_layout(
        template="simple_white",

        width=W,
        height=H,

        font=dict(
            family="Arial",
            size=BASE_FONT,
            color="black",
        ),

        margin=dict(
            l=65,
            r=15,
            t=10,
            b=45,
        ),

        xaxis=dict(
            title=(
                "<b>Prosthetic ankle "
                "dorsi/plantarflexion "
                "alignment (°)</b>"
            ),

            tickmode="array",
            tickvals=x_base,

            ticktext=[
                COND_LABELS[c]
                for c in COND_ORDER
            ],

            title_font=dict(
                size=BASE_FONT
            ),

            tickfont=dict(
                size=TICK_FONT
            ),

            showline=True,
            linecolor="black",
            mirror=True,

            ticks="outside",
            ticklen=4,
        ),

        yaxis=dict(
            title=(
                "<b>Δ Minimum toe "
                "clearance<br>"" from neutral (mm)</b>"
            ),

            title_font=dict(
                size=BASE_FONT
            ),

            tickfont=dict(
                size=TICK_FONT
            ),

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
            y=1.00,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=LEGEND_FONT),
        ),
            )

    return fig


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------

if __name__ == "__main__":

    summary, rmse_df = build_summary()

    fig = make_toe_clearance_relative_plot(
        summary
    )

    fig.show()

    pio.kaleido.scope.mathjax = None

    fig.write_image(
        OUT_PDF
    )

    fig.write_image(
        OUT_PNG,
        scale=3,
    )

    summary.to_csv(
        OUT_SUMMARY_CSV,
        index=False,
    )

    if not rmse_df.empty:
        rmse_df.to_csv(
            OUT_RMSE_CSV,
            index=False,
        )


    print(
        "\nToe Clearance "
        "Relative-to-Neutral Summary"
    )

    print(
        summary[
            [
                "tracker",
                "condition",
                "mean_height",
                "neutral_mean_height",
                "mean_height_relative",
                "std_height",
                "n_strides",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


    if not rmse_df.empty:

        print(
            "\nMinimum Toe Clearance "
            "Error Summary"
        )

        print(
            rmse_df
            .round(3)
            .to_string(index=False)
        )


        print(
            "\nMean RMSE across conditions"
        )

        tracker_summary = (
            rmse_df
            .groupby(
                "tracker",
                observed=True,
            )[
                "rmse_height_mm"
            ]
            .agg(
                [
                    "mean",
                    "std",
                ]
            )
            .reset_index()
        )

        tracker_summary[
            "system"
        ] = (
            tracker_summary[
                "tracker"
            ]
            .map(
                SYSTEM_LABELS
            )
        )

        print(
            tracker_summary[
                [
                    "system",
                    "mean",
                    "std",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )


    print("\nSaved:")
    print(
        f"  {OUT_PDF}"
    )
    print(
        f"  {OUT_PNG}"
    )
    print(
        f"  {OUT_SUMMARY_CSV}"
    )

    if not rmse_df.empty:
        print(
            f"  {OUT_RMSE_CSV}"
        )