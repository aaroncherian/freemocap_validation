from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

recordings = {
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1"),
    "neg_2_8": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1"),
    "neg_5_6": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1"),
    "pos_2_8": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1"),
    "pos_5_6": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1"),
}

# Draw Qualisys first so the reference sits behind FMC-Hybrid where they overlap.
SYSTEMS = [
            "mediapipe", 
           "rtmpose",
           "qualisys", 
           "rtmpose_dlc", 
           ]

SYSTEM_LABELS = {
    "rtmpose_dlc": "FMC-Hybrid",
    "qualisys": "Qualisys",
    "rtmpose": "FMC-RTMPose",
    "mediapipe": "FMC-MediaPipe",
}


SYSTEM_STYLES = {
    "rtmpose_dlc": {"color": "#1f77b4", "symbol": "circle"},       # blue
    "rtmpose":     {"color": "#d62728", "symbol": "diamond"},      # red
    "mediapipe":   {"color": "#e69f00", "symbol": "triangle-up"},  # orange
    "qualisys":    {"color": "#4d4d4d", "symbol": "square"},       # charcoal
}


COND_ORDER = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]
COND_LABELS = {
    "neg_5_6": "−5.6°",
    "neg_2_8": "−2.8°",
    "neutral": "Neutral",
    "pos_2_8": "+2.8°",
    "pos_5_6": "+5.6°",
}

OUTPUT_DIR = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTPUT_DIR / "toe_clearance_direct_overlay.pdf"
OUT_PNG = OUTPUT_DIR / "toe_clearance_direct_overlay.png"
OUT_RMSE_CSV = OUTPUT_DIR / "toe_clearance_direct_overlay_rmse_summary.csv"


# -------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------

def extract_minimum_toe_clearance_per_stride(
    path_to_csv: Path,
    marker: str = "right_foot_index",
) -> pd.DataFrame:
    data = pd.read_csv(path_to_csv)
    toe = data.query("marker == @marker")[["cycle", "percent_gait_cycle", "z"]].copy()

    rows = []
    for cycle, group in toe.groupby("cycle"):
        swing = group.query("percent_gait_cycle > 70 and percent_gait_cycle < 95").copy()

        if swing.empty:
            continue

        min_idx = swing["z"].idxmin()
        min_row = swing.loc[min_idx]

        rows.append({
            "cycle": cycle,
            "mtc_height": min_row["z"],
            "mtc_pct": min_row["percent_gait_cycle"],
        })

    return pd.DataFrame(rows)


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    rmse_rows = []

    for condition, recording in recordings.items():
        per_system = {}

        for system in SYSTEMS:
            csv_path = recording / "validation" / system / "trajectories" / "trajectories_per_stride.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing: {csv_path}")

            df = extract_minimum_toe_clearance_per_stride(csv_path)
            per_system[system] = df

            summary_rows.append({
                "condition": condition,
                "tracker": system,
                "mean_height": df["mtc_height"].mean(),
                "std_height": df["mtc_height"].std(),
                "mean_pct": df["mtc_pct"].mean(),
                "std_pct": df["mtc_pct"].std(),
                "n_strides": len(df),
            })

        paired = per_system["qualisys"].merge(
            per_system["rtmpose_dlc"],
            on="cycle",
            how="inner",
            suffixes=("_q", "_fmc"),
        )

        if paired.empty:
            print(f"No matched cycles found for {condition}")
            continue

        height_error = paired["mtc_height_fmc"] - paired["mtc_height_q"]
        pct_error = paired["mtc_pct_fmc"] - paired["mtc_pct_q"]

        rmse_rows.append({
            "condition": condition,
            "n_strides": len(paired),
            "qualisys_mean_height": paired["mtc_height_q"].mean(),
            "fmc_mean_height": paired["mtc_height_fmc"].mean(),
            "bias_height_mm": height_error.mean(),
            "mae_height_mm": height_error.abs().mean(),
            "rmse_height_mm": np.sqrt(np.mean(height_error ** 2)),
            "bias_pct_gc": pct_error.mean(),
            "mae_pct_gc": pct_error.abs().mean(),
            "rmse_pct_gc": np.sqrt(np.mean(pct_error ** 2)),
        })

    summary = pd.DataFrame(summary_rows)
    summary["condition"] = pd.Categorical(summary["condition"], categories=COND_ORDER, ordered=True)
    summary = summary.sort_values(["condition", "tracker"]).reset_index(drop=True)

    rmse = pd.DataFrame(rmse_rows)
    rmse["condition"] = pd.Categorical(rmse["condition"], categories=COND_ORDER, ordered=True)
    rmse = rmse.sort_values("condition").reset_index(drop=True)

    return summary, rmse


# -------------------------------------------------------------------
# FIGURE
# -------------------------------------------------------------------

def make_toe_clearance_direct_overlay(summary: pd.DataFrame) -> go.Figure:
    # Same dimensions as the current toe-clearance figure.
    FIG_W_IN = 1.8
    FIG_H_IN = 1.3
    DPI = 300
    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE_FONT = 15
    TICK_FONT = 14
    LEGEND_FONT = 12
    MARKER_SIZE = 7

    x_base = np.arange(len(COND_ORDER))
    fig = go.Figure()

    # Exact same x positions: no horizontal jitter.
    for system in SYSTEMS:
        system_df = (
            summary[summary["tracker"] == system]
            .set_index("condition")
            .loc[COND_ORDER]
            .reset_index()
        )
        style = SYSTEM_STYLES[system]

        fig.add_trace(
            go.Scatter(
                x=x_base,
                y=system_df["mean_height"],
                mode="lines+markers",
                name=SYSTEM_LABELS[system],
                line=dict(color=style["color"], width=1.5),
                marker=dict(
                    size=MARKER_SIZE,
                    symbol=style["symbol"],
                    color=style["color"],
                    line=dict(width=0.5, color="black"),
                ),
                error_y=dict(
                    type="data",
                    array=system_df["std_height"],
                    visible=True,
                    thickness=1.2,
                    width=4,
                    color=style["color"],
                ),
                customdata=np.column_stack([
                    system_df["condition"],
                    system_df["std_height"],
                    system_df["n_strides"],
                ]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Condition: %{customdata[0]}<br>"
                    "Mean: %{y:.1f} mm<br>"
                    "SD: %{customdata[1]:.1f} mm<br>"
                    "Strides: %{customdata[2]:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="simple_white",
        width=W,
        height=H,
        font=dict(family="Arial", size=BASE_FONT, color="black"),
        margin=dict(l=55, r=15, t=10, b=45),
        xaxis=dict(
            title="<b>Prosthetic ankle dorsi/plantarflexion alignment (°)</b>",
            tickmode="array",
            tickvals=x_base,
            ticktext=[COND_LABELS[c] for c in COND_ORDER],
            title_font=dict(size=BASE_FONT),
            tickfont=dict(size=TICK_FONT),
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="outside",
            ticklen=4,
        ),
        yaxis=dict(
            title="<b>Minimum toe clearance (mm)</b>",
            title_font=dict(size=BASE_FONT),
            tickfont=dict(size=TICK_FONT),
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="outside",
            ticklen=4,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=LEGEND_FONT),
        ),
    )

    return fig


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------

if __name__ == "__main__":
    summary, rmse_df = build_summary()

    fig = make_toe_clearance_direct_overlay(summary)
    fig.show()

    pio.kaleido.scope.mathjax = None
    fig.write_image(OUT_PDF)
    fig.write_image(OUT_PNG, scale=3)
    rmse_df.to_csv(OUT_RMSE_CSV, index=False)

    print("\nMinimum Toe Clearance Error Summary")
    print(rmse_df.round(3))

    avg = rmse_df["rmse_height_mm"].mean()
    std = rmse_df["rmse_height_mm"].std()
    print(f"\nAverage RMSE across conditions: {avg:.3f} mm (std: {std:.3f} mm)")

    print("\nSaved:")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PNG}")
    print(f"  {OUT_RMSE_CSV}")
