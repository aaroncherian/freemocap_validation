from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skellymodels.managers.human import Human


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

recordings = {
    "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
    "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
    "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
    "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
}

condition_order = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]

# Prosthetic alignment offsets, originally in inches
inch_offsets = {
    "neg_5": -0.5,
    "neg_25": -0.25,
    "neutral": 0.0,
    "pos_25": 0.25,
    "pos_5": 0.5,
}

INCH_TO_MM = 25.4
mm_offsets = {cond: val * INCH_TO_MM for cond, val in inch_offsets.items()}

tick_label_map = {
    cond: f"{mm_offsets[cond]:.2f} mm" if cond != "neutral" else "neutral"
    for cond in condition_order
}

# Plot styling
COLOR_FMC = "#1f77b4"   # FreeMoCap blue
COLOR_Q   = "#d62728"   # Qualisys red
COLOR_EXP = "#504F4F"   # expected line
MARKER_SIZE = 10


# -------------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------------

@dataclass
class LegResults:
    data: np.ndarray   # per-frame leg lengths [mm]
    median: float      # median leg length [mm]
    mad: float         # MAD [mm]


def median_and_mad(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(med), float(mad)


def leg_length_from_human(human: Human) -> LegResults:
    """
    Right-leg length as distance between right_knee and right_ankle in mm.
    """
    leg_lengths_mm = np.linalg.norm(
        human.body.xyz.as_dict["right_knee"] - human.body.xyz.as_dict["right_ankle"],
        axis=1,
    )
    med, mad = median_and_mad(leg_lengths_mm)
    return LegResults(data=leg_lengths_mm, median=med, mad=mad)


def load_qualisys_gait_events(csv_path: Path, foot: str = "right") -> tuple[np.ndarray, np.ndarray]:
    """
    Returns heel_strike_frames, toe_off_frames for requested foot.

    Expected CSV columns: ['foot', 'event', 'frame']
    Events: 'heel_strike' and 'toe_off'
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Gait events CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"foot", "event", "frame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df_foot = df[df["foot"].astype(str).str.lower() == foot.lower()].copy()
    if df_foot.empty:
        raise ValueError(f"No rows found for foot='{foot}' in {csv_path}")

    # Normalize event strings just in case
    df_foot["event"] = df_foot["event"].astype(str).str.strip().str.lower()

    hs = (
        df_foot[df_foot["event"] == "heel_strike"]["frame"]
        .astype(int)
        .sort_values()
        .to_numpy()
    )
    to = (
        df_foot[df_foot["event"] == "toe_off"]["frame"]
        .astype(int)
        .sort_values()
        .to_numpy()
    )

    return hs, to


def build_stance_swing_masks(
    n_frames: int,
    heel_strikes: np.ndarray,
    toe_offs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stance: heel_strike -> next toe_off
    Swing : toe_off -> next heel_strike

    Frames not covered by valid HS/TO pairing remain False in both.
    """
    stance = np.zeros(n_frames, dtype=bool)
    swing = np.zeros(n_frames, dtype=bool)

    hs = np.asarray(heel_strikes, dtype=int)
    to = np.asarray(toe_offs, dtype=int)

    hs = hs[(hs >= 0) & (hs < n_frames)]
    to = to[(to >= 0) & (to < n_frames)]

    if len(hs) == 0 or len(to) == 0:
        return stance, swing

    for hs_i in hs:
        to_after = to[to > hs_i]
        if len(to_after) == 0:
            continue
        to_i = to_after[0]

        # stance segment [hs_i, to_i)
        if to_i > hs_i:
            stance[hs_i:to_i] = True

        # swing segment [to_i, next_hs)
        hs_after = hs[hs > to_i]
        if len(hs_after) == 0:
            continue
        hs_next = hs_after[0]

        if hs_next > to_i:
            swing[to_i:hs_next] = True

    return stance, swing


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    freemocap_results: dict[str, LegResults] = {}
    qualisys_results: dict[str, LegResults] = {}

    # Output tables
    rows_phase_samples: list[dict] = []
    rows_phase_summary: list[dict] = []
    rows_phase_comp: list[dict] = []
    ba_store: dict[tuple[str, str], dict] = {}

    for condition in condition_order:
        recording_path = recordings[condition]

        path_to_freemocap_folder = recording_path / "validation" / "mediapipe_dlc"
        path_to_qualisys_folder = recording_path / "validation" / "qualisys"

        f_human: Human = Human.from_data(path_to_freemocap_folder)
        q_human: Human = Human.from_data(path_to_qualisys_folder)

        f_res = leg_length_from_human(f_human)
        q_res = leg_length_from_human(q_human)

        freemocap_results[condition] = f_res
        qualisys_results[condition] = q_res

        gait_csv = recording_path / "validation" / "qualisys" / "gait_events" / "qualisys_gait_events.csv"
        hs_frames, to_frames = load_qualisys_gait_events(gait_csv, foot="right")

        # Align frame length conservatively
        n_frames = min(len(f_res.data), len(q_res.data))
        stance_mask, swing_mask = build_stance_swing_masks(n_frames, hs_frames, to_frames)

        f_data = f_res.data[:n_frames]
        q_data = q_res.data[:n_frames]

        # Per-frame samples with phase labels
        def add_samples(system_name: str, data: np.ndarray):
            for phase_name, mask in [("stance", stance_mask), ("swing", swing_mask)]:
                idx = np.where(mask)[0]
                for i in idx:
                    rows_phase_samples.append(
                        dict(
                            condition=condition,
                            system=system_name,
                            phase=phase_name,
                            frame=int(i),
                            leg_length_mm=float(data[i]),
                        )
                    )

        add_samples("freemocap", f_data)
        add_samples("qualisys", q_data)

        # Phase summaries (median + MAD)
        for system_name, data in [("freemocap", f_data), ("qualisys", q_data)]:
            for phase_name, mask in [("stance", stance_mask), ("swing", swing_mask)]:
                med, mad = median_and_mad(data[mask])
                rows_phase_summary.append(
                    dict(
                        condition=condition,
                        system=system_name,
                        phase=phase_name,
                        median_leg_length_mm=med,
                        mad_leg_length_mm=mad,
                        n_frames=int(mask.sum()),
                    )
                )
        for phase_name, mask in [("stance", stance_mask), ("swing", swing_mask)]:
            ba = bland_altman_stats(f_data, q_data, mask)
            ba_store[(phase_name, condition)] = ba

    df_leglength_phase_samples = pd.DataFrame(rows_phase_samples)
    df_leglength_phase_summary = pd.DataFrame(rows_phase_summary)

    # ---------------------------------------------------------------
    # Δ FROM NEUTRAL PER PHASE + EXPECTED OFFSETS
    # ---------------------------------------------------------------
    baseline = (
        df_leglength_phase_summary[df_leglength_phase_summary["condition"] == "neutral"]
        .set_index(["system", "phase"])["median_leg_length_mm"]
        .to_dict()
    )

    for condition in condition_order:
        for phase in ["stance", "swing"]:
            f_row = df_leglength_phase_summary.query(
                "condition == @condition and system == 'freemocap' and phase == @phase"
            ).iloc[0]
            q_row = df_leglength_phase_summary.query(
                "condition == @condition and system == 'qualisys' and phase == @phase"
            ).iloc[0]

            f_base = baseline.get(("freemocap", phase), np.nan)
            q_base = baseline.get(("qualisys", phase), np.nan)

            rows_phase_comp.append(
                dict(
                    condition=condition,
                    phase=phase,
                    expected_delta_mm=mm_offsets[condition],

                    fmc_median_mm=float(f_row["median_leg_length_mm"]),
                    fmc_delta_mm=float(f_row["median_leg_length_mm"] - f_base),
                    fmc_variability_mm=float(f_row["mad_leg_length_mm"]),
                    fmc_n_frames=int(f_row["n_frames"]),
                    fmc_residual_mm = float(f_row["median_leg_length_mm"] - f_base) - mm_offsets[condition], #fmc_delta_mm - expected_delta_mm

                    q_median_mm=float(q_row["median_leg_length_mm"]),
                    q_delta_mm=float(q_row["median_leg_length_mm"] - q_base),
                    q_variability_mm=float(q_row["mad_leg_length_mm"]),
                    q_n_frames=int(q_row["n_frames"]),
                    q_residual_mm = float(q_row["median_leg_length_mm"] - q_base) - mm_offsets[condition] #q_delta_mm - expected_delta_mm
                )
            )

    df_leglength_phase_comparison = pd.DataFrame(rows_phase_comp)

    # ---------------------------------------------------------------
    # PLOTLY FIGURE: stance vs swing panels
    # ---------------------------------------------------------------
    # x positions
    x_base = np.arange(len(condition_order))
    offset = 0.12
    x_fmc = x_base - offset
    x_q = x_base + offset
    x_exp = x_base

    tick_text = [tick_label_map[c] for c in condition_order]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Stance", "Swing"),
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for col, phase in enumerate(["stance", "swing"], start=1):
        dfp = (
            df_leglength_phase_comparison[df_leglength_phase_comparison["phase"] == phase]
            .set_index("condition")
            .loc[condition_order]
            .reset_index()
        )

        # FreeMoCap
        fig.add_trace(
            go.Scatter(
                x=x_fmc,
                y=dfp["fmc_delta_mm"],
                mode="markers",
                name="FreeMoCap Δ" if col == 1 else "FreeMoCap Δ ",
                marker=dict(color=COLOR_FMC, size=MARKER_SIZE, line=dict(width=0.5, color="black")),
                error_y=dict(type="data", array=dfp["fmc_variability_mm"], visible=True, thickness=1.2, width=4),
                customdata=dfp["condition"],
                hovertemplate=(
                    "Condition: %{customdata}<br>"
                    "Phase: " + phase + "<br>"
                    "FreeMoCap Δ: %{y:.2f} mm<br>"
                    "MAD: %{error_y.array:.2f} mm<extra></extra>"
                ),
                showlegend=(col == 1),
            ),
            row=1, col=col
        )

        # Qualisys
        fig.add_trace(
            go.Scatter(
                x=x_q,
                y=dfp["q_delta_mm"],
                mode="markers",
                name="Qualisys Δ" if col == 1 else "Qualisys Δ ",
                marker=dict(color=COLOR_Q, size=MARKER_SIZE, symbol="square", line=dict(width=0.5, color="black")),
                error_y=dict(type="data", array=dfp["q_variability_mm"], visible=True, thickness=1.2, width=4),
                customdata=dfp["condition"],
                hovertemplate=(
                    "Condition: %{customdata}<br>"
                    "Phase: " + phase + "<br>"
                    "Qualisys Δ: %{y:.2f} mm<br>"
                    "MAD: %{error_y.array:.2f} mm<extra></extra>"
                ),
                showlegend=(col == 1),
            ),
            row=1, col=col
        )

        # Expected line
        fig.add_trace(
            go.Scatter(
                x=x_exp,
                y=dfp["expected_delta_mm"],
                mode="lines+markers",
                name="Expected Δ (mm)",
                line=dict(dash="dash", width=2, color=COLOR_EXP),
                marker=dict(size=MARKER_SIZE, line=dict(width=1.5, color=COLOR_EXP)),
                hovertemplate="Expected Δ: %{y:.2f} mm<extra></extra>",
                showlegend=(col == 1),
            ),
            row=1, col=col
        )

        # x axis ticks per subplot
        fig.update_xaxes(
            title_text="Prosthetic alignment condition",
            tickmode="array",
            tickvals=x_base,
            ticktext=tick_text,
            row=1, col=col
        )

    ZEROLINE_COLOR = "rgba(0,0,0,0.35)"

    for col in [1, 2]:
        fig.update_yaxes(
            zeroline=True,
            zerolinecolor=ZEROLINE_COLOR,
            zerolinewidth=1,
            row=1,
            col=col
        )

    # Put the y-axis title only on the left subplot (cleaner)
    fig.update_yaxes(
        title_text="Δ Median leg length from neutral [mm]",
        row=1,
        col=1
    )

    fig.update_layout(
        template="simple_white",
        width=1100,
        height=650,
        font=dict(size=16),
        title=dict(
            text="Change in median leg length relative to neutral (stance vs swing, Qualisys events)",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=14)),
        margin=dict(l=80, r=40, t=80, b=110),
    )

    fig.show()


    def downsample(mean: np.ndarray, diff: np.ndarray, max_points: int = 800):
        n = mean.size
        if n <= max_points:
            return mean, diff
        step = int(np.ceil(n / max_points))
        return mean[::step], diff[::step]

    fig_ba = make_subplots(
        rows=2, cols=5,
        subplot_titles=[c for c in condition_order] * 2,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for col, condition in enumerate(condition_order, start=1):
        for row, phase in enumerate(["stance", "swing"], start=1):
            ba = ba_store.get((phase, condition), None)
            if ba is None:
                continue

            mean = ba["mean"]
            diff = ba["diff"]
            bias = ba["bias"]
            loa_low = ba["loa_low"]
            loa_high = ba["loa_high"]
            n = ba["n"]

            mean_ds, diff_ds = downsample(mean, diff, max_points=600)

            # Scatter
            fig_ba.add_trace(
                go.Scatter(
                    x=mean_ds,
                    y=diff_ds,
                    mode="markers",
                    marker=dict(size=4, opacity=0.35),
                    showlegend=False,
                    hovertemplate=(
                        f"Phase: {phase}<br>"
                        f"Condition: {condition}<br>"
                        "Mean: %{x:.2f} mm<br>"
                        "Diff (FMC−Q): %{y:.2f} mm<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

            # Bias + LoA lines (only if valid)
            if np.isfinite(bias):
                x0 = float(np.nanmin(mean)) if mean.size else 0.0
                x1 = float(np.nanmax(mean)) if mean.size else 1.0

                for y_val, dash, text in [
                    (bias, "solid", f"bias={bias:.2f}"),
                    (loa_low, "dash", f"LoA-={loa_low:.2f}"),
                    (loa_high, "dash", f"LoA+={loa_high:.2f}"),
                ]:
                    fig_ba.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[y_val, y_val],
                            mode="lines",
                            line=dict(dash=dash, width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row, col=col,
                    )

                axis_index = (row - 1) * 5 + col

                xref = "x domain" if axis_index == 1 else f"x{axis_index} domain"
                yref = "y domain" if axis_index == 1 else f"y{axis_index} domain"

                fig_ba.add_annotation(
                    x=0.02, y=0.98,
                    xref=xref, yref=yref,
                    text=f"N={n}<br>bias={bias:.2f} mm",
                    showarrow=False,
                    align="left",
                    font=dict(size=12),
)
    # Axis labels
    for col in range(1, 6):
        fig_ba.update_xaxes(title_text="Mean leg length (FMC+Q)/2 [mm]", row=2, col=col)

    fig_ba.update_yaxes(title_text="Difference (FMC − Qualisys) [mm]", row=1, col=1)

    # Row labels (phase)
    fig_ba.add_annotation(x=-0.02, y=0.79, xref="paper", yref="paper", text="Stance", showarrow=False, font=dict(size=16))
    fig_ba.add_annotation(x=-0.02, y=0.24, xref="paper", yref="paper", text="Swing", showarrow=False, font=dict(size=16))

    fig_ba.update_layout(
        template="simple_white",
        width=1400,
        height=650,
        title=dict(text="Bland–Altman: FreeMoCap vs Qualisys leg length (by condition × phase)", x=0.5),
        margin=dict(l=90, r=30, t=80, b=60),
    )

    fig_ba.show()

    # ---------------------------------------------------------------
    # PRINT / RETURN TABLES
    # ---------------------------------------------------------------
    print("\nPhase summary head:")
    print(df_leglength_phase_summary.head())

    print("\nPhase comparison head:")
    print(df_leglength_phase_comparison.head())

    return df_leglength_phase_samples, df_leglength_phase_summary, df_leglength_phase_comparison

def bland_altman_stats(f: np.ndarray, q: np.ndarray, mask: np.ndarray) -> dict:
    f = np.asarray(f)[mask]
    q = np.asarray(q)[mask]

    mean = (f + q) / 2.0
    diff = f - q  # FreeMoCap - Qualisys

    if diff.size < 2:
        return dict(mean=mean, diff=diff, bias=np.nan, sd=np.nan, loa_low=np.nan, loa_high=np.nan, n=int(diff.size))

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    return dict(mean=mean, diff=diff, bias=bias, sd=sd, loa_low=loa_low, loa_high=loa_high, n=int(diff.size))

if __name__ == "__main__":
    df_leglength_phase_samples, df_leglength_phase_summary, df_leglength_phase_comparison = main()

    print("Qualisys stance delta")
    print(df_leglength_phase_comparison[(df_leglength_phase_comparison["phase"] == "stance")]['q_delta_mm'])

    print("Qualisys swing delta")
    print(df_leglength_phase_comparison[(df_leglength_phase_comparison["phase"] == "swing")]['q_delta_mm'])

    print("FreeMoCap stance delta")
    print(df_leglength_phase_comparison[(df_leglength_phase_comparison["phase"] == "stance")]['fmc_delta_mm'])

    print("FreeMoCap swing delta")
    print(df_leglength_phase_comparison[(df_leglength_phase_comparison["phase"] == "swing")]['fmc_delta_mm'])

    f = 2
