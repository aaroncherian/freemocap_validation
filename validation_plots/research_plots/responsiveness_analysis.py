from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from skellymodels.managers.human import Human


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

OUTPUT_ROOT = Path(r"C:\Users\aaron\Documents\prosthetics_paper\responsiveness_analysis")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TRACKERS = ["qualisys", "rtmpose_dlc", "mediapipe", "rtmpose"]
FMC_TRACKERS = ["rtmpose_dlc", "mediapipe", "rtmpose"]

SYSTEM_LABELS = {
    "qualisys": "Qualisys",
    "rtmpose_dlc": "FMC-Hybrid",
    "mediapipe": "FMC-MediaPipe",
    "rtmpose": "FMC-RTMPose",
}

SYSTEM_STYLES = {
    "qualisys": {"color": "#4d4d4d", "symbol": "square"},
    "rtmpose_dlc": {"color": "#1f77b4", "symbol": "circle"},
    "mediapipe": {"color": "#e69f00", "symbol": "triangle-up"},
    "rtmpose": {"color": "#d62728", "symbol": "diamond"},
}

LEG_LENGTH_RECORDINGS = {
    "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
    "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
    "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
    "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
}
LEG_LENGTH_ORDER = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]
LEG_LENGTH_IMPOSED_MM = {"neg_5": -12.7, "neg_25": -6.35, "neutral": 0.0, "pos_25": 6.35, "pos_5": 12.7}

TOE_ANGLE_RECORDINGS = {
    "neg_6_0": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1"),
    "neg_3_0": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1"),
    "pos_3_0": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1"),
    "pos_6_0": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1"),
}
TOE_ANGLE_ORDER = ["neg_6_0", "neg_3_0", "neutral", "pos_3_0", "pos_6_0"]
TOE_ANGLE_IMPOSED_DEG = {"neg_6_0": -6.0, "neg_3_0": -3.0, "neutral": 0.0, "pos_3_0": 3.0, "pos_6_0": 6.0}

ANKLE_ALIGNMENT_RECORDINGS = {
    "neg_5_6": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1"),
    "neg_2_8": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1"),
    "pos_2_8": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1"),
    "pos_5_6": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1"),
}
ANKLE_ALIGNMENT_ORDER = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]
ANKLE_ALIGNMENT_IMPOSED_DEG = {"neg_5_6": -5.6, "neg_2_8": -2.8, "neutral": 0.0, "pos_2_8": 2.8, "pos_5_6": 5.6}

# Gait-cycle windows. These are easy to change if we decide on different definitions.
STANCE_END = 60.0
FPA_STANCE = (0.0, 60.0)
TOE_CLEARANCE_WINDOW = (70.0, 95.0)
KNEE_STANCE_WINDOW = (0.0, 60.0)
ANKLE_DORSIFLEXION_WINDOW = (20.0, 60.0)   # matches the existing peak_values.py logic
ANKLE_PLANTARFLEXION_WINDOW = (40.0, 80.0)
PELVIC_STANCE_WINDOW = (0.0, 60.0)

# FPA reference direction and ground normal, matching the existing paper scripts.
FPA_REFERENCE = np.array([0.0, 1.0, 0.0])
FPA_NORMAL = np.array([0.0, 0.0, 1.0])


# -----------------------------------------------------------------------------
# GENERIC HELPERS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    unit: str
    condition_order: list[str]


def metric_dir(metric_name: str) -> Path:
    out = OUTPUT_ROOT / metric_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_csv(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    df.to_csv(out_dir / name, index=False)


def add_change_from_neutral(condition_values: pd.DataFrame, condition_order: list[str]) -> pd.DataFrame:
    """Add delta-from-neutral values and descriptive uncertainty for each tracker/condition.

    For stride-level metrics, ``delta_sem`` propagates the SEM of the condition
    mean and neutral mean as sqrt(SEM_condition^2 + SEM_neutral^2). This is
    intended as descriptive within-trial uncertainty, not an independent-subject
    inferential standard error.
    """
    rows = []
    for tracker, tracker_df in condition_values.groupby("tracker"):
        neutral = tracker_df.loc[tracker_df["condition"] == "neutral"]
        if len(neutral) != 1:
            raise ValueError(f"Expected exactly one neutral value for {tracker}, found {len(neutral)}")

        neutral_value = float(neutral["value"].iloc[0])
        neutral_sd = float(neutral["sd"].iloc[0]) if "sd" in neutral.columns else np.nan
        neutral_n = int(neutral["n"].iloc[0]) if "n" in neutral.columns else 0
        neutral_sem = neutral_sd / np.sqrt(neutral_n) if neutral_n > 0 and np.isfinite(neutral_sd) else np.nan

        for condition in condition_order:
            row = tracker_df[tracker_df["condition"] == condition]
            if len(row) != 1:
                raise ValueError(f"Expected one value for {tracker}/{condition}, found {len(row)}")

            value = float(row["value"].iloc[0])
            sd = float(row["sd"].iloc[0]) if "sd" in row.columns else np.nan
            n = int(row["n"].iloc[0]) if "n" in row.columns else 0
            sem = sd / np.sqrt(n) if n > 0 and np.isfinite(sd) else np.nan
            delta_sem = np.sqrt(sem ** 2 + neutral_sem ** 2) if np.isfinite(sem) and np.isfinite(neutral_sem) else np.nan

            rows.append({
                "tracker": tracker,
                "condition": condition,
                "value": value,
                "neutral_value": neutral_value,
                "delta_from_neutral": value - neutral_value,
                "n": n,
                "sd": sd,
                "sem": sem,
                "neutral_sem": neutral_sem,
                "delta_sem": delta_sem,
            })

    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(out["condition"], categories=condition_order, ordered=True)
    return out.sort_values(["tracker", "condition"]).reset_index(drop=True)


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope for y = beta*x, constrained to pass through zero."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = float(np.dot(x, x))
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def fit_unconstrained(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def direction_agreement(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (~np.isclose(x, 0)) & (~np.isclose(y, 0))
    if not np.any(valid):
        return 0, 0, np.nan
    matched = int(np.sum(np.sign(x[valid]) == np.sign(y[valid])))
    total = int(np.sum(valid))
    return matched, total, 100.0 * matched / total


def build_fit_summary(
    changes: pd.DataFrame,
    reference_tracker: str = "qualisys",
    reference_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build slopes for each non-reference tracker.

    If reference_values is None:
        x = Qualisys delta-from-neutral for the same condition.
    Otherwise:
        x = externally known condition perturbation, e.g. imposed pylon-length change.
    """
    non_neutral = changes[changes["condition"] != "neutral"].copy()

    if reference_values is None:
        ref = (
            non_neutral[non_neutral["tracker"] == reference_tracker]
            [["condition", "delta_from_neutral"]]
            .rename(columns={"delta_from_neutral": "reference_delta"})
        )
    else:
        ref = pd.DataFrame({
            "condition": [c for c in reference_values if c != "neutral"],
            "reference_delta": [reference_values[c] for c in reference_values if c != "neutral"],
        })

    rows = []
    trackers = TRACKERS if reference_values is not None else FMC_TRACKERS

    for tracker in trackers:
        test = (
            non_neutral[non_neutral["tracker"] == tracker]
            [["condition", "delta_from_neutral"]]
            .rename(columns={"delta_from_neutral": "test_delta"})
        )
        paired = ref.merge(test, on="condition", how="inner")
        x = paired["reference_delta"].to_numpy(dtype=float)
        y = paired["test_delta"].to_numpy(dtype=float)

        beta = fit_through_origin(x, y)
        slope, intercept = fit_unconstrained(x, y)
        pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        rows.append({
            "tracker": tracker,
            "system": SYSTEM_LABELS[tracker],
            "slope_through_origin_beta": beta,
            "unconstrained_slope": slope,
            "unconstrained_intercept": intercept,
            "pearson_r": pearson_r,
            "n_conditions": len(paired),
        })

    return pd.DataFrame(rows)


def make_response_plot(
    changes: pd.DataFrame,
    fit_summary: pd.DataFrame,
    spec: MetricSpec,
    out_dir: Path,
    reference_values: dict[str, float] | None = None,
    reference_label: str = "Qualisys Δ",
) -> go.Figure:
    """Plot condition deltas against the chosen reference and through-zero fits."""
    non_neutral = changes[changes["condition"] != "neutral"].copy()

    if reference_values is None:
        ref = (
            non_neutral[non_neutral["tracker"] == "qualisys"]
            [["condition", "delta_from_neutral"]]
            .rename(columns={"delta_from_neutral": "reference_delta"})
        )
        systems_to_plot = FMC_TRACKERS
    else:
        ref = pd.DataFrame({
            "condition": [c for c in spec.condition_order if c != "neutral"],
            "reference_delta": [reference_values[c] for c in spec.condition_order if c != "neutral"],
        })
        systems_to_plot = TRACKERS

    paired_by_tracker = {}
    x_vals, y_vals = [0.0], [0.0]
    for tracker in systems_to_plot:
        test = non_neutral[non_neutral["tracker"] == tracker][["condition", "delta_from_neutral", "delta_sem"]].rename(
            columns={"delta_from_neutral": "test_delta"}
        )
        paired = ref.merge(test, on="condition", how="inner")
        paired_by_tracker[tracker] = paired
        x_vals.extend(paired["reference_delta"].tolist())
        y_vals.extend(paired["test_delta"].tolist())

    x_lo, x_hi = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    y_lo, y_hi = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
    x_pad = 0.12 * max(x_hi - x_lo, 1e-6)
    y_pad = 0.12 * max(y_hi - y_lo, 1e-6)
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad

    fig = go.Figure()

    if spec.name == "shank_length" and reference_values is not None:
        lo = min(x_lo, y_lo)
        hi = max(x_hi, y_hi)
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="1:1 response",
            line=dict(color="#A6A6A6", dash="dash", width=1.6), hoverinfo="skip",
        ))
        x_lo, x_hi, y_lo, y_hi = lo, hi, lo, hi

    for tracker in systems_to_plot:
        paired = paired_by_tracker[tracker]
        if paired.empty:
            continue

        style = SYSTEM_STYLES[tracker]
        beta_row = fit_summary[fit_summary["tracker"] == tracker]
        beta = float(beta_row["slope_through_origin_beta"].iloc[0]) if not beta_row.empty else np.nan
        error_y = None
        if spec.name != "shank_length" and paired["delta_sem"].notna().any():
            error_y = dict(type="data", array=paired["delta_sem"].fillna(0).to_numpy(), visible=True, thickness=1.0, width=3, color=style["color"])

        fig.add_trace(go.Scatter(
            x=paired["reference_delta"], y=paired["test_delta"], mode="markers",
            name=SYSTEM_LABELS[tracker], legendgroup=tracker, error_y=error_y,
            marker=dict(size=8, symbol=style["symbol"], color=style["color"], line=dict(width=0.7, color="black")),
            customdata=np.column_stack([paired["condition"].astype(str)]),
            hovertemplate=(
                f"<b>{SYSTEM_LABELS[tracker]}</b><br>"
                "Condition: %{customdata[0]}<br>"
                f"{reference_label}: %{{x:.2f}}<br>"
                f"Response Δ: %{{y:.2f}} {spec.unit}<extra></extra>"
            ),
        ))

        x_line = np.array([x_lo, x_hi])
        fig.add_trace(go.Scatter(
            x=x_line, y=beta * x_line, mode="lines",
            name=f"{SYSTEM_LABELS[tracker]} fit (β={beta:.2f})", legendgroup=tracker,
            line=dict(color=style["color"], width=1.6), hoverinfo="skip",
        ))

    fig.update_layout(
        template="simple_white", width=650, height=500,
        font=dict(family="Arial", size=15, color="black"),
        margin=dict(l=80, r=20, t=20, b=70),
        xaxis=dict(title=f"<b>{reference_label}</b>", range=[x_lo, x_hi], showline=True, linecolor="black", mirror=True, ticks="outside"),
        yaxis=dict(title=f"<b>Δ {spec.label} from neutral ({spec.unit})</b>", range=[y_lo, y_hi], showline=True, linecolor="black", mirror=True, ticks="outside"),
        legend=dict(orientation="h", x=0.5, y=1.02, xanchor="center", yanchor="bottom", font=dict(size=11)),
    )
    if spec.name == "shank_length" and reference_values is not None:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.write_image(out_dir / "response_match.pdf")
    fig.write_image(out_dir / "response_match.png", scale=3)
    fig.write_html(out_dir / "response_match.html")
    return fig


def finish_metric(
    per_stride: pd.DataFrame | None,
    condition_values: pd.DataFrame,
    spec: MetricSpec,
    reference_values: dict[str, float] | None = None,
    reference_label: str = "Qualisys Δ",
) -> pd.DataFrame:
    out_dir = metric_dir(spec.name)
    if per_stride is not None:
        save_csv(per_stride, out_dir, "per_stride_values.csv")
    save_csv(condition_values, out_dir, "condition_values.csv")

    changes = add_change_from_neutral(condition_values, spec.condition_order)
    save_csv(changes, out_dir, "changes_from_neutral.csv")

    fits = build_fit_summary(changes, reference_values=reference_values)
    fits.insert(0, "metric", spec.name)
    save_csv(fits, out_dir, "fit_summary.csv")

    make_response_plot(
        changes=changes,
        fit_summary=fits,
        spec=spec,
        out_dir=out_dir,
        reference_values=reference_values,
        reference_label=reference_label,
    )
    return fits


# -----------------------------------------------------------------------------
# SHANK LENGTH
# -----------------------------------------------------------------------------

def run_shank_length() -> list[pd.DataFrame]:
    rows = []
    for condition in LEG_LENGTH_ORDER:
        recording = LEG_LENGTH_RECORDINGS[condition]
        for tracker in TRACKERS:
            human = Human.from_data(recording / "validation" / tracker)
            knee = human.body.xyz.as_dict["right_knee"]
            ankle = human.body.xyz.as_dict["right_ankle"]
            values = np.linalg.norm(knee - ankle, axis=1)
            rows.append({
                "tracker": tracker,
                "condition": condition,
                "value": float(np.nanmedian(values)),
                "sd": float(np.nanstd(values, ddof=1)),
                "n": int(np.sum(np.isfinite(values))),
            })

    condition_values = pd.DataFrame(rows)
    spec = MetricSpec(
        name="shank_length",
        label="median shank length",
        unit="mm",
        condition_order=LEG_LENGTH_ORDER,
    )

    fits = finish_metric(
        per_stride=None,
        condition_values=condition_values,
        spec=spec,
        reference_values=LEG_LENGTH_IMPOSED_MM,
        reference_label="Imposed pylon-length change (mm)",
    )
    fits["reference"] = "imposed_pylon_length"
    return [fits]


# -----------------------------------------------------------------------------
# FPA
# -----------------------------------------------------------------------------

def calculate_fpa(foot_vector: np.ndarray) -> float:
    n = FPA_NORMAL / (np.linalg.norm(FPA_NORMAL) + 1e-12)
    a = FPA_REFERENCE
    b = foot_vector
    a_proj = a - np.dot(a, n) * n
    b_proj = b - np.dot(b, n) * n
    a_hat = a_proj / (np.linalg.norm(a_proj) + 1e-12)
    b_hat = b_proj / (np.linalg.norm(b_proj) + 1e-12)
    return float(np.degrees(np.arctan2(np.dot(np.cross(a_hat, b_hat), n), np.dot(a_hat, b_hat))))


def run_fpa() -> pd.DataFrame:
    stride_rows = []
    lo, hi = FPA_STANCE

    for condition in TOE_ANGLE_ORDER:
        recording = TOE_ANGLE_RECORDINGS[condition]
        for tracker in TRACKERS:
            csv_path = recording / "validation" / tracker / "trajectories" / "trajectories_per_stride.csv"
            df = pd.read_csv(csv_path)
            foot = df.query("marker in ['right_foot_index', 'right_heel']").pivot_table(
                index=["cycle", "percent_gait_cycle"], columns="marker", values=["x", "y", "z"]
            ).reset_index()

            vectors = pd.DataFrame({
                "cycle": foot["cycle"],
                "percent_gait_cycle": foot["percent_gait_cycle"],
                "x": foot["x"]["right_foot_index"] - foot["x"]["right_heel"],
                "y": foot["y"]["right_foot_index"] - foot["y"]["right_heel"],
                "z": foot["z"]["right_foot_index"] - foot["z"]["right_heel"],
            })
            vectors["fpa"] = vectors.apply(lambda r: calculate_fpa(np.array([r["x"], r["y"], r["z"]])), axis=1)
            vectors = vectors[(vectors["percent_gait_cycle"] >= lo) & (vectors["percent_gait_cycle"] <= hi)]

            for cycle, g in vectors.groupby("cycle"):
                stride_rows.append({
                    "tracker": tracker,
                    "condition": condition,
                    "cycle": cycle,
                    "value": float(g["fpa"].mean()),
                })

    per_stride = pd.DataFrame(stride_rows)
    condition_values = summarize_stride_metric(per_stride)
    spec = MetricSpec("fpa_stance_mean", "stance FPA", "°", TOE_ANGLE_ORDER)
    fits = finish_metric(
        per_stride, condition_values, spec,
        reference_values=TOE_ANGLE_IMPOSED_DEG,
        reference_label="Imposed toe-angle change",
    )
    fits["reference"] = "imposed_toe_angle"
    return fits


# -----------------------------------------------------------------------------
# TOE CLEARANCE
# -----------------------------------------------------------------------------

def run_toe_clearance() -> pd.DataFrame:
    stride_rows = []
    lo, hi = TOE_CLEARANCE_WINDOW

    for condition in ANKLE_ALIGNMENT_ORDER:
        recording = ANKLE_ALIGNMENT_RECORDINGS[condition]
        for tracker in TRACKERS:
            csv_path = recording / "validation" / tracker / "trajectories" / "trajectories_per_stride.csv"
            df = pd.read_csv(csv_path)
            toe = df.query("marker == 'right_foot_index'")[["cycle", "percent_gait_cycle", "z"]].copy()

            for cycle, g in toe.groupby("cycle"):
                window = g[(g["percent_gait_cycle"] > lo) & (g["percent_gait_cycle"] < hi)]
                if window.empty:
                    continue
                stride_rows.append({
                    "tracker": tracker,
                    "condition": condition,
                    "cycle": cycle,
                    "value": float(window["z"].min()),
                })

    per_stride = pd.DataFrame(stride_rows)
    condition_values = summarize_stride_metric(per_stride)
    spec = MetricSpec("minimum_toe_clearance", "minimum toe clearance", "mm", ANKLE_ALIGNMENT_ORDER)
    fits = finish_metric(
        per_stride, condition_values, spec,
        reference_values=ANKLE_ALIGNMENT_IMPOSED_DEG,
        reference_label="Imposed ankle-alignment change",
    )
    fits["reference"] = "imposed_ankle_alignment"
    return fits


# -----------------------------------------------------------------------------
# JOINT-ANGLE HELPERS
# -----------------------------------------------------------------------------

def load_joint_angle_strides(
    recordings: dict[str, Path],
    condition_order: list[str],
    tracker: str,
    joint: str,
    component: str,
    side_preference: tuple[str, ...] = ("right", "mid", "left"),
) -> pd.DataFrame:
    rows = []
    for condition in condition_order:
        csv_path = recordings[condition] / "validation" / tracker / "joint_angles" / "joint_angles_per_stride.csv"
        df = pd.read_csv(csv_path)
        df = df[(df["joint"] == joint) & (df["component"] == component)].copy()

        if "side" in df.columns and not df.empty:
            sides = set(df["side"].dropna().astype(str))
            chosen = next((s for s in side_preference if s in sides), None)
            if chosen is not None:
                df = df[df["side"] == chosen]

        if df.empty:
            raise ValueError(f"No rows for {tracker}/{condition}: joint={joint}, component={component}")

        df["condition"] = condition
        df["tracker"] = tracker
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def summarize_stride_metric(per_stride: pd.DataFrame) -> pd.DataFrame:
    return (
        per_stride.groupby(["tracker", "condition"], as_index=False)["value"]
        .agg(value="mean", sd="std", n="count")
    )


def extract_window_feature(
    df: pd.DataFrame,
    window: tuple[float, float],
    reducer: str,
    multiply: float = 1.0,
) -> pd.DataFrame:
    lo, hi = window
    d = df[(df["percent_gait_cycle"] >= lo) & (df["percent_gait_cycle"] <= hi)].copy()
    d["angle_for_metric"] = d["angle"] * multiply

    if reducer == "max":
        values = d.groupby(["tracker", "condition", "cycle"])["angle_for_metric"].max()
    elif reducer == "min":
        values = d.groupby(["tracker", "condition", "cycle"])["angle_for_metric"].min()
    elif reducer == "mean":
        values = d.groupby(["tracker", "condition", "cycle"])["angle_for_metric"].mean()
    elif reducer == "excursion":
        values = d.groupby(["tracker", "condition", "cycle"])["angle_for_metric"].agg(lambda x: x.max() - x.min())
    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    return values.reset_index(name="value")


# -----------------------------------------------------------------------------
# PELVIC OBLIQUITY
# -----------------------------------------------------------------------------

def run_pelvic_obliquity() -> list[pd.DataFrame]:
    all_rows = []
    for tracker in TRACKERS:
        all_rows.append(load_joint_angle_strides(
            LEG_LENGTH_RECORDINGS, LEG_LENGTH_ORDER, tracker,
            joint="pelvis", component="obliquity", side_preference=("mid", "right", "left"),
        ))
    df = pd.concat(all_rows, ignore_index=True)

    outputs = []

    # Signed mean during stance: preserves whether the pelvis shifts in one direction or the other.
    stance = extract_window_feature(df, PELVIC_STANCE_WINDOW, reducer="mean")
    stance_values = summarize_stride_metric(stance)
    stance_spec = MetricSpec("pelvic_obliquity_stance_mean", "stance pelvic obliquity", "°", LEG_LENGTH_ORDER)
    stance_fits = finish_metric(
        stance, stance_values, stance_spec,
        reference_values=LEG_LENGTH_IMPOSED_MM,
        reference_label="Imposed pylon-length change",
    )
    stance_fits["reference"] = "imposed_pylon_length"
    outputs.append(stance_fits)

    # Excursion is also useful to inspect because it is insensitive to a constant offset.
    excursion = extract_window_feature(df, (0.0, 100.0), reducer="excursion")
    excursion_values = summarize_stride_metric(excursion)
    excursion_spec = MetricSpec("pelvic_obliquity_excursion", "pelvic obliquity excursion", "°", LEG_LENGTH_ORDER)
    excursion_fits = finish_metric(
        excursion, excursion_values, excursion_spec,
        reference_values=LEG_LENGTH_IMPOSED_MM,
        reference_label="Imposed pylon-length change",
    )
    excursion_fits["reference"] = "imposed_pylon_length"
    outputs.append(excursion_fits)

    return outputs


# -----------------------------------------------------------------------------
# KNEE + ANKLE ANGLE FEATURES
# -----------------------------------------------------------------------------

def run_knee_and_ankle_features() -> list[pd.DataFrame]:
    knee_rows = []
    ankle_rows = []

    for tracker in TRACKERS:
        knee_rows.append(load_joint_angle_strides(
            ANKLE_ALIGNMENT_RECORDINGS, ANKLE_ALIGNMENT_ORDER, tracker,
            joint="knee", component="flex_ext", side_preference=("right",),
        ))
        ankle_rows.append(load_joint_angle_strides(
            ANKLE_ALIGNMENT_RECORDINGS, ANKLE_ALIGNMENT_ORDER, tracker,
            joint="ankle", component="dorsi_plantar", side_preference=("right",),
        ))

    knee = pd.concat(knee_rows, ignore_index=True)
    ankle = pd.concat(ankle_rows, ignore_index=True)

    outputs = []

    # Existing paper plotting flips the knee sign; do the same here so positive = flexion.
    knee_peak = extract_window_feature(knee, KNEE_STANCE_WINDOW, reducer="max", multiply=-1.0)
    knee_values = summarize_stride_metric(knee_peak)
    knee_spec = MetricSpec("knee_peak_flexion_stance", "peak stance knee flexion", "°", ANKLE_ALIGNMENT_ORDER)
    knee_fits = finish_metric(
        knee_peak, knee_values, knee_spec,
        reference_values=ANKLE_ALIGNMENT_IMPOSED_DEG,
        reference_label="Imposed ankle-alignment change",
    )
    knee_fits["reference"] = "imposed_ankle_alignment"
    outputs.append(knee_fits)

    # Existing peak_values.py already treats the maximum from 20-60% as peak dorsiflexion.
    ankle_dorsi = extract_window_feature(ankle, ANKLE_DORSIFLEXION_WINDOW, reducer="max")
    ankle_dorsi_values = summarize_stride_metric(ankle_dorsi)
    ankle_dorsi_spec = MetricSpec("ankle_peak_dorsiflexion", "peak ankle dorsiflexion", "°", ANKLE_ALIGNMENT_ORDER)
    dorsi_fits = finish_metric(
        ankle_dorsi, ankle_dorsi_values, ankle_dorsi_spec,
        reference_values=ANKLE_ALIGNMENT_IMPOSED_DEG,
        reference_label="Imposed ankle-alignment change",
    )
    dorsi_fits["reference"] = "imposed_ankle_alignment"
    outputs.append(dorsi_fits)

    # Minimum angle in late stance / early swing as peak plantarflexion.
    ankle_plantar = extract_window_feature(ankle, ANKLE_PLANTARFLEXION_WINDOW, reducer="min")
    ankle_plantar_values = summarize_stride_metric(ankle_plantar)
    ankle_plantar_spec = MetricSpec("ankle_peak_plantarflexion", "peak ankle plantarflexion", "°", ANKLE_ALIGNMENT_ORDER)
    plantar_fits = finish_metric(
        ankle_plantar, ankle_plantar_values, ankle_plantar_spec,
        reference_values=ANKLE_ALIGNMENT_IMPOSED_DEG,
        reference_label="Imposed ankle-alignment change",
    )
    plantar_fits["reference"] = "imposed_ankle_alignment"
    outputs.append(plantar_fits)

    return outputs



# -----------------------------------------------------------------------------
# CONSOLIDATED PAPER / SUPPLEMENT OUTPUTS
# -----------------------------------------------------------------------------

SUMMARY_DIR = OUTPUT_ROOT / "summary_outputs"

MAIN_METRICS = [
    "shank_length",
    "fpa_stance_mean",
    "minimum_toe_clearance",
    "pelvic_obliquity_excursion",
    "ankle_peak_dorsiflexion",
    "ankle_peak_plantarflexion",
]

METRIC_DISPLAY = {
    "shank_length": "Shank length",
    "fpa_stance_mean": "Stance FPA",
    "minimum_toe_clearance": "Minimum toe clearance",
    "pelvic_obliquity_excursion": "Pelvic obliquity excursion",
    "ankle_peak_dorsiflexion": "Peak ankle dorsiflexion",
    "ankle_peak_plantarflexion": "Peak ankle plantarflexion",
}

METRIC_UNITS = {
    "shank_length": "mm",
    "fpa_stance_mean": "°",
    "minimum_toe_clearance": "mm",
    "pelvic_obliquity_excursion": "°",
    "ankle_peak_dorsiflexion": "°",
    "ankle_peak_plantarflexion": "°",
}

METRIC_Y_LABELS = {
    "shank_length": "Δ shank length from neutral",
    "fpa_stance_mean": "Δ stance FPA from neutral",
    "minimum_toe_clearance": "Δ minimum toe clearance from neutral",
    "pelvic_obliquity_excursion": "Δ pelvic obliquity excursion from neutral",
    "ankle_peak_dorsiflexion": "Δ peak dorsiflexion from neutral",
    "ankle_peak_plantarflexion": "Δ peak plantarflexion from neutral",
}

METRIC_X_LABELS = {
    "shank_length": "Imposed pylon-length change",
    "fpa_stance_mean": "Imposed toe-angle change",
    "minimum_toe_clearance": "Imposed ankle-alignment change",
    "pelvic_obliquity_excursion": "Imposed pylon-length change",
    "ankle_peak_dorsiflexion": "Imposed ankle-alignment change",
    "ankle_peak_plantarflexion": "Imposed ankle-alignment change",
}

METRIC_X_UNITS = {
    "shank_length": "mm",
    "fpa_stance_mean": "°",
    "minimum_toe_clearance": "°",
    "pelvic_obliquity_excursion": "mm",
    "ankle_peak_dorsiflexion": "°",
    "ankle_peak_plantarflexion": "°",
}

METRIC_SLOPE_UNITS = {
    "shank_length": "mm/mm",
    "fpa_stance_mean": "°/°",
    "minimum_toe_clearance": "mm/°",
    "pelvic_obliquity_excursion": "°/mm",
    "ankle_peak_dorsiflexion": "°/°",
    "ankle_peak_plantarflexion": "°/°",
}

METRIC_DEFINITIONS = [
    {
        "metric": "shank_length",
        "display_name": METRIC_DISPLAY["shank_length"],
        "definition": "Median 3D right knee-to-ankle distance across frames",
        "condition_summary": "Median across frames",
        "response_reference": "Known imposed pylon-length change",
    },
    {
        "metric": "fpa_stance_mean",
        "display_name": METRIC_DISPLAY["fpa_stance_mean"],
        "definition": f"Mean foot progression angle within {FPA_STANCE[0]:.0f}-{FPA_STANCE[1]:.0f}% gait cycle for each stride",
        "condition_summary": "Mean across strides",
        "response_reference": "Known imposed alignment change",
    },
    {
        "metric": "minimum_toe_clearance",
        "display_name": METRIC_DISPLAY["minimum_toe_clearance"],
        "definition": f"Minimum right toe height within {TOE_CLEARANCE_WINDOW[0]:.0f}-{TOE_CLEARANCE_WINDOW[1]:.0f}% gait cycle for each stride",
        "condition_summary": "Mean across strides",
        "response_reference": "Known imposed alignment change",
    },
    {
        "metric": "pelvic_obliquity_excursion",
        "display_name": METRIC_DISPLAY["pelvic_obliquity_excursion"],
        "definition": "Peak-to-peak pelvic obliquity excursion over the full gait cycle for each stride",
        "condition_summary": "Mean across strides",
        "response_reference": "Known imposed alignment change",
    },
    {
        "metric": "ankle_peak_dorsiflexion",
        "display_name": METRIC_DISPLAY["ankle_peak_dorsiflexion"],
        "definition": f"Maximum ankle dorsi/plantarflexion angle within {ANKLE_DORSIFLEXION_WINDOW[0]:.0f}-{ANKLE_DORSIFLEXION_WINDOW[1]:.0f}% gait cycle for each stride",
        "condition_summary": "Mean across strides",
        "response_reference": "Known imposed alignment change",
    },
    {
        "metric": "ankle_peak_plantarflexion",
        "display_name": METRIC_DISPLAY["ankle_peak_plantarflexion"],
        "definition": f"Minimum ankle dorsi/plantarflexion angle within {ANKLE_PLANTARFLEXION_WINDOW[0]:.0f}-{ANKLE_PLANTARFLEXION_WINDOW[1]:.0f}% gait cycle for each stride",
        "condition_summary": "Mean across strides",
        "response_reference": "Known imposed alignment change",
    },
]


def load_saved_metric(metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = OUTPUT_ROOT / metric
    changes_path = out_dir / "changes_from_neutral.csv"
    fits_path = out_dir / "fit_summary.csv"
    if not changes_path.exists() or not fits_path.exists():
        raise FileNotFoundError(
            f"Missing saved outputs for {metric}. Expected:\n  {changes_path}\n  {fits_path}"
        )
    return pd.read_csv(changes_path), pd.read_csv(fits_path)


def imposed_reference_values(metric: str) -> dict[str, float]:
    if metric in {"shank_length", "pelvic_obliquity_excursion", "pelvic_obliquity_stance_mean"}:
        return LEG_LENGTH_IMPOSED_MM
    if metric == "fpa_stance_mean":
        return TOE_ANGLE_IMPOSED_DEG
    if metric in {"minimum_toe_clearance", "knee_peak_flexion_stance", "ankle_peak_dorsiflexion", "ankle_peak_plantarflexion"}:
        return ANKLE_ALIGNMENT_IMPOSED_DEG
    raise KeyError(f"No imposed perturbation mapping defined for metric={metric}")


def response_data(metric: str, changes: pd.DataFrame, tracker: str) -> pd.DataFrame:
    reference = imposed_reference_values(metric)
    test = changes[(changes["tracker"] == tracker) & (changes["condition"] != "neutral")].copy()
    test["imposed_change"] = test["condition"].map(reference).astype(float)
    return test[["condition", "imposed_change", "delta_from_neutral", "delta_sem"]].rename(
        columns={"delta_from_neutral": "response_delta"}
    )


def fit_row(fits: pd.DataFrame, tracker: str) -> pd.Series:
    row = fits[fits["tracker"] == tracker]
    if row.empty:
        raise ValueError(f"No fit row found for tracker={tracker}")
    return row.iloc[0]


def add_response_panel(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    metric: str,
    changes: pd.DataFrame,
    fits: pd.DataFrame,
    trackers: list[str],
    show_legend_for: set[str],
    show_reference_legend: bool,
    annotate_stats: bool,
) -> None:
    """Add one imposed-perturbation response panel.

    x is always the known imposed alignment change. Qualisys and each FMC
    reconstruction are therefore fitted independently against the same
    noise-free experimental input. A 1:1 line is shown only for shank length,
    where the imposed and measured quantities are directly comparable.
    """
    data_by_tracker = {tracker: response_data(metric, changes, tracker) for tracker in trackers}

    x_values = [0.0]
    y_values = [0.0]
    for data in data_by_tracker.values():
        x_values.extend(data["imposed_change"].to_list())
        y_values.extend(data["response_delta"].to_list())
        if "delta_sem" in data.columns:
            sem = data["delta_sem"].to_numpy(dtype=float)
            y = data["response_delta"].to_numpy(dtype=float)
            valid = np.isfinite(sem)
            y_values.extend((y[valid] - sem[valid]).tolist())
            y_values.extend((y[valid] + sem[valid]).tolist())

    x_lo, x_hi = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    y_lo, y_hi = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    x_pad = 0.10 * max(x_hi - x_lo, 1e-6)
    y_pad = 0.14 * max(y_hi - y_lo, 1e-6)
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad

    # A 1:1 reference is physically meaningful only for measured shank length.
    if metric == "shank_length":
        lo = min(x_lo, y_lo)
        hi = max(x_hi, y_hi)
        fig.add_trace(
            go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", name="1:1 response",
                legendgroup="one_to_one", showlegend=show_reference_legend,
                line=dict(color="#A6A6A6", dash="dash", width=1.5), hoverinfo="skip",
            ),
            row=row, col=col,
        )
        x_lo, x_hi, y_lo, y_hi = lo, hi, lo, hi

    stats_text = []
    for tracker in trackers:
        data = data_by_tracker[tracker]
        if data.empty:
            continue

        style = SYSTEM_STYLES[tracker]
        stats = fit_row(fits, tracker)
        beta = float(stats["slope_through_origin_beta"])

        # Propagated SEM is shown for stride-level metrics. Shank length is
        # frame-based and temporally autocorrelated, so no SEM bars are drawn.
        error_y = None
        if metric != "shank_length" and data["delta_sem"].notna().any():
            error_y = dict(
                type="data", array=data["delta_sem"].fillna(0).to_numpy(),
                visible=True, thickness=1.0, width=3, color=style["color"],
            )

        fig.add_trace(
            go.Scatter(
                x=data["imposed_change"], y=data["response_delta"], mode="markers",
                name=SYSTEM_LABELS[tracker], legendgroup=tracker,
                showlegend=tracker in show_legend_for,
                marker=dict(size=8, symbol=style["symbol"], color=style["color"], line=dict(width=0.6, color="black")),
                error_y=error_y,
                customdata=np.column_stack([data["condition"].astype(str)]),
                hovertemplate=(
                    f"<b>{SYSTEM_LABELS[tracker]}</b><br>"
                    "Condition: %{customdata[0]}<br>"
                    f"Imposed change: %{{x:.2f}} {METRIC_X_UNITS[metric]}<br>"
                    f"Response Δ: %{{y:.2f}} {METRIC_UNITS[metric]}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

        x_line = np.array([x_lo, x_hi])
        fig.add_trace(
            go.Scatter(
                x=x_line, y=beta * x_line, mode="lines", showlegend=False,
                legendgroup=tracker, line=dict(color=style["color"], width=1.8), hoverinfo="skip",
            ),
            row=row, col=col,
        )
        stats_text.append(f"{SYSTEM_LABELS[tracker]}: β = {beta:.2f}")

    fig.update_xaxes(
        range=[x_lo, x_hi], title_text=f"<b>{METRIC_X_LABELS[metric]} ({METRIC_X_UNITS[metric]})</b>",
        showline=True, linecolor="black", mirror=True, ticks="outside", ticklen=3,
        row=row, col=col,
    )
    fig.update_yaxes(
        range=[y_lo, y_hi], title_text=f"<b>{METRIC_Y_LABELS[metric]} ({METRIC_UNITS[metric]})</b>",
        showline=True, linecolor="black", mirror=True, ticks="outside", ticklen=3,
        row=row, col=col,
    )

    if metric == "shank_length":
        subplot_idx = (row - 1) * 3 + col
        xref = "x" if subplot_idx == 1 else f"x{subplot_idx}"
        fig.update_yaxes(scaleanchor=xref, scaleratio=1, row=row, col=col)

    if annotate_stats:
        subplot_idx = (row - 1) * 3 + col
        xdomain = "x domain" if subplot_idx == 1 else f"x{subplot_idx} domain"
        ydomain = "y domain" if subplot_idx == 1 else f"y{subplot_idx} domain"
        fig.add_annotation(
            x=0.03, y=0.97, xref=xdomain, yref=ydomain,
            text="<br>".join(stats_text), showarrow=False,
            xanchor="left", yanchor="top", align="left",
            font=dict(size=13), bgcolor="rgba(255,255,255,0.72)", borderpad=2,
        )


def make_main_response_figure() -> go.Figure:
    """
    Main-text 2x3 figure.

    Every panel uses the known imposed alignment perturbation on x. Qualisys and
    FMC-Hybrid are fit separately against that same experimental input. Only the
    shank-length panel includes a 1:1 line because only there is a unit-for-unit
    physical response expected.
    """
    from plotly.subplots import make_subplots

    panel_titles = [
        "A  Shank length",
        "B  Stance FPA",
        "C  Minimum toe clearance",
        "D  Pelvic obliquity excursion",
        "E  Peak ankle dorsiflexion",
        "F  Peak ankle plantarflexion",
    ]
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=panel_titles,
        horizontal_spacing=0.06, vertical_spacing=0.11,
    )

    legend_seen: set[str] = set()
    for idx, metric in enumerate(MAIN_METRICS):
        row, col = idx // 3 + 1, idx % 3 + 1
        changes, fits = load_saved_metric(metric)
        trackers = ["qualisys", "rtmpose_dlc"]
        show_legend_for = {t for t in trackers if t not in legend_seen}
        show_reference = idx == 0
        add_response_panel(
            fig, row=row, col=col, metric=metric, changes=changes, fits=fits,
            trackers=trackers, show_legend_for=show_legend_for,
            show_reference_legend=show_reference, annotate_stats=True,
        )
        legend_seen.update(trackers)

    for ann in fig.layout.annotations:
        if ann.text.startswith(tuple("ABCDEF")):
            ann.font = dict(size=18, family="Arial", color="black")

    fig.update_layout(
        template="simple_white", width=1650, height=980,
        font=dict(family="Arial", size=14, color="black"),
        margin=dict(l=90, r=30, t=90, b=75),
        legend=dict(
            orientation="h", x=0.5, y=1.055, xanchor="center", yanchor="bottom",
            font=dict(size=14), bgcolor="rgba(255,255,255,0)",
        ),
    )
    fig.update_xaxes(tickfont=dict(size=13), title_font=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=13), title_font=dict(size=16))
    return fig


def make_supplementary_all_systems_figure() -> go.Figure:
    """Optional 2x3 supplement figure with every available reconstruction system."""
    from plotly.subplots import make_subplots

    panel_titles = [
        "A  Shank length",
        "B  Stance FPA",
        "C  Minimum toe clearance",
        "D  Pelvic obliquity excursion",
        "E  Peak ankle dorsiflexion",
        "F  Peak ankle plantarflexion",
    ]
    fig = make_subplots(rows=2, cols=3, subplot_titles=panel_titles, horizontal_spacing=0.06, vertical_spacing=0.11)

    legend_seen: set[str] = set()
    for idx, metric in enumerate(MAIN_METRICS):
        row, col = idx // 3 + 1, idx % 3 + 1
        changes, fits = load_saved_metric(metric)
        trackers = ["qualisys", "rtmpose_dlc", "rtmpose", "mediapipe"]
        show_legend_for = {t for t in trackers if t not in legend_seen}
        add_response_panel(
            fig, row=row, col=col, metric=metric, changes=changes, fits=fits,
            trackers=trackers, show_legend_for=show_legend_for,
            show_reference_legend=idx == 0, annotate_stats=False,
        )
        legend_seen.update(trackers)

    for ann in fig.layout.annotations:
        if ann.text.startswith(tuple("ABCDEF")):
            ann.font = dict(size=14, family="Arial", color="black")

    fig.update_layout(
        template="simple_white", width=1800, height=1200,
        font=dict(family="Arial", size=13, color="black"),
        margin=dict(l=80, r=25, t=80, b=80),
        legend=dict(
            orientation="h", x=0.5, y=1.055, xanchor="center", yanchor="bottom",
            font=dict(size=13), bgcolor="rgba(255,255,255,0)",
        ),
    )
    fig.update_xaxes(tickfont=dict(size=11), title_font=dict(size=12))
    fig.update_yaxes(tickfont=dict(size=11), title_font=dict(size=12))
    return fig


def build_response_summary_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Supplementary slope + Pearson-r table against the imposed perturbation."""
    keep = combined[combined["metric"].isin(MAIN_METRICS)].copy()
    keep = keep[keep["reference"].isin(["imposed_pylon_length", "imposed_toe_angle", "imposed_ankle_alignment"])]

    metric_order = {metric: i for i, metric in enumerate(MAIN_METRICS)}
    system_order = {"qualisys": 0, "rtmpose_dlc": 1, "rtmpose": 2, "mediapipe": 3}
    keep["metric_order"] = keep["metric"].map(metric_order)
    keep["system_order"] = keep["tracker"].map(system_order)
    keep = keep.sort_values(["metric_order", "system_order"])

    keep["Outcome"] = keep["metric"].map(METRIC_DISPLAY)
    keep["Perturbation"] = keep["metric"].map({
        "shank_length": "Pylon length",
        "fpa_stance_mean": "Toe angle",
        "minimum_toe_clearance": "Ankle alignment",
        "pelvic_obliquity_excursion": "Pylon length",
        "ankle_peak_dorsiflexion": "Ankle alignment",
        "ankle_peak_plantarflexion": "Ankle alignment",
    })
    keep["System"] = keep["system"]
    keep["Beta"] = keep["slope_through_origin_beta"]
    keep["Slope_units"] = keep["metric"].map(METRIC_SLOPE_UNITS)
    keep["Pearson_r"] = keep["pearson_r"]
    keep["N_conditions"] = keep["n_conditions"]

    return keep[["Outcome", "Perturbation", "System", "Beta", "Slope_units", "Pearson_r", "N_conditions"]].reset_index(drop=True)


def build_main_response_stats(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in MAIN_METRICS:
        subset = combined[(combined["metric"] == metric) & (combined["tracker"].isin(["qualisys", "rtmpose_dlc"]))].copy()
        rows.append(subset)

    out = pd.concat(rows, ignore_index=True)
    out["Outcome"] = out["metric"].map(METRIC_DISPLAY)
    out["System"] = out["system"]
    out["Beta"] = out["slope_through_origin_beta"]
    out["Slope_units"] = out["metric"].map(METRIC_SLOPE_UNITS)
    out["Pearson_r"] = out["pearson_r"]
    return out[["Outcome", "System", "Beta", "Slope_units", "Pearson_r", "n_conditions"]].rename(columns={"n_conditions": "N_conditions"})


def build_main_slope_comparison(main_stats: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-outcome comparison of Qualisys and Hybrid response slopes."""
    wide_beta = main_stats.pivot(index="Outcome", columns="System", values="Beta").reset_index()
    wide_r = main_stats.pivot(index="Outcome", columns="System", values="Pearson_r").reset_index()
    wide_beta = wide_beta.rename(columns={"Qualisys": "Qualisys_beta", "FMC-Hybrid": "Hybrid_beta"})
    wide_r = wide_r.rename(columns={"Qualisys": "Qualisys_r", "FMC-Hybrid": "Hybrid_r"})
    out = wide_beta.merge(wide_r, on="Outcome", how="left")
    out["Hybrid_minus_Qualisys_beta"] = out["Hybrid_beta"] - out["Qualisys_beta"]
    return out


def write_typst_response_table(table: pd.DataFrame, path: Path) -> None:
    """Write a compact Typst table snippet for the supplement."""
    lines = [
        "#table(",
        "  columns: (1.45fr, 1.05fr, 1.15fr, 0.55fr, 0.7fr, 0.55fr),",
        "  inset: 5pt,",
        "  align: (left, left, left, center, center, center),",
        "  table.header([*Outcome*], [*Perturbation*], [*System*], [$beta$], [*Units*], [*Pearson $r$]),",
    ]
    for row in table.itertuples(index=False):
        outcome = str(row.Outcome).replace("[", "\\[").replace("]", "\\]")
        perturbation = str(row.Perturbation).replace("[", "\\[").replace("]", "\\]")
        system = str(row.System).replace("[", "\\[").replace("]", "\\]")
        lines.append(f"  [{outcome}], [{perturbation}], [{system}], [{row.Beta:.2f}], [{row.Slope_units}], [{row.Pearson_r:.2f}],")
    lines.append(")")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_outputs(combined: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    definitions = pd.DataFrame(METRIC_DEFINITIONS)
    definitions.to_csv(SUMMARY_DIR / "metric_definitions.csv", index=False)

    main_stats = build_main_response_stats(combined)
    main_stats.to_csv(SUMMARY_DIR / "main_response_stats.csv", index=False)
    slope_comparison = build_main_slope_comparison(main_stats)
    slope_comparison.to_csv(SUMMARY_DIR / "main_slope_comparison.csv", index=False)

    supp_table = build_response_summary_table(combined)
    supp_table.to_csv(SUMMARY_DIR / "supplementary_response_table.csv", index=False)
    write_typst_response_table(supp_table, SUMMARY_DIR / "supplementary_response_table.typ")

    main_fig = make_main_response_figure()
    main_fig.write_image(SUMMARY_DIR / "main_response_match_2x3.pdf")
    main_fig.write_image(SUMMARY_DIR / "main_response_match_2x3.png", scale=2)
    main_fig.write_html(SUMMARY_DIR / "main_response_match_2x3.html")

    supp_fig = make_supplementary_all_systems_figure()
    supp_fig.write_image(SUMMARY_DIR / "supplementary_all_systems_2x3.pdf")
    supp_fig.write_image(SUMMARY_DIR / "supplementary_all_systems_2x3.png", scale=2)
    supp_fig.write_html(SUMMARY_DIR / "supplementary_all_systems_2x3.html")

    print("\nSummary outputs:")
    print(f"  {SUMMARY_DIR / 'main_response_match_2x3.pdf'}")
    print(f"  {SUMMARY_DIR / 'main_response_stats.csv'}")
    print(f"  {SUMMARY_DIR / 'main_slope_comparison.csv'}")
    print(f"  {SUMMARY_DIR / 'supplementary_response_table.csv'}")
    print(f"  {SUMMARY_DIR / 'supplementary_response_table.typ'}")
    print(f"  {SUMMARY_DIR / 'supplementary_all_systems_2x3.pdf'}")


# -----------------------------------------------------------------------------
# RUN ALL
# -----------------------------------------------------------------------------

def main() -> None:
    pio.kaleido.scope.mathjax = None
    all_fits = []

    print("Running shank length...")
    all_fits.extend(run_shank_length())

    print("Running FPA...")
    all_fits.append(run_fpa())

    print("Running toe clearance...")
    all_fits.append(run_toe_clearance())

    print("Running pelvic obliquity...")
    all_fits.extend(run_pelvic_obliquity())

    print("Running knee/ankle features...")
    all_fits.extend(run_knee_and_ankle_features())

    combined = pd.concat(all_fits, ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "all_fit_summaries.csv", index=False)

    # Paper-facing outputs: main Hybrid-focused 2x3 figure, compact main stats,
    # and all-system supplementary table/figure.
    write_summary_outputs(combined)

    print("\n" + "=" * 100)
    print("RESPONSE-MATCHING SLOPES")
    print("β = 1 means the magnitude of change matches the reference across conditions.")
    print("Pearson r describes how closely the pattern of condition changes tracks the reference.")
    print("=" * 100)
    cols = ["metric", "system", "slope_through_origin_beta", "pearson_r"]
    print(combined[cols].round(3).to_string(index=False))
    print(f"\nSaved all outputs under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
