from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# FINAL PAPER-FACING RESPONSE ANALYSIS
# -----------------------------------------------------------------------------
# Primary question:
#   When prosthesis alignment changes, does FMC-Hybrid measure the same change
#   from neutral that Qualisys measures?
#
# Main figure:
#   x = imposed alignment condition
#   y = change from neutral
#   Qualisys + FMC-Hybrid connected through the observed condition values
#   NO fitted response lines for biological outcomes.
#
# Main numerical summaries:
#   - Pearson r between the four non-neutral Qualisys and Hybrid changes
#   - mean signed difference: mean(Delta Hybrid - Delta Qualisys)
#
# Supplement:
#   - condition-level differences and paired-stride bootstrap CIs
#   - all-system overlays and agreement table
#   - response slopes versus imposed perturbation, clearly labeled descriptive
#
# Shank length is the exception: the imposed pylon-length change is itself a
# physical ground-truth change in length, so a 1:1 reference is meaningful.
# -----------------------------------------------------------------------------

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

LEG_LENGTH_IMPOSED_MM = {
    "neg_5": -12.7, "neg_25": -6.35, "neutral": 0.0,
    "pos_25": 6.35, "pos_5": 12.7,
}
TOE_ANGLE_IMPOSED_DEG = {
    "neg_6_0": -6.0, "neg_3_0": -3.0, "neutral": 0.0,
    "pos_3_0": 3.0, "pos_6_0": 6.0,
}
ANKLE_ALIGNMENT_IMPOSED_DEG = {
    "neg_5_6": -5.6, "neg_2_8": -2.8, "neutral": 0.0,
    "pos_2_8": 2.8, "pos_5_6": 5.6,
}

SYSTEM_LABELS = {
    "qualisys": "Qualisys",
    "rtmpose_dlc": "FMC-Hybrid",
    "rtmpose": "FMC-RTMPose",
    "mediapipe": "FMC-MediaPipe",
}

SYSTEM_STYLES = {
    "qualisys": {"color": "#4d4d4d", "symbol": "square"},
    "rtmpose_dlc": {"color": "#1f77b4", "symbol": "circle"},
    "rtmpose": {"color": "#d62728", "symbol": "diamond"},
    "mediapipe": {"color": "#e69f00", "symbol": "triangle-up"},
}

FMC_TRACKERS = ["rtmpose_dlc", "rtmpose", "mediapipe"]


def imposed_values(metric: str) -> dict[str, float]:
    if metric in {"shank_length", "pelvic_obliquity_excursion"}:
        return LEG_LENGTH_IMPOSED_MM
    if metric == "fpa_stance_mean":
        return TOE_ANGLE_IMPOSED_DEG
    return ANKLE_ALIGNMENT_IMPOSED_DEG


def load_changes(root: Path, metric: str) -> pd.DataFrame:
    path = root / metric / "changes_from_neutral.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    # Neutral is the reference subtracted from itself, so its delta uncertainty
    # is exactly zero. Older versions propagated neutral SEM against itself.
    if "delta_sem" in df.columns:
        df.loc[df["condition"] == "neutral", "delta_sem"] = 0.0
    return df


def load_fits(root: Path, metric: str) -> pd.DataFrame:
    path = root / metric / "fit_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def paired_condition_changes(root: Path, metric: str, tracker: str) -> pd.DataFrame:
    changes = load_changes(root, metric)
    q_cols = ["condition", "delta_from_neutral"]
    f_cols = ["condition", "delta_from_neutral"]
    if "delta_sem" in changes.columns:
        q_cols.append("delta_sem")
        f_cols.append("delta_sem")

    q = changes[(changes["tracker"] == "qualisys") & (changes["condition"] != "neutral")][q_cols].copy()
    f = changes[(changes["tracker"] == tracker) & (changes["condition"] != "neutral")][f_cols].copy()
    q = q.rename(columns={"delta_from_neutral": "qualisys_delta", "delta_sem": "qualisys_delta_sem"})
    f = f.rename(columns={"delta_from_neutral": "fmc_delta", "delta_sem": "fmc_delta_sem"})

    out = q.merge(f, on="condition", how="inner")
    out["metric"] = metric
    out["outcome"] = METRIC_DISPLAY[metric]
    out["system"] = SYSTEM_LABELS[tracker]
    out["unit"] = METRIC_UNITS[metric]
    out["imposed_change"] = out["condition"].map(imposed_values(metric)).astype(float)
    out["change_difference"] = out["fmc_delta"] - out["qualisys_delta"]
    out["abs_change_difference"] = out["change_difference"].abs()
    out["same_direction"] = np.sign(out["fmc_delta"]) == np.sign(out["qualisys_delta"])
    return out.sort_values("imposed_change").reset_index(drop=True)


def agreement_summary(paired: pd.DataFrame) -> dict:
    q = paired["qualisys_delta"].to_numpy(float)
    f = paired["fmc_delta"].to_numpy(float)
    e = f - q
    r = float(np.corrcoef(q, f)[0, 1]) if len(q) > 1 and np.std(q) > 0 and np.std(f) > 0 else np.nan
    return {
        "metric": paired["metric"].iloc[0],
        "outcome": paired["outcome"].iloc[0],
        "system": paired["system"].iloc[0],
        "unit": paired["unit"].iloc[0],
        "pearson_r_vs_qualisys_change": r,
        "mean_signed_change_difference": float(np.mean(e)),
        "mean_abs_change_difference": float(np.mean(np.abs(e))),
        "max_abs_change_difference": float(np.max(np.abs(e))),
        "direction_agreement_n": int(paired["same_direction"].sum()),
        "direction_agreement_total": int(len(paired)),
    }


def fit_beta(root: Path, metric: str, tracker: str) -> float:
    fits = load_fits(root, metric)
    if fits.empty:
        return np.nan
    row = fits[fits["tracker"] == tracker]
    if row.empty or "slope_through_origin_beta" not in row.columns:
        return np.nan
    return float(row["slope_through_origin_beta"].iloc[0])


def bootstrap_paired_change_difference(root: Path, metric: str, n_boot: int = 5000, seed: int = 12345):
    """Paired-stride bootstrap for (ΔHybrid - ΔQualisys) in each condition.

    These CIs describe precision of the within-participant condition summaries;
    they are not population-level confidence intervals.
    """
    path = root / metric / "per_stride_values.csv"
    if not path.exists():
        return None

    strides = pd.read_csv(path)
    q = strides[strides["tracker"] == "qualisys"][["condition", "cycle", "value"]].rename(columns={"value": "q"})
    h = strides[strides["tracker"] == "rtmpose_dlc"][["condition", "cycle", "value"]].rename(columns={"value": "h"})
    paired = q.merge(h, on=["condition", "cycle"], how="inner")
    if paired.empty:
        return None

    paired["system_difference"] = paired["h"] - paired["q"]
    conditions = [c for c in imposed_values(metric) if c != "neutral"]
    groups = {c: paired.loc[paired["condition"] == c, "system_difference"].to_numpy(float) for c in ["neutral", *conditions]}
    if any(len(v) == 0 for v in groups.values()):
        return None

    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, len(conditions)), dtype=float)
    for b in range(n_boot):
        neutral = groups["neutral"]
        neutral_mean = float(np.mean(rng.choice(neutral, size=len(neutral), replace=True)))
        for i, condition in enumerate(conditions):
            values = groups[condition]
            condition_mean = float(np.mean(rng.choice(values, size=len(values), replace=True)))
            samples[b, i] = condition_mean - neutral_mean

    point = paired_condition_changes(root, metric, "rtmpose_dlc")
    rows = []
    for i, condition in enumerate(conditions):
        row = point[point["condition"] == condition].iloc[0]
        rows.append({
            "metric": metric,
            "outcome": METRIC_DISPLAY[metric],
            "condition": condition,
            "imposed_change": float(imposed_values(metric)[condition]),
            "change_difference": float(row["change_difference"]),
            "bootstrap_ci_low": float(np.quantile(samples[:, i], 0.025)),
            "bootstrap_ci_high": float(np.quantile(samples[:, i], 0.975)),
            "n_condition_paired_strides": int(len(groups[condition])),
            "n_neutral_paired_strides": int(len(groups["neutral"])),
        })
    return pd.DataFrame(rows)


def panel_titles() -> list[str]:
    return [f"{letter}  {METRIC_DISPLAY[m]}" for letter, m in zip("ABCDEF", MAIN_METRICS)]


def style_figure(fig: go.Figure, width: int = 1650, height: int = 980) -> None:
    for ann in fig.layout.annotations:
        if ann.text and ann.text.startswith(tuple("ABCDEF")):
            ann.font = dict(size=18, family="Arial", color="black")
    fig.update_layout(
        template="simple_white", width=width, height=height,
        font=dict(family="Arial", size=14, color="black"),
        margin=dict(l=85, r=25, t=85, b=70),
        legend=dict(orientation="h", x=0.5, y=1.055, xanchor="center", yanchor="bottom", font=dict(size=14)),
    )
    fig.update_xaxes(tickfont=dict(size=13), title_font=dict(size=16), showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(tickfont=dict(size=13), title_font=dict(size=16), showline=True, linecolor="black", ticks="outside")


def add_condition_trace(fig: go.Figure, d: pd.DataFrame, tracker: str, metric: str, row: int, col: int, showlegend: bool) -> None:
    style = SYSTEM_STYLES[tracker]
    error_y = None
    if metric != "shank_length" and "delta_sem" in d.columns:
        sem = d["delta_sem"].fillna(0).to_numpy(float)
        error_y = dict(type="data", array=sem, visible=True, thickness=1.1, width=3, color=style["color"])

    fig.add_trace(go.Scatter(
        x=d["imposed_change"], y=d["delta_from_neutral"], mode="lines+markers",
        name=SYSTEM_LABELS[tracker], legendgroup=tracker, showlegend=showlegend,
        line=dict(color=style["color"], width=2.0),
        marker=dict(color=style["color"], symbol=style["symbol"], size=8, line=dict(color="black", width=0.7)),
        error_y=error_y,
        customdata=d[["condition"]].to_numpy(),
        hovertemplate=(
            f"<b>{SYSTEM_LABELS[tracker]}</b><br>"
            "Condition: %{customdata[0]}<br>"
            "Imposed change: %{x:.2f}<br>"
            "Change from neutral: %{y:.3f}<extra></extra>"
        ),
    ), row=row, col=col)


def make_main_condition_response_figure(root: Path) -> go.Figure:
    """Primary paper figure: actual condition responses, no biological response fit."""
    fig = make_subplots(rows=2, cols=3, subplot_titles=panel_titles(), horizontal_spacing=0.055, vertical_spacing=0.105)

    for idx, metric in enumerate(MAIN_METRICS):
        row, col = idx // 3 + 1, idx % 3 + 1
        changes = load_changes(root, metric)
        imposed = imposed_values(metric)

        # Light zero-change reference for biological outcomes.
        if metric != "shank_length":
            fig.add_hline(y=0, line=dict(color="#C8C8C8", dash="dash", width=1.0), row=row, col=col)

        for tracker in ["qualisys", "rtmpose_dlc"]:
            d = changes[changes["tracker"] == tracker].copy()
            d["imposed_change"] = d["condition"].map(imposed).astype(float)
            d = d.sort_values("imposed_change")
            add_condition_trace(fig, d, tracker, metric, row, col, showlegend=(idx == 0))

        if metric == "shank_length":
            lo, hi = min(imposed.values()), max(imposed.values())
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", name="1:1 physical response",
                legendgroup="one_to_one", showlegend=True,
                line=dict(color="#A6A6A6", dash="dash", width=1.5), hoverinfo="skip",
            ), row=row, col=col)

        fig.update_xaxes(title_text=f"<b>{METRIC_X_LABELS[metric]} ({METRIC_X_UNITS[metric]})</b>", row=row, col=col)
        fig.update_yaxes(title_text=f"<b>{METRIC_Y_LABELS[metric]} ({METRIC_UNITS[metric]})</b>", row=row, col=col)

    style_figure(fig)
    return fig


def make_supplementary_all_systems_figure(root: Path) -> go.Figure:
    """Same condition-response view with all reconstruction systems."""
    fig = make_subplots(rows=2, cols=3, subplot_titles=panel_titles(), horizontal_spacing=0.055, vertical_spacing=0.105)

    for idx, metric in enumerate(MAIN_METRICS):
        row, col = idx // 3 + 1, idx % 3 + 1
        changes = load_changes(root, metric)
        imposed = imposed_values(metric)
        if metric != "shank_length":
            fig.add_hline(y=0, line=dict(color="#C8C8C8", dash="dash", width=1.0), row=row, col=col)

        for tracker in ["qualisys", "rtmpose_dlc", "rtmpose", "mediapipe"]:
            d = changes[changes["tracker"] == tracker].copy()
            if d.empty:
                continue
            d["imposed_change"] = d["condition"].map(imposed).astype(float)
            d = d.sort_values("imposed_change")
            add_condition_trace(fig, d, tracker, metric, row, col, showlegend=(idx == 0))

        if metric == "shank_length":
            lo, hi = min(imposed.values()), max(imposed.values())
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", name="1:1 physical response",
                legendgroup="one_to_one", showlegend=True,
                line=dict(color="#A6A6A6", dash="dash", width=1.5), hoverinfo="skip",
            ), row=row, col=col)

        fig.update_xaxes(title_text=f"<b>{METRIC_X_LABELS[metric]} ({METRIC_X_UNITS[metric]})</b>", row=row, col=col)
        fig.update_yaxes(title_text=f"<b>{METRIC_Y_LABELS[metric]} ({METRIC_UNITS[metric]})</b>", row=row, col=col)

    style_figure(fig, width=1800, height=1050)
    return fig


def save_figure(fig: go.Figure, base: Path) -> None:
    fig.write_html(base.with_suffix(".html"))
    try:
        fig.write_image(base.with_suffix(".png"), scale=2)
        fig.write_image(base.with_suffix(".pdf"))
    except Exception as exc:
        print(f"Static export skipped for {base.name}: {exc}")


def build_outputs(root: Path, out_dir: Path, n_boot: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = []
    hybrid_summary_rows = []
    all_system_rows = []
    bootstrap_rows = []

    for metric in MAIN_METRICS:
        # Hybrid: main numerical summary.
        hybrid = paired_condition_changes(root, metric, "rtmpose_dlc")
        condition_rows.append(hybrid)
        h_summary = agreement_summary(hybrid)
        h_summary["qualisys_response_beta_vs_imposed"] = fit_beta(root, metric, "qualisys")
        h_summary["system_response_beta_vs_imposed"] = fit_beta(root, metric, "rtmpose_dlc")
        hybrid_summary_rows.append(h_summary)

        boot = bootstrap_paired_change_difference(root, metric, n_boot=n_boot)
        if boot is not None:
            bootstrap_rows.append(boot)

        # All FMC systems: supplementary transparency.
        for tracker in FMC_TRACKERS:
            paired = paired_condition_changes(root, metric, tracker)
            s = agreement_summary(paired)
            s["qualisys_response_beta_vs_imposed"] = fit_beta(root, metric, "qualisys")
            s["system_response_beta_vs_imposed"] = fit_beta(root, metric, tracker)
            s["system_minus_qualisys_beta"] = s["system_response_beta_vs_imposed"] - s["qualisys_response_beta_vs_imposed"]
            all_system_rows.append(s)

    condition_df = pd.concat(condition_rows, ignore_index=True)
    hybrid_summary = pd.DataFrame(hybrid_summary_rows)
    all_systems = pd.DataFrame(all_system_rows)
    bootstrap_df = pd.concat(bootstrap_rows, ignore_index=True) if bootstrap_rows else pd.DataFrame()

    # Main-paper numbers: intentionally compact.
    main_numbers = hybrid_summary[[
        "outcome", "unit", "pearson_r_vs_qualisys_change",
        "mean_signed_change_difference", "direction_agreement_n", "direction_agreement_total",
    ]].copy()

    main_numbers.to_csv(out_dir / "main_hybrid_agreement_summary.csv", index=False)
    condition_df.to_csv(out_dir / "condition_level_hybrid_vs_qualisys.csv", index=False)
    all_systems.to_csv(out_dir / "supplementary_all_systems_agreement.csv", index=False)
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(out_dir / "supplementary_paired_bootstrap_difference_ci.csv", index=False)

    # Separate table of the descriptive response slopes so their meaning stays clear.
    slope_cols = [
        "outcome", "system", "qualisys_response_beta_vs_imposed",
        "system_response_beta_vs_imposed", "system_minus_qualisys_beta",
    ]
    all_systems[slope_cols].to_csv(out_dir / "supplementary_response_slopes_vs_imposed.csv", index=False)

    save_figure(make_main_condition_response_figure(root), out_dir / "main_condition_response_2x3")
    save_figure(make_supplementary_all_systems_figure(root), out_dir / "supplementary_all_systems_condition_response_2x3")

    caption = (
        "Changes from the neutral alignment condition across the imposed prosthetic alignment settings. "
        "Qualisys and FMC-Hybrid are connected through the observed condition means; no linear biological "
        "response model is assumed. Error bars show propagated within-trial SEM for stride-based outcomes. "
        "The dashed 1:1 line is shown only for shank length, for which the imposed pylon-length change provides "
        "a directly comparable physical reference."
    )
    (out_dir / "main_figure_caption_draft.txt").write_text(caption, encoding="utf-8")

    notes = """FINAL ANALYSIS DIRECTION\n\nPrimary question\nDoes FMC-Hybrid reproduce the condition-dependent change from neutral measured by Qualisys?\n\nMain figure\n- x: known imposed alignment condition\n- y: observed change from neutral\n- Qualisys and FMC-Hybrid connected through the actual condition values\n- no fitted response lines for FPA, toe clearance, pelvic obliquity, or ankle-angle outcomes\n- shank length keeps a 1:1 physical reference because imposed and measured quantities are both length changes\n\nMain descriptive numbers\n- Pearson r across the four non-neutral Qualisys and Hybrid changes: pattern agreement only; n=4, descriptive\n- mean signed change difference = mean(Delta Hybrid - Delta Qualisys): systematic difference in measured response\n\nSupplement\n- all-system condition-response figure\n- all-system direct agreement table\n- response slopes versus imposed perturbation, retained as descriptive summaries rather than the primary test\n- paired-stride bootstrap CIs for condition-level Hybrid-minus-Qualisys change differences\n\nImportant\nThe analysis does not assume that biological responses to prosthetic alignment are linear or symmetric.\n"""
    (out_dir / "ANALYSIS_NOTES.txt").write_text(notes, encoding="utf-8")

    print("\nMain Hybrid-vs-Qualisys summary:")
    print(main_numbers.round(3).to_string(index=False))
    print(f"\nSaved final paper-facing outputs under: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final paper-facing Qualisys-vs-FMC response outputs.")
    parser.add_argument("root", type=Path, help="Path to the responsiveness_analysis output folder")
    parser.add_argument("--n-boot", type=int, default=5000, help="Paired-stride bootstrap replicates (default: 5000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.root.resolve()
    build_outputs(root, root / "summary_outputs" / "final_response_analysis", n_boot=args.n_boot)
