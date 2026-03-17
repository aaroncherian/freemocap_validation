"""
Generate Bland-Altman summary tables for gait spatiotemporal parameters.

Outputs:
  1. Main table   — pooled across all speeds (Kanko-comparable)
  2. Supplementary — stratified by speed condition
"""

from gait_ba_utils import (
    load_paired_gait_data, ba_stats, compute_icc_2_1,
    ALL_METRICS, TRACKERS, TRACKER_LABELS,
    SPEED_ORDER, SPEED_LABELS,
)
import numpy as np
import pingouin as pg
import pandas as pd

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
paired_df = load_paired_gait_data("validation.db")

# ------------------------------------------------------------------
# Compute stats for a given slice
# ------------------------------------------------------------------
def compute_row(df_slice, metric_key, units, scale, tracker):
    """Compute BA + ICC stats for one metric/tracker combination."""
    sub:pd.DataFrame = df_slice.query("metric == @metric_key and tracker == @tracker").dropna(
        subset=["tracker_value", "reference_value"]
    )
    if sub.empty:
        return None

    

    diffs = sub["ba_diff"].values * scale
    stats = ba_stats(diffs)
    tracker_rows = sub[['participant_code', 'trial_name', 'condition', 'side', 'metric', 'event_index', 'tracker']].copy()
    tracker_rows['value'] = sub['tracker_value']

    ref_rows = sub[['participant_code', 'trial_name', 'condition', 'side', 'metric', 'event_index']].copy()
    ref_rows['tracker'] = 'qualisys'
    ref_rows['value'] = sub['reference_value']

    long_df = pd.concat([tracker_rows, ref_rows], ignore_index=True)

    long_df['target'] = (
        long_df['trial_name'] + '|' +
        long_df['condition'] + '|' +
        long_df['side'] + '|' +
        long_df['event_index'].astype(str)
    )
    icc_overall = pg.intraclass_corr(
        data = long_df,
        targets = "target",
        raters = "tracker",
        ratings = "value"
    )
    icc_overall = icc_overall.set_index("Type")
    icc_result = icc_overall.loc["ICC2", ["ICC", "CI95%"]]


    return dict(
        n=stats["n"],
        bias=stats["bias"],
        sd=stats["sd"],
        loa_lower=stats["loa_lower"],
        loa_upper=stats["loa_upper"],
        icc=icc_result["ICC"],
        icc_lb=icc_result["CI95%"][0],
        icc_ub=icc_result["CI95%"][1],
        units=units,
    )


# ------------------------------------------------------------------
# 1. Pooled table (all speeds)
# ------------------------------------------------------------------
print("=" * 80)
print("MAIN TABLE — Pooled across all speeds")
print("=" * 80)

pooled_rows = []
for m_key, m_label, m_units, m_scale in ALL_METRICS:
    for tracker in TRACKERS:
        row = compute_row(paired_df, m_key, m_units, m_scale, tracker)
        if row is None:
            continue
        row["metric"] = m_label
        row["tracker"] = TRACKER_LABELS[tracker]
        pooled_rows.append(row)

# Print readable summary
fmt_header = f"{'Metric':<20s} {'Tracker':<12s} {'N':>5s}  {'Bias':>8s} {'SD':>8s} {'LoA Low':>8s} {'LoA Up':>8s}  {'ICC':>6s} ({'LB':>5s}, {'UB':>5s})"
print(fmt_header)
print("-" * len(fmt_header))
for r in pooled_rows:
    print(
        f"{r['metric']:<20s} {r['tracker']:<12s} {r['n']:>5d}  "
        f"{r['bias']:>+8.2f} {r['sd']:>8.2f} {r['loa_lower']:>+8.2f} {r['loa_upper']:>+8.2f}  "
        f"{r['icc']:>6.3f} ({r['icc_lb']:>5.3f}, {r['icc_ub']:>5.3f})  {r['units']}"
    )


# ------------------------------------------------------------------
# 2. Supplementary table — stratified by speed
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUPPLEMENTARY TABLE — Stratified by speed")
print("=" * 80)

speed_rows = []
for m_key, m_label, m_units, m_scale in ALL_METRICS:
    for speed in SPEED_ORDER:
        df_speed = paired_df[paired_df["condition"] == speed]
        for tracker in TRACKERS:
            row = compute_row(df_speed, m_key, m_units, m_scale, tracker)
            if row is None:
                continue
            row["metric"] = m_label
            row["speed"] = SPEED_LABELS[speed]
            row["tracker"] = TRACKER_LABELS[tracker]
            speed_rows.append(row)

fmt_header2 = f"{'Metric':<20s} {'Speed':<8s} {'Tracker':<12s} {'N':>5s}  {'Bias':>8s} {'SD':>8s} {'LoA Low':>8s} {'LoA Up':>8s}  {'ICC':>6s} ({'LB':>5s}, {'UB':>5s})"
print(fmt_header2)
print("-" * len(fmt_header2))
for r in speed_rows:
    print(
        f"{r['metric']:<20s} {r['speed']:<8s} {r['tracker']:<12s} {r['n']:>5d}  "
        f"{r['bias']:>+8.2f} {r['sd']:>8.2f} {r['loa_lower']:>+8.2f} {r['loa_upper']:>+8.2f}  "
        f"{r['icc']:>6.3f} ({r['icc_lb']:>5.3f}, {r['icc_ub']:>5.3f})  {r['units']}"
    )


# ------------------------------------------------------------------
# 3. Export as Typst tables
# ------------------------------------------------------------------
from pathlib import Path

save_root = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\gait\tables")
save_root.mkdir(exist_ok=True, parents=True)


def fmt_val(v, decimals=2):
    """Format a float with sign for bias, or plain for others."""
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def fmt_bias(v, decimals=2):
    if np.isnan(v):
        return "—"
    return f"{v:+.{decimals}f}"


def fmt_icc(v, decimals=3):
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


# --- Main table ---
typst_lines = []
typst_lines.append('#figure(')
typst_lines.append('  table(')
typst_lines.append('    columns: (auto, auto, auto, auto, auto, auto, auto, auto),')
typst_lines.append('    align: (left, left, right, right, right, right, right, right),')
typst_lines.append('    stroke: none,')
typst_lines.append('    table.hline(stroke: 1pt),')
typst_lines.append('    table.header(')
typst_lines.append('      [*Metric*], [*Tracker*], [*N*], [*Bias*], [*SD*], [*Lower LoA*], [*Upper LoA*], [*ICC (95% CI)*],')
typst_lines.append('    ),')
typst_lines.append('    table.hline(stroke: 0.5pt),')

prev_metric = None
for r in pooled_rows:
    # Add thin separator between metric groups
    if prev_metric is not None and r["metric"] != prev_metric:
        typst_lines.append('    table.hline(stroke: 0.3pt),')
    prev_metric = r["metric"]

    metric_cell = f'[{r["metric"]} ({r["units"]})]' if r["tracker"] == "MediaPipe" else '[]'
    icc_str = f'{fmt_icc(r["icc"])} ({fmt_icc(r["icc_lb"])}, {fmt_icc(r["icc_ub"])})'

    typst_lines.append(
        f'    {metric_cell}, [{r["tracker"]}], [{r["n"]}], '
        f'[{fmt_bias(r["bias"])}], [{fmt_val(r["sd"])}], '
        f'[{fmt_bias(r["loa_lower"])}], [{fmt_bias(r["loa_upper"])}], '
        f'[{icc_str}],'
    )

typst_lines.append('    table.hline(stroke: 1pt),')
typst_lines.append('  ),')
typst_lines.append('  caption: [Bland-Altman agreement statistics for spatiotemporal gait parameters across all walking speeds. Bias and limits of agreement (LoA) are reported in the native units of each metric. ICC = intraclass correlation coefficient (2,1) with 95% confidence interval.],')
typst_lines.append(') <tbl-ba-gait-pooled>')

typst_main = "\n".join(typst_lines)
main_path = save_root / "ba_gait_pooled.typ"
main_path.write_text(typst_main, encoding="utf-8")
print(f"\nMain table written to: {main_path}")

# --- Supplementary table ---
typst_lines_s = []
typst_lines_s.append('#figure(')
typst_lines_s.append('  table(')
typst_lines_s.append('    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),')
typst_lines_s.append('    align: (left, left, left, right, right, right, right, right, right),')
typst_lines_s.append('    stroke: none,')
typst_lines_s.append('    table.hline(stroke: 1pt),')
typst_lines_s.append('    table.header(')
typst_lines_s.append('      [*Metric*], [*Speed*], [*Tracker*], [*N*], [*Bias*], [*SD*], [*Lower LoA*], [*Upper LoA*], [*ICC (95% CI)*],')
typst_lines_s.append('    ),')
typst_lines_s.append('    table.hline(stroke: 0.5pt),')

prev_metric_s = None
prev_speed_s = None
for r in speed_rows:
    if prev_metric_s is not None and r["metric"] != prev_metric_s:
        typst_lines_s.append('    table.hline(stroke: 0.5pt),')
    elif prev_speed_s is not None and r["speed"] != prev_speed_s:
        typst_lines_s.append('    table.hline(stroke: 0.3pt),')

    # Show metric label only on first row of each metric group
    metric_cell = f'[{r["metric"]} ({r["units"]})]' if r["metric"] != prev_metric_s else '[]'
    # Show speed label only on first row of each speed group within a metric
    speed_cell = f'[{r["speed"]}]' if r["speed"] != prev_speed_s or r["metric"] != prev_metric_s else '[]'

    prev_metric_s = r["metric"]
    prev_speed_s = r["speed"]

    icc_str = f'{fmt_icc(r["icc"])} ({fmt_icc(r["icc_lb"])}, {fmt_icc(r["icc_ub"])})'

    typst_lines_s.append(
        f'    {metric_cell}, {speed_cell}, [{r["tracker"]}], [{r["n"]}], '
        f'[{fmt_bias(r["bias"])}], [{fmt_val(r["sd"])}], '
        f'[{fmt_bias(r["loa_lower"])}], [{fmt_bias(r["loa_upper"])}], '
        f'[{icc_str}],'
    )

typst_lines_s.append('    table.hline(stroke: 1pt),')
typst_lines_s.append('  ),')
typst_lines_s.append('  caption: [Bland-Altman agreement statistics for spatiotemporal gait parameters stratified by walking speed. Bias and limits of agreement (LoA) are reported in the native units of each metric. ICC = intraclass correlation coefficient (2,1) with 95% confidence interval.],')
typst_lines_s.append(') <tbl-ba-gait-by-speed>')

typst_supp = "\n".join(typst_lines_s)
supp_path = save_root / "ba_gait_by_speed.typ"
supp_path.write_text(typst_supp, encoding="utf-8")
print(f"Supplementary table written to: {supp_path}")
