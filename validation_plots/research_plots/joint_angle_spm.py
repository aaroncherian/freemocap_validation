# =========================
#  SPM (paired t-test) block
# =========================
import spm1d
from dataclasses import dataclass
import re
import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


TRACKERS = ["mediapipe", "qualisys", "rtmpose", "vitpose"]  # add more if present

conn = sqlite3.connect("validation.db")
query = """
SELECT t.participant_code,
       t.trial_name,
       a.path,
       a.component_name,
       a.condition,
       a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "treadmill"
  AND a.category = "joint_angles_per_stride"
  AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
  AND a.file_exists = 1
  AND a.condition LIKE "speed_%"
  AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""
reference_system = "qualisys"

path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"]
    sub["tracker"] = (row["tracker"] or "").lower()
    sub["condition"] = row["condition"] or "none"
    dfs.append(sub)

combined_df = pd.concat(dfs, ignore_index=True)

# ------------------------
# 2) Normalize + keep only major motions
# ------------------------
for col in ["joint", "side", "tracker", "stat", "component"]:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].astype(str).str.lower()

combined_df["component"] = combined_df["component"].replace(
    {"inversion_eversion": "inv_ev"}
)

MAJOR = {
    ("hip", "flex_ext"),
    ("knee", "flex_ext"),
    ("ankle", "dorsi_plantar"),
}
combined_df = combined_df[
    combined_df.apply(lambda r: (r["joint"], r["component"]) in MAJOR, axis=1)
].copy()

JOINT_ORDER = ["hip", "knee", "ankle"]

COMP_LABEL = {
    "flex_ext": "Flex/Ext",
    "dorsi_plantar": "Dorsi/Plantar",
}

# ------------------------
# 3) Speed parsing / ordering
# ------------------------
def speed_key(cond: str) -> float:
    m = re.search(r"speed_(\d+)[_\.](\d+)", str(cond))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"speed_(\d+)", str(cond))
    if m2:
        return float(m2.group(1))
    return float("inf")

def speed_label(cond: str) -> str:
    k = speed_key(cond)
    return "?" if not np.isfinite(k) else f"{k:g} m/s"

SPEEDS = sorted(combined_df["condition"].unique().tolist(), key=speed_key)

# ------------------------
# 4) Collapse sides (within-trial L/R mean), then summarize across trials
# ------------------------
df_means = combined_df[combined_df["stat"] == "mean"].copy()

df_trial_lr_mean = (
    df_means
    .groupby(
        ["condition", "tracker", "participant_code", "trial_name",
         "joint", "component", "percent_gait_cycle"],
        as_index=False
    )
    .agg(trial_mean_angle=("value", "mean"))
)

REFERENCE_SYSTEM = "qualisys"
ALPHA = 0.05
TWO_TAILED = True
Q_EXPECTED = 100  # set to None if you want to infer from data

@dataclass
class SpmResultRow:
    condition: str
    joint: str
    component: str
    tracker: str
    n_trials: int
    two_tailed: bool
    alpha: float
    t_star: float
    n_clusters: int
    # cluster fields
    cluster_idx: int
    p_value: float
    start_node: int
    end_node: int
    start_gait_pct: float
    end_gait_pct: float
    extent_nodes: int


def _build_trial_waveform_matrix(
    df_trial_lr_mean: pd.DataFrame,
    condition: str,
    joint: str,
    component: str,
    tracker: str,
    reference: str = "qualisys",
    q_expected: int | None = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Y_ref: (J, Q)
      Y_trk: (J, Q)
      x:     (Q,) gait percent values (sorted)
    Builds paired waveforms by trial (participant_code + trial_name), requiring both systems present.
    Drops trials that don't have a full set of gait nodes.
    """
    sub = df_trial_lr_mean[
        (df_trial_lr_mean["condition"] == condition) &
        (df_trial_lr_mean["joint"] == joint) &
        (df_trial_lr_mean["component"] == component) &
        (df_trial_lr_mean["tracker"].isin([reference, tracker]))
    ].copy()

    if sub.empty:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))

    # Pivot to get ref + tracker per (participant, trial, gait%)
    piv = (
        sub.pivot_table(
            index=["participant_code", "trial_name", "percent_gait_cycle"],
            columns="tracker",
            values="trial_mean_angle",
            aggfunc="first",
        )
        .reset_index()
    )

    # Keep only rows where BOTH ref and tracker exist
    if reference not in piv.columns or tracker not in piv.columns:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))
    piv = piv.dropna(subset=[reference, tracker])

    # Now build one waveform per (participant, trial)
    # We'll pivot again: index=(participant, trial), columns=percent, values=angle
    ref_wide = piv.pivot_table(
        index=["participant_code", "trial_name"],
        columns="percent_gait_cycle",
        values=reference,
        aggfunc="first",
    )
    trk_wide = piv.pivot_table(
        index=["participant_code", "trial_name"],
        columns="percent_gait_cycle",
        values=tracker,
        aggfunc="first",
    )

    # Keep only trials present in both
    common_trials = ref_wide.index.intersection(trk_wide.index)
    ref_wide = ref_wide.loc[common_trials]
    trk_wide = trk_wide.loc[common_trials]

    # Ensure consistent x-axis and order
    x = np.array(sorted(ref_wide.columns.astype(float)))
    ref_wide = ref_wide.reindex(columns=x)
    trk_wide = trk_wide.reindex(columns=x)

    # Drop trials with missing nodes
    good = (~ref_wide.isna().any(axis=1)) & (~trk_wide.isna().any(axis=1))
    ref_wide = ref_wide.loc[good]
    trk_wide = trk_wide.loc[good]

    # Optionally enforce Q length (e.g., 101 nodes)
    if q_expected is not None:
        if ref_wide.shape[1] != q_expected:
            # If you expect 101 but got fewer/more, bail so you notice
            return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))

    Y_ref = ref_wide.to_numpy(dtype=float)
    Y_trk = trk_wide.to_numpy(dtype=float)

    return Y_ref, Y_trk, x


def run_spm_paired_ttests(
    df_trial_lr_mean: pd.DataFrame,
    trackers: list[str],
    reference: str = "qualisys",
    alpha: float = 0.05,
    two_tailed: bool = True,
    q_expected: int | None = 101,
) -> pd.DataFrame:
    """
    Runs SPM{t} paired tests for each condition × joint × component × tracker vs reference.
    Returns a long results table including cluster intervals and p-values.
    """
    results: list[SpmResultRow] = []
    curves = []

    conditions = sorted(df_trial_lr_mean["condition"].unique().tolist(), key=speed_key)

    # Your major motions (match your plotting logic)
    targets = [
        ("hip", "flex_ext"),
        ("knee", "flex_ext"),
        ("ankle", "dorsi_plantar"),
    ]

    for condition in conditions:
        for joint, component in targets:
            for tracker in trackers:
                if tracker == reference:
                    continue

                Y_ref, Y_trk, x = _build_trial_waveform_matrix(
                    df_trial_lr_mean=df_trial_lr_mean,
                    condition=condition,
                    joint=joint,
                    component=component,
                    tracker=tracker,
                    reference=reference,
                    q_expected=q_expected,
                )

                J, Q = Y_ref.shape if Y_ref.size else (0, 0)
                if J < 3 or Q == 0:
                    # Too few paired trials to do meaningful inference
                    continue

                # --- SPM paired t-test ---
                t = spm1d.stats.ttest_paired(Y_ref, Y_trk)
                ti = t.inference(alpha=alpha, two_tailed=two_tailed)

                t_star = float(ti.zstar)  # critical threshold
                clusters = getattr(ti, "clusters", []) or []

                spm_t = np.asarray(t.z, dtype=float)  # length Q

                curves.append(pd.DataFrame({
                    "condition": condition,
                    "joint": joint,
                    "component": component,
                    "tracker": tracker,
                    "reference": reference,
                    "alpha": alpha,
                    "two_tailed": two_tailed,
                    "n_trials": J,
                    "t_star": t_star,
                    "percent_gait_cycle": x.astype(float),
                    "spm_t": spm_t,
                }))

                if len(clusters) == 0:
                    # Record a "no clusters" row for bookkeeping
                    results.append(
                        SpmResultRow(
                            condition=condition, joint=joint, component=component, tracker=tracker,
                            n_trials=J, two_tailed=two_tailed, alpha=alpha,
                            t_star=t_star, n_clusters=0,
                            cluster_idx=-1, p_value=np.nan,
                            start_node=-1, end_node=-1,
                            start_gait_pct=np.nan, end_gait_pct=np.nan,
                            extent_nodes=0,
                        )
                    )
                    continue

                # Each cluster has node indices and p-value
                # ti.clusters are Cluster objects; indices live in cluster.endpoints (in node units)
                for k, c in enumerate(clusters):
                    # endpoints are in "node space" (0..Q-1), inclusive bounds
                    # spm1d cluster endpoints are floats sometimes if interp=True; we'll coerce safely
                    a, b = c.endpoints  # start, end (node coordinates)
                    start_node = int(np.floor(a))
                    end_node = int(np.ceil(b))
                    start_node = max(0, min(Q - 1, start_node))
                    end_node = max(0, min(Q - 1, end_node))

                    results.append(
                        SpmResultRow(
                            condition=condition, joint=joint, component=component, tracker=tracker,
                            n_trials=J, two_tailed=two_tailed, alpha=alpha,
                            t_star=t_star, n_clusters=len(clusters),
                            cluster_idx=k, p_value=float(c.P),
                            start_node=start_node, end_node=end_node,
                            start_gait_pct=float(x[start_node]), end_gait_pct=float(x[end_node]),
                            extent_nodes=int(end_node - start_node + 1),
                        )
                    )

                # Quick console summary
                print(f"\nSPM paired t-test: {condition} | {joint} {component} | {tracker} vs {reference}")
                print(f"  n_trials = {J}, Q = {Q}, t* = {t_star:.3f}, clusters = {len(clusters)}")
                for k, c in enumerate(clusters):
                    a, b = c.endpoints
                    sn = int(np.floor(a)); en = int(np.ceil(b))
                    sn = max(0, min(Q - 1, sn)); en = max(0, min(Q - 1, en))
                    print(f"   - cluster {k}: {x[sn]:.1f}%–{x[en]:.1f}% GC, p={c.P:.4f}")

    out_clusters = pd.DataFrame([r.__dict__ for r in results])
    out_curves = pd.concat(curves, ignore_index=True) if len(curves) else pd.DataFrame()
    return  out_clusters, out_curves


# ----------------------------
# Run SPM for your comparison
# ----------------------------
SPM_TRACKERS = ["mediapipe", "rtmpose", "vitpose"]  # add more if present

print("Unique trackers in df_trial_lr_mean:", sorted(df_trial_lr_mean["tracker"].unique()))
print("Unique joints:", sorted(df_trial_lr_mean["joint"].unique()))
print("Unique components:", sorted(df_trial_lr_mean["component"].unique()))

spm_clusters, spm_curves = run_spm_paired_ttests(
    df_trial_lr_mean=df_trial_lr_mean,
    trackers=SPM_TRACKERS,
    reference=REFERENCE_SYSTEM,
    alpha=ALPHA,
    two_tailed=TWO_TAILED,
    q_expected=Q_EXPECTED,
)

print("\nSPM clusters (head):")
print(spm_clusters.head(20))

root_dir = Path(r"D:\validation\joint_angles")
root_dir.mkdir(exist_ok=True, parents=True)
# Save a tidy table (easy to join back into reports)
spm_clusters.to_csv(root_dir / "spm_paired_ttest_clusters.csv", index=False)
spm_curves.to_csv(root_dir / "spm_paired_ttest_curves.csv", index=False)

print("Saved clusters:", root_dir / "spm_paired_ttest_clusters.csv")
print("Saved curves:",   root_dir / "spm_paired_ttest_curves.csv")


