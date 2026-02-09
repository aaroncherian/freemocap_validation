import re
import sqlite3
import numpy as np
import pandas as pd

# ------------------------
# Config
# ------------------------
TRACKERS = ["mediapipe", "rtmpose", "qualisys"]
REFERENCE_SYSTEM = "qualisys"

JOINT_ORDER = ["hip", "knee", "ankle", "foot_index"]
AXES = ["x", "y", "z"]
AXIS_PRETTY = {"x": "ML", "y": "AP", "z": "Vertical"}

DB_PATH = "validation.db"

# Where to save slide-ready CSVs (edit)
OUT_DIR = r"D:\validation"

# ------------------------
# Helpers
# ------------------------
def speed_key(cond: str) -> float:
    m = re.search(r"speed_(\d+)[_\.](\d+)", str(cond))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"speed_(\d+)", str(cond))
    if m2:
        return float(m2.group(1))
    return float("inf")

def parse_speed_float(cond: str) -> float:
    s = (
        str(cond)
        .replace("speed_", "")
        .replace("_", ".")
    )
    try:
        return float(s)
    except Exception:
        return np.nan

def calculate_rmse(tracker: pd.Series, reference: pd.Series) -> float:
    t = tracker.to_numpy(dtype=float)
    r = reference.to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(r)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((t[m] - r[m]) ** 2)))

def trajectory_rmse_slide_table(rmse_df: pd.DataFrame, joint_name: str, axis: str) -> pd.DataFrame:
    """
    Returns a slide-ready table:
      rows = Tracker
      cols = speed (m/s)
      values = RMSE (mm)
    """
    out = (
        rmse_df[(rmse_df["joint"] == joint_name) & (rmse_df["axis"] == axis)]
        .pivot_table(index="tracker", columns="speed", values="rmse", aggfunc="first")
        .sort_index(axis=1)
    )

    out = out.round(1)
    out.index = out.index.str.capitalize()
    out.columns = [f"{c:g}" for c in out.columns]  # 0.5, 1.0, etc.

    # Put tracker names into first column header like your joint-angle tables
    out.insert(0, "Speed (m/s)", out.index)
    out = out.set_index("Speed (m/s)")
    return out

# ------------------------
# 1) Load paths from SQLite
# ------------------------
conn = sqlite3.connect(DB_PATH)
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
  AND a.category = "trajectories_per_stride"
  AND a.tracker IN ("mediapipe", "rtmpose", "qualisys")
  AND a.file_exists = 1
  AND a.condition LIKE "speed_%"
  AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"]
    sub["tracker"] = (row["tracker"] or "").lower()
    sub["condition"] = row["condition"] or "none"
    dfs.append(sub)

if not dfs:
    raise RuntimeError("No trajectory summary_stats CSVs found from the query.")

combined_df = pd.concat(dfs, ignore_index=True)

# ------------------------
# 2) Normalize columns: marker -> side/joint; mirror ML for left
# Expected columns include: marker, axis, stat, value, percent_gait_cycle
# ------------------------
# normalize marker strings first
m = combined_df["marker"].astype(str).str.strip().str.lower()

combined_df["side"] = np.select(
    [m.str.startswith("left_"), m.str.startswith("right_")],
    ["left", "right"],
    default="unknown"
)

combined_df["joint"] = m.str.replace(r"^(left_|right_)", "", regex=True)

# filter joints/axes (optional but usually what you want)
combined_df["axis"] = combined_df["axis"].astype(str).str.lower()
combined_df["tracker"] = combined_df["tracker"].astype(str).str.lower()
combined_df["condition"] = combined_df["condition"].astype(str)

combined_df = combined_df[
    combined_df["joint"].isin(JOINT_ORDER) &
    combined_df["axis"].isin(AXES)
].copy()

# mirror ML for left (x axis, stat=mean only), keep original for everything else
combined_df["value_mirrored"] = combined_df["value"].astype(float)

ml_left_mean_mask = (
    (combined_df["axis"] == "x") &
    (combined_df["side"] == "left") &
    (combined_df["stat"].astype(str).str.lower() == "mean")
)
combined_df.loc[ml_left_mean_mask, "value_mirrored"] *= -1

# ------------------------
# 3) Collapse sides within trial (L/R mean), then build per-tracker mean trajectories
# ------------------------
df_means = combined_df[combined_df["stat"].astype(str).str.lower() == "mean"].copy()

df_trial_lr_mean = (
    df_means
    .groupby(
        ["condition", "tracker", "participant_code", "trial_name",
         "joint", "axis", "percent_gait_cycle"],
        as_index=False
    )
    .agg(trial_mean_value=("value_mirrored", "mean"))
)

print(
    df_trial_lr_mean.groupby(["condition", "tracker"])["trial_name"]
    .nunique()
    .unstack(fill_value=0)
)

traj_summary = (
    df_trial_lr_mean
    .groupby(["condition", "tracker", "joint", "axis", "percent_gait_cycle"], as_index=False)
    .agg(
        mean_value=("trial_mean_value", "mean"),
        std_value=("trial_mean_value", "std"),
        n_trials=("trial_name", "nunique"),
    )
)

# ------------------------
# 4) Pivot wide by tracker, compute RMSE vs reference (per speed/joint/axis)
# ------------------------
id_cols = ["condition", "joint", "axis", "percent_gait_cycle"]

wide = (
    traj_summary.pivot_table(
        index=id_cols,
        columns="tracker",
        values="mean_value",
        aggfunc="first",
    )
    .reset_index()
)
wide.columns.name = None

if REFERENCE_SYSTEM not in wide.columns:
    raise RuntimeError(
        f"Reference system '{REFERENCE_SYSTEM}' not present in wide table columns: {list(wide.columns)}"
    )

wide = wide.rename(columns={REFERENCE_SYSTEM: "reference_system"})
tracker_cols_present = [t for t in TRACKERS if t in wide.columns and t != REFERENCE_SYSTEM]

paired_df = wide.melt(
    id_vars=id_cols + ["reference_system"],
    value_vars=tracker_cols_present,
    var_name="tracker",
    value_name="tracker_value"
)

rmse_table = (
    paired_df
    .groupby(["condition", "joint", "axis", "tracker"], as_index=False)
    .apply(lambda g: calculate_rmse(g["tracker_value"], g["reference_system"]))
    .rename(columns={None: "rmse"})
)

rmse_table["speed"] = rmse_table["condition"].map(parse_speed_float)

rmse_table = rmse_table.sort_values(
    by=["joint", "axis", "speed", "tracker"],
    key=lambda s: s if s.name != "tracker" else s
)

# ------------------------
# 5) Print + export slide-ready tables (one per joint x axis)
# ------------------------
for joint in JOINT_ORDER:
    for axis in AXES:
        tbl = trajectory_rmse_slide_table(rmse_table, joint, axis)

        print(f"\n{joint.title().replace('_',' ')} trajectory RMSE ({AXIS_PRETTY.get(axis, axis).upper()}) [mm]")
        print(tbl)

        out_path = fr"{OUT_DIR}\{joint}_{axis}_trajectory_rmse_table.csv"
        tbl.to_csv(out_path)

print("\nSaved trajectory RMSE tables to:", OUT_DIR)
