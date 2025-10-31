import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
import plotly.express as px

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
    AND a.category = "trajectories_per_stride"
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE '%summary_stats'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)


dfs = []

for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    condition = row.get("condition") or ""  # handle None/empty
    participant = row["participant_code"]
    trial = row["trial_name"]

    # Load file — autodetect type
    sub_df = pd.read_csv(path)

    # Add metadata columns
    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker
    sub_df["condition"] = condition if condition else "none"

    dfs.append(sub_df)

# Concatenate all into one tidy DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

left = combined_df[combined_df["marker"] == "left_ankle"].copy()

# 2) Ensure condition exists
left_mean = (
    combined_df
    .query("marker == 'left_ankle' and stat == 'mean'")
    .copy()
)

fig = px.line(
    left_mean,
    x="percent_gait_cycle",
    y="value",
    color="condition",        # color by condition
    line_dash="tracker",      # dash by tracker (mediapipe vs qualisys)
    facet_col="axis",         # x / y / z panels
    facet_col_wrap=3,
    labels={
        "percent_gait_cycle":"% gait cycle",
        "value":"position (units)",
        "condition":"Condition",
        "tracker":"Tracker",
        "axis":"Axis"
    },
    title="Mean Left Ankle Trajectory per Condition"
)
fig.update_layout(legend_title_text="")
fig.show()

f = 2
# table_sql = cursor.fetchone()
# print(table_sql)

# dfc = pd.read_sql_query("SELECT * FROM v_completeness_condition", conn)
# pivot = dfc.pivot_table(index=["trial_name","condition"], columns="tracker", values="pct_present")
# plt.figure(figsize=(10,6))
# plt.imshow(pivot.fillna(0).values, aspect="auto")
# plt.yticks(range(len(pivot.index)), [f"{t} | {c}" for t,c in pivot.index], fontsize=8)
# plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
# plt.colorbar(label="% present")
# plt.title("Completeness heatmap (% present) per trial/condition × tracker")
# plt.tight_layout()
# plt.show()