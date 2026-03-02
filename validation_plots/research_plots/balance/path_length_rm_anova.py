import pandas as pd
import sqlite3
from pathlib import Path
import pingouin as pg
from pathlib import Path

conn = sqlite3.connect("validation.db")

# root_path = Path(r"D:\validation\balance")

root_path = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\balance")
root_path.mkdir(exist_ok=True, parents=True)

reference = "qualisys"
freemocap_trackers = ["mediapipe", "rtmpose", "vitpose"]

trackers = [reference] + freemocap_trackers

placeholders = ",".join(["?"] * len(trackers))

query = f"""
SELECT t.participant_code, 
        t.trial_name,
        a.path,
        a.component_name,
        a.condition,
        a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
    AND a.category = "com_analysis"
    AND a.tracker IN ({placeholders})
    AND a.file_exists = 1
    AND a.component_name LIKE '%path_length_com'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn, params=trackers)

dfs = []
for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_json(path)
    sub_df = sub_df.rename(columns={
        "Frame Intervals": "frame_interval",
        "Path Lengths:": "path_length"
    }).reset_index()
    sub_df = sub_df.rename(columns={"index": "condition"})

    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker
    dfs.append(sub_df)

combined_df = pd.concat(dfs, ignore_index=True)

wide_df = combined_df.pivot_table(
    index = ['condition', 'participant_code', 'trial_name'],
    columns = 'tracker',
    values = 'path_length'
).reset_index()

mean_df = (
    wide_df
    .groupby(['participant_code', 'condition'], as_index=False)[trackers]
    .mean()
)

mean_df_long = pd.melt(mean_df,
                       id_vars = ['participant_code', 'condition'],
                       value_vars=trackers,
                       var_name = 'tracker',
                       value_name = 'mean_path_length')


for tracker in freemocap_trackers:
    sub_df = mean_df_long[mean_df_long['tracker'].isin([reference, tracker])]
    result = pg.rm_anova(
        data = sub_df,
        dv = "mean_path_length",
        within = ["condition", "tracker"],
        subject = "participant_code"
    )

    print(tracker)
    print(result)
f = 2