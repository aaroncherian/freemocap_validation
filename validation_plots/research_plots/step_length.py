import sqlite3
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

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
    AND a.category = "gait_events"
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE "%gait_events"
ORDER BY t.trial_name, a.path
"""

reference_system = "qualisys"
TRACKERS = ["mediapipe"]

path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"].lower()
    sub['tracker'] = row['tracker']
    dfs.append(sub)

df:pd.DataFrame = pd.concat(dfs, ignore_index=True)


for trial in df['trial_name'].unique():
    df_trial = df[df['trial_name'] == trial]

    for tracker in TRACKERS + [reference_system]:
        df_tracker = df_trial[df_trial['tracker'] == tracker]
        f =2



