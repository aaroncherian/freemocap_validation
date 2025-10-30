import pandas as pd
import sqlite3
from matplotlib import pyplot as plt

conn = sqlite3.connect("validation.db")

query = """
SELECT t.participant_code, 
        t.trial_name,
        a.path,
        a.component_name
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "treadmill"
    AND a.condition = "speed_0_5"
    AND a.category = "trajectories_per_stride"
    AND a.tracker = "mediapipe"
    AND a.file_exists = 1
ORDER BY t.trial_name, a.path;
"""
df = pd.read_sql_query(query, conn)
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