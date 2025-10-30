import pandas as pd
import sqlite3
from matplotlib import pyplot as plt

conn = sqlite3.connect("validation.db")

query = """
SELECT a.id, a.tracker, a.component_name, a.category, a.condition,
       a.path, a.file_exists, a.mtime_utc
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_name='2025-07-31_16-00-42_GMT-4_jsm_nih_trial_1'
  AND a.category='path_length_analysis';
"""
cursor = conn.execute(query)
table_sql = cursor.fetchone()
print(table_sql)

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