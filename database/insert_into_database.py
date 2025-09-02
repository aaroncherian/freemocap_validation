import sqlite3
from pathlib import Path
import pandas as pd

con = sqlite3.connect("mydb.sqlite")
con.execute("PRAGMA foreign_keys = ON;")

con.execute(
    "INSERT OR IGNORE INTO participants (participant_code, session_date) VALUES (?, ?)",
    ("JSM", "2025-07-31")
)

con.commit()
# participant_id = con.execute(
#     "SELECT participant_id FROM participants WHERE participant_code = ?",
#     ("JSM",)
# ).fetchone()[0]

# print("participant_id:", participant_id)

# data_root = r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1"
# con.execute(
#     "INSERT OR IGNORE INTO trials(participant_id, trial_type, data_root, notes)" \
#     "VALUES ((SELECT participant_id FROM participants WHERE participant_code = ?),?,?,?)",
#     ("JSM", "treadmill", data_root, "")
# )
# con.commit()

# trial_type = con.execute(
#     "SELECT trial_type FROM trials where participant_id = (SELECT participant_id from participants WHERE participant_code = ?)",
#     ("JSM",)
# ).fetchone()[0]

# con.execute(
#     "INSERT OR IGNORE INTO metrics(trial_id, metric_name, tracker, folder_name)"\
#     "VALUES ((SELECT trial_id from trials where participant_id = (SELECT participant_id from participants WHERE)))"
# )

# print(trial_type)