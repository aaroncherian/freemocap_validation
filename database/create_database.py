import sqlite3
import pandas as pd

con = sqlite3.connect("mydb.sqlite")
con.execute("PRAGMA foreign_keys = ON;")


participants = """
CREATE TABLE IF NOT EXISTS participants (
    participant_id INTEGER PRIMARY KEY,
    participant_code TEXT UNIQUE, 
    session_date TEXT
)
"""

trials = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id INTEGER PRIMARY KEY,
    participant_id INTEGER NOT NULL REFERENCES participants(participant_id) ON DELETE CASCADE,
    trial_type TEXT,
    trial_number TEXT,
    data_root TEXT UNIQUE,
    notes TEXT
    )
    """

metrics = """
CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY,
    trial_id INTEGER NOT NULL REFERENCES trials(trial_id) ON DELETE CASCADE,
    metric_name TEXT,
    tracker TEXT,
    folder_name TEXT,
    UNIQUE(trial_id, metric_name, tracker, folder_name)
    )
"""

view = """
CREATE VIEW IF NOT EXISTS view_metrics AS
SELECT 
    p.participant_code, 
    p.session_date, 
    t.trial_type, 
    t.trial_number, 
    t.data_root,
    m.tracker,  
    m.metric_name, 
    m.folder_name
    
FROM metrics m
JOIN trials t ON m.trial_id = t.trial_id
JOIN participants p ON t.participant_id = p.participant_id
"""


con.execute(participants)
con.execute(trials)
con.execute(metrics)
con.execute(view)

df = pd.read_sql_query("SELECT * FROM trials", con)
f = 2