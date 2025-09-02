
from pathlib import Path
import sqlite3
from database_schema import METRICS, TRACKERS, TYPE

class DatabaseHelper:
    def __init__(self, path_to_database: Path|str):
         self.path_to_database = path_to_database
         self._connect(path_to_database)
    
    def _connect(self,path_to_database:Path):
        self.con = sqlite3.connect(path_to_database)
        self.con.execute("PRAGMA foreign_keys = ON;")
    
    def _commit(self):
        self.con.commit()
    
    def add_participant(self,
                            participant_code:str,
                            session_date:str):
        self.con.execute("INSERT OR IGNORE INTO participants (participant_code, session_date) VALUES (?,?)",
                         (participant_code, session_date)
        )
        self._commit()

    def add_trial(self,
                  participant_code:str,
                  trial_type:str,
                  trial_number:str,
                  data_root:str|Path,
                  notes:str):
        if not trial_type in TYPE:
            raise ValueError(f"Trial type must be one of {TYPE}, got {trial_type}")
        data_root = str(data_root)
        self.con.execute(
            "INSERT OR IGNORE INTO trials(participant_id, trial_type, trial_number, data_root, notes)"\
            "VALUES ((SELECT participant_id FROM participants WHERE participant_code = ?),?,?,?,?)",
            (participant_code, trial_type, trial_number, data_root, notes)
        )
        self._commit()

    def add_data_to_trial(
            self,
            data_root:str|Path,
            metric_name:str,
            tracker:str,
            folder_name:str
    ):  
        if not metric_name in METRICS:
            raise ValueError(f"Metric type must be one of {METRICS}, got {metric_name}")
        if not tracker in TRACKERS:
            raise ValueError(f"Tracker type must be one of {TRACKERS}, got {tracker}")

        data_root = str(data_root)
        self.con.execute(
            "INSERT OR IGNORE INTO metrics(trial_id, metric_name, tracker, folder_name)"\
            "VALUES ((SELECT trial_id FROM trials WHERE data_root = ?), ?, ?, ?)",
            (data_root, metric_name, tracker, folder_name)
        )

        self._commit()


    def print_table(self, table_name: str, limit: int = 10):
        cur = self.con.cursor()
        cur.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cur.fetchall()
        for row in rows:
            print(row)


    @property
    def df(self):
        return pd.read_sql("SELECT * from view_metrics", self.con)
        



if __name__ == "__main__":
    import pandas as pd
    path_to_database = r"mydb.sqlite"

    db = DatabaseHelper(path_to_database)
    db.track_progress()
    db.add_participant(
        participant_code="JSM",
        session_date= "2025-07-31"
    )

    path_to_recording = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1")

    db.add_trial(
        participant_code="JSM",
        trial_type="treadmill",
        trial_number="1",
        data_root=path_to_recording,
        notes=""
    )

    db.add_data_to_trial(
        data_root=path_to_recording,
        metric_name="rmse",
        tracker="mediapipe",
        folder_name="metrics"
    )
    f = 2




