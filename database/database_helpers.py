
from pathlib import Path
import sqlite3
from database.database_schema import METRICS, TRACKERS, TYPE
import yaml

class DatabaseHelper:
    def __init__(self, path_to_database: Path|str):
         self.path_to_database = path_to_database
         self._connect(path_to_database)
    
    def _connect(self,path_to_database:Path):
        self.con = sqlite3.connect(path_to_database)
        self.con.execute("PRAGMA foreign_keys = ON;")
        self.cur = self.con.cursor()
    
    def _commit(self):
        self.con.commit()
    
    def add_participant(self,
                            participant_code:str,
                            session_date:str,
                            data_root:str|Path,
                            notes: str = ""):
        data_root = str(data_root)
        self.con.execute("INSERT OR IGNORE INTO participants (participant_code, session_date, data_root, notes) VALUES (?,?,?,?)",
                         (participant_code, session_date, data_root, notes)
        )
        self._commit()

    def add_trial(self,
                  participant_code:str,
                  trial_type:str,
                  trial_number:str,
                  trial_name:str,
                  notes:str):
        if not trial_type in TYPE:
            raise ValueError(f"Trial type must be one of {TYPE}, got {trial_type}")

        self.con.execute(
            "INSERT OR IGNORE INTO trials(participant_id, trial_type, trial_number, trial_name, notes)"\
            "VALUES ((SELECT participant_id FROM participants WHERE participant_code = ?),?,?,?,?)",
            (participant_code, trial_type, trial_number, trial_name, notes)
        )
        self._commit()

    def add_data_to_trial(
            self,
            trial_name:str,
            metric_name:str,
            tracker:str,
            folder_name:str,
            notes: str = ""
    ):  
        if not metric_name in METRICS:
            raise ValueError(f"Metric type must be one of {METRICS}, got {metric_name}")
        if not tracker in TRACKERS:
            raise ValueError(f"Tracker type must be one of {TRACKERS}, got {tracker}")

        self.con.execute(
            "INSERT OR IGNORE INTO metrics(trial_id, metric_name, tracker, folder_name, notes)"\
            "VALUES ((SELECT trial_id FROM trials WHERE trial_name = ?), ?, ?, ?, ?)",
            (trial_name, metric_name, tracker, folder_name, notes)
        )

        self._commit()


    def print_table(self, table_name: str, limit: int = 10):
        cur = self.con.cursor()
        cur.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cur.fetchall()
        for row in rows:
            print(row)

    def load_from_yaml(self, path_to_yaml: Path|str):
        path_to_yaml = Path(path_to_yaml)

        with open(path_to_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        participant_code = data['participant_code']
        trials = data['trials']
        self.add_participant(participant_code=participant_code,
                            session_date= data['session_date'],
                            data_root=data['data_root'],
                            notes = data['notes'])

        for trial in trials:
            trial_name = trial['trial_name']
            self.add_trial(
                participant_code=participant_code,
                trial_type=trial['trial_type'],
                trial_number=str(trial['trial_number']),
                trial_name=trial_name,
                notes=trial['notes']
            )

            for metric in trial['metrics']:
                self.add_data_to_trial(
                    trial_name=trial_name,
                    metric_name=metric['metric_name'],
                    tracker=metric['tracker'],
                    folder_name=metric['folder_name'],
                    notes=metric['notes']
            )

    @property
    def sql_view(self):
       return  """
        SELECT
            participant_code,
            session_date,
            trial_type,
            trial_number,
            data_root,
            trial_name,
            metric_name,
            tracker,
            folder_name
        FROM view_metrics
        WHERE trial_type = ?
            AND tracker = ?
            AND metric_name = ?;
        """
    

    @property
    def df(self):
        return pd.read_sql("SELECT * from view_metrics", self.con)
        



if __name__ == "__main__":
    import pandas as pd
    path_to_database = r"mydb3.sqlite"

    participants = ['JSM', 'OKK']

    for participant in participants:
        path_to_yaml = Path(r"session_yamls")/f"{participant.lower()}.yaml"
        db = DatabaseHelper(path_to_database)
        db.load_from_yaml(path_to_yaml)

    # db.add_participant(
    #     participant_code="JSM",
    #     session_date= "2025-07-31"
    # )

    # path_to_recording = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-16-23_GMT-4_jsm_nih_trial_2")

    # db.add_trial(
    #     participant_code="JSM",
    #     trial_type="balance",
    #     trial_number="1",
    #     data_root=path_to_recording,
    #     notes=""
    # )

    # db.add_data_to_trial(
    #     data_root=path_to_recording,
    #     metric_name="path_length",
    #     tracker="mediapipe",
    #     folder_name="analysis_2025-08-26_22_25_36"
    # )
    # f = 2




