from pathlib import Path
import sqlite3
from time import time

from database.db import connect, upsert_trial, upsert_artifact
from database.scanner import ValidationScanner
from collections import defaultdict

def main(session_yaml:str, db_path: str = "validation.db"):
    scanner = ValidationScanner(session_yaml)
    rows = scanner.scan_for_updates(only_existing=False)

    conn = connect(db_path=db_path)

    yaml = scanner.yaml_data
    participant_code = yaml.get("participant_code")
    session_date = yaml.get("session_data")
    data_root = yaml.get("data_root")

    by_trial: dict[str, list[[dict]]] = defaultdict(list)

    for r in rows:
        by_trial[r["trial_name"]].append(r)
    
    for trial in yaml["trials"]:
        trial_name = trial["trial_name"]
        trial_type = trial["trial_type"]
        trial_number = trial["trial_number"]

        trial_id = upsert_trial(
            conn=conn,
            participant_code=participant_code,
            trial_name=trial_name,
            trial_type=trial_type,
            trial_number=str(trial_number),
            session_date=session_date,
            data_root=data_root,
            notes=trial.get("notes")
        )

        upsert_artifact(
            conn=conn,
            trial_id=trial_id,
            rows=by_trial.get(trial_name, [])
        )

        conn.commit()
        print("Upserted trial:", trial_name)
    f = 2

if __name__ == "__main__":
    session_yaml = Path(r"session_yamls")/"jsm copy.yaml"
    main(session_yaml, db_path= "validation.db")
    f = 2
