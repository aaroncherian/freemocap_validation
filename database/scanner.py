from pathlib import Path
import yaml
from time import time
from database.expected import (
    TREADMILL, BALANCE,
    FREEMOCAP_PATH_LENGTH_COM, QUALISYS_PATH_LENGTH_COM
)

SOLO_METRICS = {"synced_data", "gait_events", "joint_angles"}  # not per-condition

class ValidationScanner:
    def __init__(self, path_to_yaml: Path | str):
        with open(Path(path_to_yaml), "r") as f:
            self.yaml_data = yaml.safe_load(f)
        self.data_root = Path(self.yaml_data["data_root"])
        self.participant_code = self.yaml_data.get("participant_code")

    def scan_for_updates(self, only_existing: bool = False) -> list[dict]:
        rows: list[dict] = []
        for trial in self.yaml_data["trials"]:
            rec_path = self.data_root / trial["trial_name"]
            base_info = {
                "participant_code": self.participant_code,
                "trial_name": trial["trial_name"],
                "trial_type": trial["trial_type"],
                "trial_number": trial.get("trial_number"),
            }

            trackers = trial.get("trackers", [])
            for tr in trackers:
                tracker_name = tr["tracker"]

                if trial["trial_type"] == "balance":
                    rows += self._scan_balance(
                        base_info=base_info,
                        path_to_recording=rec_path,
                        tracker=tracker_name,
                        qualisys_analysis_folder=trial.get("qualisys_analysis_folder", ""),
                        freemocap_analysis_folder=tr.get("analysis_folder", ""),
                        only_existing=only_existing
                    )
                elif trial["trial_type"] == "treadmill":
                    rows += self._scan_treadmill(
                        base_info=base_info,
                        path_to_recording=rec_path,
                        tracker=tracker_name,
                        conditions=trial.get("conditions"),
                        only_existing=only_existing
                    )
        return rows

    # ---------- helpers ----------

    def _make_row(self, *, dc, base_dir: Path, ctx: dict,
                  base_info: dict, category: str,
                  condition: str | None, tracker: str) -> dict:
        p = dc.full_path(base_dir=base_dir, **ctx)
        exists = p.exists()
        size_b = p.stat().st_size if exists else None
        mtime = p.stat().st_mtime if exists else None

        row = {
            **base_info,
            "category": category,
            "condition": (condition or ""),               # None for non-conditioned outputs
            "tracker": tracker,                   # e.g., mediapipe, mediapipe_dlc, qualisys
            "component_name": dc.name,
            "path": str(p),
            "file_exists": int(exists),
            "size_bytes": size_b,
            "mtime_utc": mtime,
        }
        return row

    def _scan_treadmill(self, *, base_info: dict, path_to_recording: Path,
                        tracker: str, conditions: list[str] | None,
                        only_existing: bool) -> list[dict]:
        rows: list[dict] = []
        ctx = {"tracker": tracker, "recording_name": path_to_recording.name}

        for category, dc_list in TREADMILL.items():
            # No conditions → single, unprefixed check
            if not conditions or category in SOLO_METRICS:
                for dc in dc_list:
                    row = self._make_row(dc=dc, base_dir=path_to_recording, ctx=ctx,
                                         base_info=base_info, category=category,
                                         condition=None, tracker=tracker)
                    if (not only_existing) or row["file_exists"]:
                        rows.append(row)
                continue

            # Per-condition expansion
            for condition in conditions:
                for dc in dc_list:
                    dc_pref = dc.clone_with_prefix(f"{condition}")
                    row = self._make_row(dc=dc_pref, base_dir=path_to_recording, ctx=ctx,
                                         base_info=base_info, category=category,
                                         condition=condition, tracker=tracker)
                    if (not only_existing) or row["file_exists"]:
                        rows.append(row)
        return rows

    def _scan_balance(self, *, base_info: dict, path_to_recording: Path,
                      tracker: str, qualisys_analysis_folder: str,
                      freemocap_analysis_folder: str,
                      only_existing: bool) -> list[dict]:
        rows: list[dict] = []
        ctx = {"tracker": tracker, "recording_name": path_to_recording.name}

        for category, dc_list in BALANCE.items():
            for dc in dc_list:
                # Thread in analysis-folder as a subfolder prefix when present
                if dc.name == FREEMOCAP_PATH_LENGTH_COM.name and freemocap_analysis_folder:
                    dc = dc.clone_with_prefix(f"{freemocap_analysis_folder}", change_name=False)

                if dc.name == QUALISYS_PATH_LENGTH_COM.name and qualisys_analysis_folder:
                    dc_q = dc.clone_with_prefix(f"{qualisys_analysis_folder}", change_name=False)
                    # ensure qualisys context if your resolver ever keys off tracker
                    q_ctx = {**ctx, "tracker": "qualisys"}
                    row = self._make_row(dc=dc_q, base_dir=path_to_recording, ctx=q_ctx,
                                         base_info=base_info, category=category,
                                         condition=None, tracker="qualisys")
                    if (not only_existing) or row["file_exists"]:
                        rows.append(row)
                    continue

                row = self._make_row(dc=dc, base_dir=path_to_recording, ctx=ctx,
                                     base_info=base_info, category=category,
                                     condition=None, tracker=tracker)
                if (not only_existing) or row["file_exists"]:
                    rows.append(row)
        return rows

# path = Path(r"session_yamls/jsm copy.yaml")

# scanner = ValidationScanner(path_to_yaml=path)
# rows = scanner.scan_for_updates(only_existing=False)
# print(rows)

# path_to_yaml = Path(path)

# with open(path_to_yaml, 'r') as f:
#     data = yaml.safe_load(f)

# data_root = Path(data['data_root'])

# for trial in data['trials']:
#     full_path = data_root/trial['trial_name']
#     print(f"Path: {full_path.stem}")

#     for tracker in trial['trackers']:
#         tracker_name = tracker['tracker']
#         print(f"Tracker: {tracker_name}")
#         match trial['trial_type']:
#             case "balance":
#                 scan_balance(
#                             full_path,
#                             qualisys_analysis_folder = trial.get("qualisys_analysis_folder", ""), 
#                             freemocap_analysis_folder = tracker.get("analysis_folder", ""), 
#                             tracker = tracker_name)
#             case "treadmill":
#                 scan_treadmill(
#                     path_to_recording=full_path,
#                     tracker = tracker_name,
#                     conditions = trial.get("conditions")
#                 )
    
f = 2
