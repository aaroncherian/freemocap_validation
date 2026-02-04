import sqlite3
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EventMatchResult:
    differences: int
    false_positives: int
    false_negatives: int

conn = sqlite3.connect("validation.db")

query = """
SELECT t.participant_code,
        t.trial_name,
        a.path,
        a.component_name,
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


def find_closest_pair(reference_frame:int, tracker_frames:list, tolerance):
    tracker_frames = np.array(tracker_frames, dtype = int)
    closest_points = tracker_frames[(tracker_frames >= (reference_frame-tolerance)) & (tracker_frames <= (reference_frame + tolerance))]
    
    if len(closest_points) == 1:
        return closest_points[0]
    elif len(closest_points) == 0:
        return None
    elif len(closest_points) > 1:
        differences = closest_points - reference_frame
        return closest_points[np.argmin(np.abs(differences))]  #find the first closest point

   

def match_events(reference_frames:list, tracker_frames:list, tolerance: int = 2):
    reference_frames = sorted(reference_frames)
    tracker_frames = sorted(tracker_frames)
    
    differences = []
    num_ref_frames = len(reference_frames)
    num_tracker_frames = len(tracker_frames)

    if num_ref_frames > num_tracker_frames:
        print("More reference frames than tracker frames")
    elif num_ref_frames < num_tracker_frames:
        print("More tracker frames than reference frames")

    remaining_rframes = reference_frames.copy()
    remaining_tframes = tracker_frames.copy()
    for rframe in reference_frames:
        closest_frame = find_closest_pair(reference_frame=rframe, tracker_frames=remaining_tframes, tolerance=tolerance)
        
        if closest_frame is not None:
            remaining_rframes.pop(remaining_rframes.index(rframe))
            remaining_tframes.pop(remaining_tframes.index(closest_frame))
            
            differences.append(closest_frame - rframe)
        
    false_positives = len(remaining_tframes)
    false_negatives = len(remaining_rframes)

    return EventMatchResult(
        differences=differences,
        false_positives=false_positives,
        false_negatives=false_negatives
    )
            
    f= 2


path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"].lower()
    sub['tracker'] = row['tracker']
    dfs.append(sub)

df:pd.DataFrame = pd.concat(dfs, ignore_index=True)

differences_per_tracker = defaultdict(list)
fp_per_tracker = defaultdict(int)
fn_per_tracker = defaultdict(int)
for trial in df['trial_name'].unique():
    df_trial = df[df["trial_name"] == trial
                  ]
    for foot in df_trial['foot'].unique():
        df_foot = df_trial[df_trial['foot'] == foot]
        tracker_frames = {}
        for event in df_foot['event'].unique():

            sub_df = df_foot[df_foot['event'] == event]

            reference_frames = list(sub_df.groupby('tracker').get_group(reference_system)['frame'])

            for tracker in TRACKERS:
                tracker_frames = list(sub_df.groupby('tracker').get_group(tracker)['frame'])
                res:EventMatchResult = match_events(reference_frames, tracker_frames, tolerance=2)
                differences_per_tracker[tracker].extend(res.differences)
                fp_per_tracker[tracker] += res.false_positives
                fn_per_tracker[tracker] += res.false_negatives

            

            

            f = 2
f = 2