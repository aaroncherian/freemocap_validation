from validation.steps.step_finder.core.models import GaitEvents, GaitResults
from validation.steps.step_finder.core.calculate_kinematics import FootKinematics
import numpy as np

def find_suspicious_events(foot_kinematics: FootKinematics, gait_events: GaitResults):
    left_hs_flagged = flag_events_for_removal(positions=foot_kinematics.left_heel_pos, events = gait_events.left_foot.heel_strikes)
    left_to_flagged = flag_events_for_removal(positions=foot_kinematics.left_toe_pos, events = gait_events.left_foot.toe_offs)
    right_hs_flagged = flag_events_for_removal(positions=foot_kinematics.right_heel_pos, events = gait_events.right_foot.heel_strikes)
    right_to_flagged = flag_events_for_removal(positions=foot_kinematics.right_toe_pos, events = gait_events.right_foot.toe_offs)

    return GaitResults(
        right_foot=GaitEvents(
            heel_strikes=right_hs_flagged,
            toe_offs=right_to_flagged,
        ),
        left_foot=GaitEvents(
            heel_strikes=left_hs_flagged,
            toe_offs=left_to_flagged,
        ),
    )


def flag_events_for_removal(positions, events):
    event_positions = positions[events]

    suspicious_events_clusters = interval_cluster(
        event_indices=events,
        median_threshold=0.6
    )

    ap_positions = event_positions[:,1]
    height_positions = event_positions[:,2]
    
    med_ap_position = np.median(ap_positions)
    mad_ap = np.median(np.abs(ap_positions - med_ap_position)) or 1e-8

    med_height_position = np.median(height_positions)
    mad_height = np.median(np.abs(height_positions - med_height_position)) or 1e-8

    flagged_for_removal = []
    for cluster in suspicious_events_clusters:
        cluster_scores = {}
        for event in cluster:
            event_position_ap = positions[event][1]
            event_position_height = positions[event][2]

            z_score_ap = (event_position_ap - med_ap_position)/mad_ap
            z_score_height = (event_position_height - med_height_position)/mad_height

            total_z = abs(z_score_ap) + abs(z_score_height)
            cluster_scores[event] = total_z
        
        if cluster_scores:
            suspicious_event_index = max(cluster_scores, key = cluster_scores.get)
            flagged_for_removal.append(suspicious_event_index)

    return np.array(flagged_for_removal, dtype=int)



def interval_cluster(event_indices:np.ndarray,
                     median_threshold: float = 0.6,):
                     
    median_interval = np.median(np.diff(event_indices))

    interval_threshold = median_interval*median_threshold
    clusters: list[np.ndarray] = []
    current_cluster: list[int] = []

    for i, event in enumerate(event_indices[1:], start=1):
        gap = event_indices[i] - event_indices[i - 1]

        if gap <= interval_threshold:
            # short gap → these belong together
            if not current_cluster:
                # start cluster with previous event
                current_cluster.append(int(event_indices[i - 1]))
            current_cluster.append(int(event))
        else:
            # normal gap → close any active cluster
            if len(current_cluster) >= 2:
                clusters.append(np.asarray(current_cluster, dtype=int))
            current_cluster = []
    
    if len(current_cluster) >= 2:
        clusters.append(np.asarray(current_cluster, dtype=int))

    return clusters
