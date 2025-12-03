
from validation.steps.step_finder.core.models import GaitEvents, GaitResults
from validation.steps.step_finder.core.calculate_kinematics import FootKinematics
import numpy as np
import logging
from dataclasses import dataclass




logger = logging.getLogger(__name__)

def _remove_flagged(events: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Remove any events that appear in `flagged`."""
    if flagged is None or flagged.size == 0:
        return events
    keep_mask = ~np.isin(events, flagged)
    return events[keep_mask]

def _clean_gait_events(
    events: GaitEvents,
    flagged: GaitEvents,
) -> GaitEvents:
    """Return a new GaitEvents with flagged events removed."""
    return GaitEvents(
        heel_strikes=_remove_flagged(events.heel_strikes, flagged.heel_strikes),
        toe_offs=_remove_flagged(events.toe_offs, flagged.toe_offs),
    )

def remove_flagged_events_from_gait_results(
    gait_events: GaitResults,
    flagged_events: GaitResults,
) -> GaitResults:

    clean_left = _clean_gait_events(
        gait_events.left_foot,
        flagged_events.left_foot,
    )

    clean_right = _clean_gait_events(
        gait_events.right_foot,
        flagged_events.right_foot,
    )

    return GaitResults(
        left_foot=clean_left,
        right_foot=clean_right,
    )

def get_heel_strike_and_toe_off_events(heel_velocity:np.ndarray, 
                                       toe_velocity:np.ndarray,
                                       frames_of_interest:tuple[int,int]|None = None,):
    
    heel_strike_candidates = np.where((heel_velocity[:-1,1]>0) & (heel_velocity[1:, 1] <= 0)) [0] + 1
    toe_off_candidates = np.where((toe_velocity[:-1,1]<=0) & (toe_velocity[1:,1]>0))[0] + 1
    
    heel_strikes = heel_strike_candidates    
    toe_offs = toe_off_candidates

    if frames_of_interest is not None:
        start_frame, end_frame = frames_of_interest
        heel_strikes = heel_strikes[(heel_strikes >= start_frame) & (heel_strikes <= end_frame)]
        toe_offs = toe_offs[(toe_offs >= start_frame) & (toe_offs <= end_frame)]
    
    
    return GaitEvents(heel_strikes=heel_strikes, toe_offs=toe_offs)

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




def find_suspicious_events(foot_kinematics: FootKinematics, gait_events: GaitResults):

    left_hs_flagged = flag_events_for_removal(positions=foot_kinematics.left_heel_pos, events = gait_events.left_foot.heel_strikes)
    left_to_flagged = flag_events_for_removal(positions=foot_kinematics.left_toe_pos, events = gait_events.left_foot.toe_offs)
    right_hs_flagged = flag_events_for_removal(positions=foot_kinematics.right_heel_pos, events = gait_events.right_foot.heel_strikes)
    right_to_flagged = flag_events_for_removal(positions=foot_kinematics.right_toe_pos, events = gait_events.right_foot.toe_offs)

    # hs_cluster_flags_left  = np.isin(gait_events.left_foot.heel_strikes, left_hs_flagged)
    # to_cluster_flags_left  = np.isin(gait_events.left_foot.toe_offs, left_to_flagged)

    # hs_cluster_flags_right = np.isin(gait_events.right_foot.heel_strikes, right_hs_flagged)
    # to_cluster_flags_right = np.isin(gait_events.right_foot.toe_offs, right_to_flagged)


    # fig_left = plot_gait_event_diagnostics(
    #     heel_pos=foot_kinematics.left_heel_pos,
    #     toe_pos=foot_kinematics.left_toe_pos,
    #     heel_strikes=gait_events.left_foot.heel_strikes,
    #     toe_offs=gait_events.left_foot.toe_offs,
    #     sampling_rate=30,
    #     hs_short_cluster_flags=hs_cluster_flags_left,
    #     to_short_cluster_flags=to_cluster_flags_left,
    # )
    # fig_left.show()

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


def detect_gait_events(left_heel_velocity:np.ndarray,
                       left_toe_velocity:np.ndarray,
                       right_heel_velocity:np.ndarray,
                       right_toe_velocity:np.ndarray,
                       frames_of_interest:tuple[int,int]|None = None,):

    right_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=right_heel_velocity,
        toe_velocity=right_toe_velocity,
        frames_of_interest=frames_of_interest,
    )
    
    left_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=left_heel_velocity,
        toe_velocity=left_toe_velocity,
        frames_of_interest=frames_of_interest,
    )

    return GaitResults(right_foot=right_foot_gait_events, left_foot=left_foot_gait_events)
