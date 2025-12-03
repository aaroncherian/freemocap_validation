
from validation.steps.step_finder.core.models import GaitEvents, GaitResults
from validation.steps.step_finder.core.calculate_kinematics import FootKinematics
import numpy as np
import logging
from validation.steps.step_finder.core.steps_plot import plot_gait_event_diagnostics
from dataclasses import dataclass

@dataclass
class GaitEvents:
    heel_strikes: np.ndarray
    toe_offs: np.ndarray

@dataclass
class GaitEventsFlagged:
    right_foot: GaitEvents
    left_foot: GaitEvents




logger = logging.getLogger(__name__)




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

    return GaitEventsFlagged(
        right_foot=GaitEvents(
            heel_strikes=right_hs_flagged,
            toe_offs=right_to_flagged,
        ),
        left_foot=GaitEvents(
            heel_strikes=left_hs_flagged,
            toe_offs=left_to_flagged,
        ),
    )





def make_cluster_flags(event_indices: np.ndarray,
                       clusters: list[np.ndarray]) -> np.ndarray:
    """
    Returns a bool mask of shape (len(event_indices),)
    True where the event is part of any cluster.
    """
    flags = np.zeros(event_indices.shape[0], dtype=bool)

    if not clusters:
        return flags

    # Flatten all clustered event frame numbers into a set for fast lookup
    clustered_events = set(int(e) for cluster in clusters for e in cluster)

    for i, ev in enumerate(event_indices):
        if int(ev) in clustered_events:
            flags[i] = True

    return flags

def suspicious_events_from_intervals(
    event_indices: np.ndarray,
    median_threshold: float = 0.6,
) -> np.ndarray:
    """
    Mark events as suspicious based on short intervals.

    - Compute HS→HS (or TO→TO) intervals.
    - An interval is "short" if it's < median_threshold * global_median.
    - An event is suspicious if it sits between *two* short intervals
      (i.e. both the interval before and after are short).

    Returns
    -------
    suspicious : bool array, same length as event_indices
                 True where the event is "core suspicious".
    """
    event_indices = np.asarray(event_indices, dtype=int)
    n = event_indices.size
    suspicious = np.zeros(n, dtype=bool)

    if n < 2:
        return suspicious

    diffs = np.diff(event_indices)  # length n-1
    median_interval = np.median(diffs)
    if median_interval <= 0:
        return suspicious

    thresh = median_threshold * median_interval
    is_short = diffs <= thresh  # length n-1

    # First event: only has "after" interval
    if is_short[0]:
        suspicious[0] = True

    # Internal events: before OR after short
    for i in range(1, n - 1):
        if is_short[i - 1] or is_short[i]:
            suspicious[i] = True

    # Last event: only has "before" interval
    if is_short[-1]:
        suspicious[-1] = True

    return suspicious

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


    # hs_clusters_left = interval_cluster(left_foot_gait_events.heel_strikes, median_threshold=0.6)
    # to_clusters_left = interval_cluster(left_foot_gait_events.toe_offs, median_threshold=0.6)

    # hs_clusters_right = interval_cluster(right_foot_gait_events.heel_strikes, median_threshold=0.6)
    # to_clusters_right = interval_cluster(right_foot_gait_events.toe_offs, median_threshold=0.6)

    # hs_cluster_flags_left = make_cluster_flags(left_foot_gait_events.heel_strikes, hs_clusters_left)
    # to_cluster_flags_left = make_cluster_flags(left_foot_gait_events.toe_offs, to_clusters_left)

    # hs_cluster_flags_right = make_cluster_flags(right_foot_gait_events.heel_strikes, hs_clusters_right)
    # to_cluster_flags_right = make_cluster_flags(right_foot_gait_events.toe_offs, to_clusters_right)

    # from validation.steps.step_finder.core.steps_plot import plot_gait_event_diagnostics
    # fig_left = plot_gait_event_diagnostics(
    #     heel_pos=left_heel,
    #     toe_pos=left_toe,
    #     heel_strikes=left_foot_gait_events.heel_strikes,
    #     toe_offs=left_foot_gait_events.toe_offs,
    #     sampling_rate=sampling_rate,
    #     hs_short_cluster_flags=hs_cluster_flags_left,
    #     to_short_cluster_flags=to_cluster_flags_left,
    # )
    # # fig_left.show()

    # fig_right = plot_gait_event_diagnostics(
    #     heel_pos=right_heel,
    #     toe_pos=right_toe,
    #     heel_strikes=right_foot_gait_events.heel_strikes,
    #     toe_offs=right_foot_gait_events.toe_offs,
    #     sampling_rate=sampling_rate,
    #     hs_short_cluster_flags=hs_cluster_flags_right,
    #     to_short_cluster_flags=to_cluster_flags_right,
    # )
    # # fig_right.show()

    return GaitResults(right_foot=right_foot_gait_events, left_foot=left_foot_gait_events)
