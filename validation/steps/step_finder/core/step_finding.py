
from skellymodels.managers.human import Human
from validation.steps.step_finder.core.models import GaitEvents, GaitResults
import numpy as np
import logging

logger = logging.getLogger(__name__)

def parse_human(human:Human):
    left_heel = human.body.xyz.as_dict['left_heel']
    right_heel = human.body.xyz.as_dict['right_heel']
    
    left_toe = human.body.xyz.as_dict['left_foot_index']
    right_toe = human.body.xyz.as_dict['right_foot_index']

    return left_heel, right_heel, left_toe, right_toe

def get_velocity(positions:np.ndarray, sampling_rate:float):
    dt = 1.0 / sampling_rate
    velocities = np.gradient(positions, dt, axis=0)

    return velocities


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


def detect_gait_events(human:Human, 
                       sampling_rate:float, 
                       frames_of_interest:tuple[int,int]|None = None):

    left_heel, right_heel, left_toe, right_toe = parse_human(human)

    left_heel_velocity = get_velocity(left_heel, sampling_rate)
    right_heel_velocity = get_velocity(right_heel, sampling_rate)
    left_toe_velocity = get_velocity(left_toe, sampling_rate)
    right_toe_velocity = get_velocity(right_toe, sampling_rate)

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


    hs_clusters_left = interval_cluster(left_foot_gait_events.heel_strikes, median_threshold=0.6)
    to_clusters_left = interval_cluster(left_foot_gait_events.toe_offs, median_threshold=0.6)

    hs_clusters_right = interval_cluster(right_foot_gait_events.heel_strikes, median_threshold=0.6)
    to_clusters_right = interval_cluster(right_foot_gait_events.toe_offs, median_threshold=0.6)

    hs_cluster_flags_left = make_cluster_flags(left_foot_gait_events.heel_strikes, hs_clusters_left)
    to_cluster_flags_left = make_cluster_flags(left_foot_gait_events.toe_offs, to_clusters_left)

    hs_cluster_flags_right = make_cluster_flags(right_foot_gait_events.heel_strikes, hs_clusters_right)
    to_cluster_flags_right = make_cluster_flags(right_foot_gait_events.toe_offs, to_clusters_right)

    from validation.steps.step_finder.core.steps_plot import plot_gait_event_diagnostics
    fig_left = plot_gait_event_diagnostics(
        heel_pos=left_heel,
        toe_pos=left_toe,
        heel_strikes=left_foot_gait_events.heel_strikes,
        toe_offs=left_foot_gait_events.toe_offs,
        sampling_rate=sampling_rate,
        hs_short_cluster_flags=hs_cluster_flags_left,
        to_short_cluster_flags=to_cluster_flags_left,
    )
    # fig_left.show()

    fig_right = plot_gait_event_diagnostics(
        heel_pos=right_heel,
        toe_pos=right_toe,
        heel_strikes=right_foot_gait_events.heel_strikes,
        toe_offs=right_foot_gait_events.toe_offs,
        sampling_rate=sampling_rate,
        hs_short_cluster_flags=hs_cluster_flags_right,
        to_short_cluster_flags=to_cluster_flags_right,
    )
    # fig_right.show()

    return GaitResults(right_foot=right_foot_gait_events, left_foot=left_foot_gait_events)
