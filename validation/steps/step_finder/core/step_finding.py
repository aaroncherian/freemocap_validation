
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

def remove_events_within_minimum_interval(event_candidates:np.ndarray, 
                                          min_interval:float, 
                                          sampling_rate:float,
                                          preview = 20):
    dt = 1/sampling_rate

    logger.info(f"Removing events that are within {min_interval:.2f}s of each other from {event_candidates.shape[0]} candidates")

    kept = [int(event_candidates[0])]
    removed: list[tuple[int, float, int]] = []  # (frame, Δt_sec, prev_frame)
    for frame in map(int, event_candidates[1:]):
        delta_t = (frame - kept[-1]) * dt
        if delta_t > min_interval:
            kept.append(frame)
        else:
            removed.append((frame, float(delta_t), kept[-1]))
    
    if removed:
        preview_items = ", ".join(
            f"{f} ({dt_sec:.3f}s after {prev})"
            for f, dt_sec, prev in removed[:preview]
        )
        more = len(removed) - min(preview, len(removed))
        tail = f", … (+{more} more)" if more > 0 else ""
        logger.info(
            "Removed %d events (min_interval=%.2fs): %s%s",
            len(removed), min_interval, preview_items, tail
        )
    return np.asarray(kept, dtype=event_candidates.dtype)

def get_heel_strike_and_toe_off_events(heel_velocity:np.ndarray, toe_velocity:np.ndarray, sampling_rate:float, min_event_interval_seconds:float):
    heel_strike_candidates = np.where((heel_velocity[:-1,1]>0) & (heel_velocity[1:, 1] <= 0)) [0] + 1
    toe_off_candidates = np.where((toe_velocity[:-1,1]<=0) & (toe_velocity[1:,1]>0))[0] + 1
    
    # heel_strikes = remove_events_within_minimum_interval(
    #     event_candidates=heel_strike_candidates,
    #     min_interval=min_event_interval_seconds,
    #     sampling_rate=sampling_rate
    # )
    # toe_offs = remove_events_within_minimum_interval(
    #     event_candidates=toe_off_candidates,
    #     min_interval=min_event_interval_seconds,
    #     sampling_rate=sampling_rate
    # )

    heel_strikes = heel_strike_candidates    
    toe_offs = toe_off_candidates
    
    
    return GaitEvents(heel_strikes=heel_strikes, toe_offs=toe_offs)




def flag_short_intervals(
    event_indices: np.ndarray,
    sampling_rate: float,
    fraction_of_median: float = 0.6,
):
    """
    event_indices : 1D array of frame indices for HS or TO (sorted)
    sampling_rate : Hz
    fraction_of_median : threshold; intervals shorter than
                         fraction_of_median * median_dt are flagged

    Returns:
        is_short : 1D bool array, same length as event_indices.
                   is_short[i] == True means the interval between
                   event_indices[i-1] and event_indices[i] was "too short".
        dts      : 1D array of dt between consecutive events (seconds)
        median_dt: float, median dt
    """
    event_indices = np.asarray(event_indices)
    n = event_indices.size

    if n < 2:
        return np.zeros(n, dtype=bool), np.array([]), np.nan

    dts = np.diff(event_indices) / float(sampling_rate)
    median_dt = np.median(dts)

    # interval is "too short" if it's much smaller than the typical step time
    thresh = fraction_of_median * median_dt

    is_short_interval = dts < thresh

    # map to events: mark the *later* event in any too-short interval
    is_short_event = np.zeros(n, dtype=bool)
    is_short_event[1:] = is_short_interval

    return is_short_event, dts, median_dt


def interval_cluster(event_indices:np.ndarray,
                     median_threshold: float = 0.6):
                     
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

def detect_clusters(event_candidates:np.ndarray, min_event_interval_seconds:float):

    min_interval_frames = int(min_event_interval_seconds * sampling_rate)
    clusters: list[np.ndarray] = []

    current_cluster = [int(event_candidates[0])]

    for event in event_candidates[1:]:
        if event - current_cluster[-1] <= min_interval_frames:
            current_cluster.append(event)
        else:
            clusters.append(np.asarray(current_cluster,dtype=int))
            current_cluster = [event]

    clusters.append(np.asarray(current_cluster, dtype=int))

    return clusters
    

def cluster_by_short_intervals(
    event_indices: np.ndarray,
    sampling_rate: float,
    fraction_of_median: float = 0.6,
    min_cluster_size: int = 2,
):
    """
    event_indices : 1D array of frame indices for HS or TO (sorted)
    sampling_rate : Hz
    fraction_of_median : threshold factor; an interval shorter than
                         fraction_of_median * median_dt is considered "too short"
    min_cluster_size : minimum number of events in a cluster to keep

    Returns:
        clusters      : list of 1D np.ndarray of event indices (frame numbers)
        cluster_flags : 1D bool array, same length as event_indices.
                        True for events that belong to any short-interval cluster.
        dts           : 1D array of dt between consecutive events (seconds)
        median_dt     : float, median dt
    """
    event_indices = np.asarray(event_indices)
    n = event_indices.size

    if n < 2:
        return [], np.zeros(n, dtype=bool), np.array([]), np.nan

    # Intervals in seconds
    dts = np.diff(event_indices) / float(sampling_rate)
    median_dt = np.median(dts)

    # "too short" intervals
    thresh = fraction_of_median * median_dt
    is_short_interval = dts < thresh  # length n-1

    clusters = []
    cluster_flags = np.zeros(n, dtype=bool)

    current_cluster_indices = []

    for i in range(len(is_short_interval)):
        if is_short_interval[i]:
            # interval between event i and i+1 is too short
            if not current_cluster_indices:
                # start a new cluster with the earlier event
                current_cluster_indices.append(i)
            # always include the later event
            current_cluster_indices.append(i + 1)
        else:
            # interval is not short; close any current cluster
            if len(current_cluster_indices) >= min_cluster_size:
                # map index positions to actual frame numbers
                cluster_events = event_indices[current_cluster_indices]
                clusters.append(cluster_events)
                cluster_flags[current_cluster_indices] = True
            current_cluster_indices = []

    # Check for cluster at end
    if len(current_cluster_indices) >= min_cluster_size:
        cluster_events = event_indices[current_cluster_indices]
        clusters.append(cluster_events)
        cluster_flags[current_cluster_indices] = True

    return clusters, cluster_flags, dts, median_dt

def detect_clusters(event_candidates:np.ndarray, min_event_interval_seconds:float, sampling_rate:float):

    min_interval_frames = int(min_event_interval_seconds * sampling_rate)
    clusters: list[np.ndarray] = []

    current_cluster = [int(event_candidates[0])]

    for event in event_candidates[1:]:
        if event - current_cluster[-1] <= min_interval_frames:
            current_cluster.append(event)
        else:
            clusters.append(np.asarray(current_cluster,dtype=int))
            current_cluster = [event]

    clusters.append(np.asarray(current_cluster, dtype=int))

    return clusters
    
    f = 2

def detect_gait_events(human:Human, sampling_rate:float, min_event_interval_seconds:float):

    left_heel, right_heel, left_toe, right_toe = parse_human(human)

    left_heel_velocity = get_velocity(left_heel, sampling_rate)
    right_heel_velocity = get_velocity(right_heel, sampling_rate)
    left_toe_velocity = get_velocity(left_toe, sampling_rate)
    right_toe_velocity = get_velocity(right_toe, sampling_rate)

    right_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=right_heel_velocity,
        toe_velocity=right_toe_velocity,
        sampling_rate=sampling_rate,
        min_event_interval_seconds=min_event_interval_seconds
    )

    left_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=left_heel_velocity,
        toe_velocity=left_toe_velocity,
        sampling_rate=sampling_rate,
        min_event_interval_seconds=min_event_interval_seconds
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
