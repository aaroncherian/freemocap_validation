
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

def remove_events_within_minimum_interval(event_candidates:np.ndarray, min_interval:float, sampling_rate:float):
    dt = 1/sampling_rate
    events = []

    i = 0
    logger.info(f"Removing events that are within {min_interval:.2f}s of each other from {event_candidates.shape[0]} candidates")

    events = [event_candidates[0]]
    for i in event_candidates[1:]:
        if (i - events[-1])*dt > min_interval:
            events.append(i)
        else:
            logger.info(f"Removing event at frame {i} which is {(i - events[-1])*dt:.3f}s after previous event at frame {events[-1]}")

    return np.array(events, dtype = int)

def get_heel_strike_and_toe_off_events(heel_velocity:np.ndarray, toe_velocity:np.ndarray, sampling_rate:float, min_event_interval_seconds:float):
    heel_strike_candidates = np.where((heel_velocity[:-1,1]>0) & (heel_velocity[1:, 1] <= 0)) [0] + 1
    toe_off_candidates = np.where((toe_velocity[:-1,1]<=0) & (toe_velocity[1:,1]>0))[0] + 1
    
    heel_strikes = remove_events_within_minimum_interval(
        event_candidates=heel_strike_candidates,
        min_interval=min_event_interval_seconds,
        sampling_rate=sampling_rate
    )
    toe_offs = remove_events_within_minimum_interval(
        event_candidates=toe_off_candidates,
        min_interval=min_event_interval_seconds,
        sampling_rate=sampling_rate
    )
    return GaitEvents(heel_strikes=heel_strikes, toe_offs=toe_offs)


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

    return GaitResults(right_foot=right_foot_gait_events, left_foot=left_foot_gait_events)
