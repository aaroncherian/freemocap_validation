from validation.pipeline.base import ValidationStep
from validation.steps.gait_parameters.config import GaitParametersConfig
from validation.steps.gait_parameters.components import REQUIRES, PRODUCES, QUALISYS_GAIT_EVENTS, FREEMOCAP_GAIT_EVENTS

from validation.steps.gait_parameters.core.get_slices import get_heel_strike_frames, get_toe_off_frames
import numpy as np
from dataclasses import dataclass

@dataclass
class ParamStats:
    data: np.ndarray
    mean: np.ndarray
    std: np.ndarray

def calculate_stride_time(heel_strikes:list,sampling_rate:float):
    stride_time = np.diff(heel_strikes)/sampling_rate
    mean_stride_time = np.mean(stride_time)
    std_stride_time = np.std(stride_time)

    return ParamStats(
        data = stride_time,
        mean = mean_stride_time,
        std = std_stride_time
    )

class GaitParametersStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = GaitParametersConfig

    def calculate(self, condition_frame_range:list[int]=None):
        self.logger.info("Calculating gait parameters")
        sampling_rate = self.cfg.sampling_rate

        qualisys_gait_events = self.data[QUALISYS_GAIT_EVENTS.name]
        freemocap_gait_events = self.data[FREEMOCAP_GAIT_EVENTS.name]

        frame_range = range(*condition_frame_range) if condition_frame_range is not None else None

        qualisys_heel_strikes = get_heel_strike_frames(qualisys_gait_events, frame_range)
        qualisys_toe_offs = get_toe_off_frames(qualisys_gait_events, frame_range)

        freemocap_heel_strikes = get_heel_strike_frames(freemocap_gait_events, frame_range)
        freemocap_toe_offs = get_toe_off_frames(freemocap_gait_events, frame_range)

        q_stride_params = calculate_stride_time(heel_strikes=qualisys_heel_strikes['right'], sampling_rate=sampling_rate)
        f_stride_params = calculate_stride_time(heel_strikes=freemocap_heel_strikes['right'], sampling_rate =sampling_rate)

        f = 2 