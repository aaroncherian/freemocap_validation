from validation.pipeline.base import ValidationStep
from validation.steps.gait_parameters.config import GaitParametersConfig
from validation.steps.gait_parameters.components import *

from validation.steps.gait_parameters.core.get_slices import get_heel_strike_frames, get_toe_off_frames
import numpy as np
import pandas as pd
from dataclasses import dataclass 

@dataclass
class ParamStats:
    data: np.ndarray
    mean: float
    std: float

@dataclass
class StanceSwingStats:
    stance: ParamStats
    swing: ParamStats

@dataclass
class GaitParameters:
    left_stride_duration: ParamStats
    right_stride_duration: ParamStats
    left_stance_duration: ParamStats
    right_stance_duration: ParamStats
    left_swing_duration: ParamStats
    right_swing_duration: ParamStats
    left_stance_percentage: ParamStats
    right_stance_percentage: ParamStats
    left_swing_percentage: ParamStats
    right_swing_percentage: ParamStats


def calculate_stride_duration(heel_strikes:list,sampling_rate:float):
    stride_duration = np.diff(heel_strikes)/sampling_rate
    mean_stride_duration = np.mean(stride_duration)
    std_stride_duration = np.std(stride_duration)

    return ParamStats(
        data = stride_duration,
        mean = mean_stride_duration,
        std = std_stride_duration
    )

def calculate_stance_swing_time(toe_offs:list, heel_strikes:list, sampling_rate:float):
    stance_times = []
    swing_times = []
    for i in range(len(heel_strikes)-1):
        hs_1 = heel_strikes[i]
        hs_2 = heel_strikes[i+1]

        toe_offs_within_step = [x for x in toe_offs if x > hs_1 and x < hs_2]

        if not toe_offs_within_step:
            print(f"No toe offs found between heel strikes at frames {hs_1, hs_2}. Inserting NaN.")
            stance_times.append(np.nan)
            swing_times.append(np.nan)
            continue

        if len(toe_offs_within_step) > 1:
            print(f"Found multiple toe offs between heel strikes at frames {hs_1,hs_2}, using the first")

        toe_off = toe_offs_within_step[0]
        stance_time = (toe_off - hs_1)/sampling_rate
        stance_times.append(stance_time)

        swing_time = (hs_2 - toe_off)/sampling_rate
        swing_times.append(swing_time)

    stance_times = np.array(stance_times)
    mean_stance_time = np.nanmean(stance_times)
    std_stance_time = np.nanstd(stance_times)

    swing_times = np.array(swing_times)
    mean_swing_time = np.nanmean(swing_times)
    std_swing_time = np.nanstd(swing_times)

    stance_data = ParamStats(
        data=stance_times,
        mean=mean_stance_time,
        std=std_stance_time
    )

    swing_data = ParamStats(
        data=swing_times,
        mean=mean_swing_time,
        std=std_swing_time
    )

    return StanceSwingStats(
        stance=stance_data,
        swing=swing_data
    )

def calculate_stance_swing_percentages(stance_swing_times:StanceSwingStats, stride_durations:ParamStats):
    stance_percentages = (stance_swing_times.stance.data / stride_durations.data) * 100.0
    mean_stance_percentage = np.nanmean(stance_percentages)
    std_stance_percentage = np.nanstd(stance_percentages)

    swing_percentages = (stance_swing_times.swing.data / stride_durations.data) * 100.0
    mean_swing_percentage = np.nanmean(swing_percentages)
    std_swing_percentage = np.nanstd(swing_percentages)

    stance_percentage_data = ParamStats(
        data=stance_percentages,
        mean=mean_stance_percentage,
        std=std_stance_percentage
    )

    swing_percentage_data = ParamStats(
        data=swing_percentages,
        mean=mean_swing_percentage,
        std=std_swing_percentage
    )

    return StanceSwingStats(
        stance=stance_percentage_data,
        swing=swing_percentage_data
    )

def calculate_gait_parameters(
    gait_events: pd.DataFrame,
    sampling_rate: float,
    frame_range: range | None,
) -> GaitParameters:
    heel_strikes: dict[str, list[int]] = get_heel_strike_frames(gait_events, frame_range)
    toe_offs: dict[str, list[int]] = get_toe_off_frames(gait_events, frame_range)

    left_stride_duration = calculate_stride_duration(
        heel_strikes=heel_strikes["left"],
        sampling_rate=sampling_rate,
    )
    right_stride_duration = calculate_stride_duration(
        heel_strikes=heel_strikes["right"],
        sampling_rate=sampling_rate,
    )

    left_stance_swing_stats = calculate_stance_swing_time(
        toe_offs=toe_offs["left"],
        heel_strikes=heel_strikes["left"],
        sampling_rate=sampling_rate,
    )
    right_stance_swing_stats = calculate_stance_swing_time(
        toe_offs=toe_offs["right"],
        heel_strikes=heel_strikes["right"],
        sampling_rate=sampling_rate,
    )

    left_stance_swing_percentages = calculate_stance_swing_percentages(
        stance_swing_times=left_stance_swing_stats,
        stride_durations=left_stride_duration,
    )
    right_stance_swing_percentages = calculate_stance_swing_percentages(
        stance_swing_times=right_stance_swing_stats,
        stride_durations=right_stride_duration,
    )

    return GaitParameters(
        left_stride_duration=left_stride_duration,
        right_stride_duration=right_stride_duration,
        left_stance_duration=left_stance_swing_stats.stance,
        right_stance_duration=right_stance_swing_stats.stance,
        left_swing_duration=left_stance_swing_stats.swing,
        right_swing_duration=right_stance_swing_stats.swing,
        left_stance_percentage=left_stance_swing_percentages.stance,
        right_stance_percentage=right_stance_swing_percentages.stance,
        left_swing_percentage=left_stance_swing_percentages.swing,
        right_swing_percentage=right_stance_swing_percentages.swing,
    )


def _param_to_gait_metrics_df(
    param: ParamStats,
    metric: str,
    side: str,
    system: str,
) -> pd.DataFrame:
    """
    Flatten a ParamStats.data array into a per-stride long-format DataFrame.
    """
    n = param.data.shape[0]
    return pd.DataFrame(
        {
            "system": system,
            "side": side,
            "metric": metric,           # e.g. "stride_duration", "stance_pct"
            "event_index": np.arange(n, dtype=int),
            "value": param.data,        # may contain NaNs
        }
    )


def gait_metrics_to_long_df(
    gp: GaitParameters,
    system: str,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    dfs.append(_param_to_gait_metrics_df(gp.left_stride_duration,  "stride_duration",  "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_stride_duration, "stride_duration",  "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_stance_duration,  "stance_duration",  "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_stance_duration, "stance_duration",  "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_swing_duration,   "swing_duration",   "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_swing_duration,  "swing_duration",   "right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_stance_percentage, "stance_pct", "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_stance_percentage,"stance_pct","right", system))

    dfs.append(_param_to_gait_metrics_df(gp.left_swing_percentage,  "swing_pct", "left",  system))
    dfs.append(_param_to_gait_metrics_df(gp.right_swing_percentage, "swing_pct", "right", system))

    return pd.concat(dfs, ignore_index=True)


def _param_to_summary_row(
    param: ParamStats,
    metric: str,
    side: str,
    system: str,
) -> dict:
    """
    Build a single summary row (mean, std, n_valid) for a given ParamStats.
    """
    data = param.data
    n_valid = int(np.isfinite(data).sum())
    return {
        "system": system,
        "side": side,
        "metric": metric,
        "mean": float(param.mean),
        "std": float(param.std),
        "n_valid": n_valid,
    }


def gait_parameters_to_summary_df(
    gp: GaitParameters,
    system: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    rows.append(_param_to_summary_row(gp.left_stride_duration,  "stride_duration",  "left",  system))
    rows.append(_param_to_summary_row(gp.right_stride_duration, "stride_duration",  "right", system))

    rows.append(_param_to_summary_row(gp.left_stance_duration,  "stance_duration",  "left",  system))
    rows.append(_param_to_summary_row(gp.right_stance_duration, "stance_duration",  "right", system))

    rows.append(_param_to_summary_row(gp.left_swing_duration,   "swing_duration",   "left",  system))
    rows.append(_param_to_summary_row(gp.right_swing_duration,  "swing_duration",   "right", system))

    rows.append(_param_to_summary_row(gp.left_stance_percentage, "stance_pct", "left",  system))
    rows.append(_param_to_summary_row(gp.right_stance_percentage,"stance_pct","right", system))

    rows.append(_param_to_summary_row(gp.left_swing_percentage,  "swing_pct", "left",  system))
    rows.append(_param_to_summary_row(gp.right_swing_percentage, "swing_pct", "right", system))

    return pd.DataFrame(rows)

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


        q_gp: GaitParameters = calculate_gait_parameters(
            gait_events=qualisys_gait_events,
            sampling_rate=sampling_rate,
            frame_range=frame_range,
        )
        f_gp: GaitParameters = calculate_gait_parameters(
            gait_events=freemocap_gait_events,
            sampling_rate=sampling_rate,
            frame_range=frame_range,
        )

        q_gait_metrics_df = gait_metrics_to_long_df(q_gp, system="qualisys")
        f_gait_metrics_df = gait_metrics_to_long_df(f_gp, system=self.ctx.project_config.freemocap_tracker)
        per_gait_metrics_df = pd.concat([q_gait_metrics_df, f_gait_metrics_df], ignore_index=True)

        q_summary_df = gait_parameters_to_summary_df(q_gp, system="qualisys")
        f_summary_df = gait_parameters_to_summary_df(f_gp, system=self.ctx.project_config.freemocap_tracker)
        summary_df = pd.concat([q_summary_df, f_summary_df], ignore_index=True)

        self.outputs[QUALISYS_GAIT_METRICS.name] = q_gait_metrics_df
        self.outputs[FREEMOCAP_GAIT_METRICS.name] = f_gait_metrics_df
        self.outputs[QUALISYS_GAIT_SUMMARY_STATS.name] = q_summary_df
        self.outputs[FREEMOCAP_GAIT_SUMMARY_STATS.name] = f_summary_df


        f = 2 