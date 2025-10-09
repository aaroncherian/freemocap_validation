from validation.pipeline.base import ValidationStep
from validation.components import QUALISYS_GAIT_EVENTS
from validation.steps.stride_separator.components import REQUIRES
from validation.steps.stride_separator.config import StrideSeparatorConfig
from validation.steps.stride_separator.core.stride_slices import get_heel_strike_slices
from validation.steps.stride_separator.core.trajectory_cycles import create_trajectory_cycles
from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from skellymodels.managers.human import Human

class StrideSeparatorStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = []
    CONFIG = StrideSeparatorConfig

    def calculate(self):
        
        gait_events = self.data[QUALISYS_GAIT_EVENTS.name]
        frame_range = range(*self.cfg.frame_range) if self.cfg.frame_range is not None else None

        freemocap_actor:Human = make_freemocap_actor_from_parquet(parquet_path=self.data["freemocap_parquet"])
        qualisys_actor:Human = make_freemocap_actor_from_parquet(parquet_path=self.data["qualisys_parquet"])

        heel_strikes:dict[str, list[slice]] = get_heel_strike_slices(gait_events=gait_events, frame_range=frame_range)

        for side, slices in heel_strikes.items():
            self.logger.info(f"Found {len(slices)} strides for the {side} foot")


        markers = ["hip", "knee", "ankle", "heel", "foot_index"]

        qtm = qualisys_actor.body.xyz.as_dict
        fmc = freemocap_actor.body.xyz.as_dict

        trajectory_per_stride = create_trajectory_cycles(
            freemocap_dict=fmc,
            qualisys_dict=qtm,
            marker_list=markers,
            gait_events=heel_strikes,
            freemocap_tracker_name = self.ctx.project_config.freemocap_tracker
        )

        f = 2








        # right_foot_mask = (
        #     (gait_events["foot"] == "right") &
        #     (gait_events["event"] == "heel_strike")
        # )

        # right_hs = gait_events.loc[right_foot_mask, "frame"]

        # if self.cfg.frame_range is not None:
        #     start, stop = self.cfg.frame_range
        #     right_hs= right_hs[right_hs.ge(start) & right_hs.le(stop)]
        

        # self.logger.info(f"Loaded {len(right_hs)} heel strikes for the right foot")

        f = 2