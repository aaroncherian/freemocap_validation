from validation.pipeline.base import ValidationStep
from validation.components import FREEMOCAP_PARQUET, QUALISYS_PARQUET, FREEMOCAP_GAIT_EVENTS, QUALISYS_GAIT_EVENTS, LEFT_FOOT_STEPS, RIGHT_FOOT_STEPS
from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from validation.steps.step_finder.components import REQUIRES, PRODUCES
from validation.steps.step_finder.core.step_finding import detect_gait_events
from validation.steps.step_finder.core.models import GaitResults
from validation.steps.step_finder.core.steps_plot import plot_gait_events_over_time
from validation.steps.step_finder.config import StepFinderConfig

import matplotlib.pyplot as plt
import numpy as np

class StepFinderStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = StepFinderConfig

    def calculate(self):
        self.logger.info("Starting step finding")

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        qualisys_parquet_path = self.data[QUALISYS_PARQUET.name]

        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        qualisys_actor = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)

        freemocap_gait_events:GaitResults = detect_gait_events(
            human=freemocap_actor,
            sampling_rate=self.cfg.sampling_rate,
            min_event_interval_seconds=self.cfg.min_event_interval_seconds
        )

        qualisys_gait_events:GaitResults = detect_gait_events(
            human=qualisys_actor,
            sampling_rate=self.cfg.sampling_rate,
            min_event_interval_seconds=self.cfg.min_event_interval_seconds
        )

        freemocap_events_df = freemocap_gait_events.to_dataframe()
        qualisys_events_df = qualisys_gait_events.to_dataframe()

        fig_right = plot_gait_events_over_time(
            q_hs=qualisys_gait_events.right_foot.heel_strikes,
            q_to=qualisys_gait_events.right_foot.toe_offs,
            fmc_hs=freemocap_gait_events.right_foot.heel_strikes,
            fmc_to=freemocap_gait_events.right_foot.toe_offs,
            sampling_rate=self.cfg.sampling_rate,
            title=f"Right foot gait events for {self.ctx.recording_dir.stem}",
            xlim=None
        )

        fig_left = plot_gait_events_over_time(
            q_hs=qualisys_gait_events.left_foot.heel_strikes,
            q_to=qualisys_gait_events.left_foot.toe_offs,
            fmc_hs=freemocap_gait_events.left_foot.heel_strikes,
            fmc_to=freemocap_gait_events.left_foot.toe_offs,
            sampling_rate=self.cfg.sampling_rate,
            title=f"Left foot gait events for {self.ctx.recording_dir.stem}",
            xlim=None
        )

        path_to_save = self.ctx.recording_dir / "validation" / self.ctx.project_config.freemocap_tracker / "gait_events"
        path_to_save.mkdir(parents=True, exist_ok=True)

        self.outputs[LEFT_FOOT_STEPS.name] = fig_left
        self.outputs[RIGHT_FOOT_STEPS.name] = fig_right

        self.outputs[FREEMOCAP_GAIT_EVENTS.name] = freemocap_events_df
        self.outputs[QUALISYS_GAIT_EVENTS.name] = qualisys_events_df
        

