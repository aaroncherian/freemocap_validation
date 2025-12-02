from validation.pipeline.base import ValidationStep
from validation.components import FREEMOCAP_PARQUET, QUALISYS_PARQUET, FREEMOCAP_GAIT_EVENTS, QUALISYS_GAIT_EVENTS, LEFT_FOOT_STEPS, RIGHT_FOOT_STEPS
from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from validation.steps.step_finder.components import REQUIRES, PRODUCES
from validation.steps.step_finder.core.step_finding import detect_gait_events, interval_cluster, make_cluster_flags, suspicious_events_from_intervals
from validation.steps.step_finder.core.models import GaitResults
from validation.steps.step_finder.core.steps_plot import plot_gait_events_over_time, plot_gait_events_over_time_debug
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
            frames_of_interest=self.cfg.frames_of_interest,
        )

        qualisys_gait_events:GaitResults = detect_gait_events(
            human=qualisys_actor,
            sampling_rate=self.cfg.sampling_rate,
            frames_of_interest=self.cfg.frames_of_interest,
        )
        fmc_left_hs = freemocap_gait_events.left_foot.heel_strikes
        fmc_left_to = freemocap_gait_events.left_foot.toe_offs
        left_hs_clusters = interval_cluster(fmc_left_hs, median_threshold=0.6)
        left_to_clusters = interval_cluster(fmc_left_to, median_threshold=0.6)
        # left_hs_cluster_flags = make_cluster_flags(fmc_left_hs, left_hs_clusters)
        # left_to_cluster_flags = make_cluster_flags(fmc_left_to, left_to_clusters)
        # left_hs_cluster_flags = suspicious_events_from_intervals(fmc_left_hs, left_hs_clusters)

        # Right foot
        fmc_right_hs = freemocap_gait_events.right_foot.heel_strikes
        fmc_right_to = freemocap_gait_events.right_foot.toe_offs

        right_hs_clusters = interval_cluster(fmc_right_hs, median_threshold=0.6)
        right_to_clusters = interval_cluster(fmc_right_to, median_threshold=0.6)

        # right_hs_cluster_flags = make_cluster_flags(fmc_right_hs, right_hs_clusters)
        # right_to_cluster_flags = make_cluster_flags(fmc_right_to, right_to_clusters)

        left_hs_suspicious = suspicious_events_from_intervals(fmc_left_hs, median_threshold=0.6)
        left_to_suspicious = suspicious_events_from_intervals(fmc_left_to, median_threshold=0.6)

        right_hs_suspicious = suspicious_events_from_intervals(fmc_right_hs, median_threshold=0.6)
        right_to_suspicious = suspicious_events_from_intervals(fmc_right_to, median_threshold=0.6)
        

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

        fig_debug = plot_gait_events_over_time_debug(
            # left
            q_left_hs=qualisys_gait_events.left_foot.heel_strikes,
            q_left_to=qualisys_gait_events.left_foot.toe_offs,
            fmc_left_hs=fmc_left_hs,
            fmc_left_to=fmc_left_to,
            # right
            q_right_hs=qualisys_gait_events.right_foot.heel_strikes,
            q_right_to=qualisys_gait_events.right_foot.toe_offs,
            fmc_right_hs=fmc_right_hs,
            fmc_right_to=fmc_right_to,
            sampling_rate=self.cfg.sampling_rate,
            fmc_left_hs_cluster_flags=left_hs_suspicious,
            fmc_left_to_cluster_flags=left_to_suspicious,
            fmc_right_hs_cluster_flags=right_hs_suspicious,
            fmc_right_to_cluster_flags=right_to_suspicious,
            title=f"Gait events with FreeMoCap clusters for {self.ctx.recording_dir.stem}",
            xlim=None,
        )

        fig_debug.show()

        path_to_save = self.ctx.recording_dir / "validation" / self.ctx.project_config.freemocap_tracker / "gait_events"
        path_to_save.mkdir(parents=True, exist_ok=True)

        self.outputs[LEFT_FOOT_STEPS.name] = fig_left
        self.outputs[RIGHT_FOOT_STEPS.name] = fig_right

        self.outputs[FREEMOCAP_GAIT_EVENTS.name] = freemocap_events_df
        self.outputs[QUALISYS_GAIT_EVENTS.name] = qualisys_events_df
        

