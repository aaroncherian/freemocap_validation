from validation.pipeline.base import ValidationStep
from validation.components import FREEMOCAP_PARQUET, QUALISYS_PARQUET, FREEMOCAP_GAIT_EVENTS, QUALISYS_GAIT_EVENTS, LEFT_FOOT_STEPS, RIGHT_FOOT_STEPS
from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from validation.steps.step_finder.components import REQUIRES, PRODUCES
from validation.steps.step_finder.core.step_finding import detect_gait_events, interval_cluster, suspicious_events_from_intervals, find_suspicious_events, GaitEventsFlagged
from validation.steps.step_finder.core.calculate_kinematics import get_foot_kinematics, FootKinematics
from validation.steps.step_finder.core.models import GaitResults
from validation.steps.step_finder.core.steps_plot import plot_gait_events_over_time, plot_gait_events_over_time_debug
from validation.steps.step_finder.config import StepFinderConfig
import numpy as np

class StepFinderStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = StepFinderConfig

    def calculate(self):
        self.logger.info("Starting step finding")
        sampling_rate = self.cfg.sampling_rate

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        qualisys_parquet_path = self.data[QUALISYS_PARQUET.name]

        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        freemocap_foot_kinematics:FootKinematics = get_foot_kinematics(freemocap_actor, sampling_rate)

        qualisys_actor = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)
        qualisys_foot_kinematics:FootKinematics = get_foot_kinematics(qualisys_actor, sampling_rate)


        freemocap_gait_events:GaitResults = detect_gait_events(
            left_heel_velocity=freemocap_foot_kinematics.left_heel_vel,
            left_toe_velocity=freemocap_foot_kinematics.left_toe_vel,
            right_heel_velocity=freemocap_foot_kinematics.right_heel_vel,
            right_toe_velocity=freemocap_foot_kinematics.right_toe_vel,
            frames_of_interest=self.cfg.frames_of_interest,
        )

        qualisys_gait_events:GaitResults = detect_gait_events(
            left_heel_velocity=qualisys_foot_kinematics.left_heel_vel,
            left_toe_velocity=qualisys_foot_kinematics.left_toe_vel,
            right_heel_velocity=qualisys_foot_kinematics.right_heel_vel,
            right_toe_velocity=qualisys_foot_kinematics.right_toe_vel,
            frames_of_interest=self.cfg.frames_of_interest,
        )



        freemocap_flagged_events:GaitEventsFlagged = find_suspicious_events(foot_kinematics=freemocap_foot_kinematics, gait_events=freemocap_gait_events)
        
        hs_cluster_flags_left  = np.isin(freemocap_gait_events.left_foot.heel_strikes, freemocap_flagged_events.left_foot.heel_strikes)
        to_cluster_flags_left  = np.isin(freemocap_gait_events.left_foot.toe_offs, freemocap_flagged_events.left_foot.toe_offs)

        hs_cluster_flags_right = np.isin(freemocap_gait_events.right_foot.heel_strikes, freemocap_flagged_events.right_foot.heel_strikes)
        to_cluster_flags_right = np.isin(freemocap_gait_events.right_foot.toe_offs, freemocap_flagged_events.right_foot.toe_offs)



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
            fmc_left_hs=freemocap_gait_events.left_foot.heel_strikes,
            fmc_left_to=freemocap_gait_events.left_foot.toe_offs,
            # right
            q_right_hs=qualisys_gait_events.right_foot.heel_strikes,
            q_right_to=qualisys_gait_events.right_foot.toe_offs,
            fmc_right_hs=freemocap_gait_events.right_foot.heel_strikes,
            fmc_right_to=freemocap_gait_events.right_foot.toe_offs,
            sampling_rate=self.cfg.sampling_rate,
            fmc_left_hs_cluster_flags=hs_cluster_flags_left,
            fmc_left_to_cluster_flags=to_cluster_flags_left,
            fmc_right_hs_cluster_flags=hs_cluster_flags_right,
            fmc_right_to_cluster_flags=to_cluster_flags_right,
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
        

