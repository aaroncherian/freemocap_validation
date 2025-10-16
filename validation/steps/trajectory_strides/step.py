from validation.pipeline.base import ValidationStep
from validation.components import   (QUALISYS_GAIT_EVENTS, 
                                    FREEMOCAP_PARQUET, 
                                    QUALISYS_PARQUET,
                                    FREEMOCAP_TRAJECTORY_CYCLES,
                                    QUALISYS_TRAJECTORY_CYCLES,
                                    FREEMOCAP_TRAJECTORY_SUMMARY_STATS,
                                    QUALISYS_TRAJECTORY_SUMMARY_STATS, 
                                    TRAJECTORY_PER_STRIDE_FIG,
                                    TRAJECTORY_MEAN_FIG)

from validation.steps.trajectory_strides.components import REQUIRES, PRODUCES
from validation.steps.trajectory_strides.config import TrajectoryStridesConfig
from validation.steps.trajectory_strides.core.stride_slices import get_heel_strike_slices
from validation.steps.trajectory_strides.core.trajectory_cycles import create_trajectory_cycles, get_trajectory_summary
from validation.steps.trajectory_strides.core.trajectory_gait_plots import plot_trajectory_cycles_grid, plot_trajectory_summary_grid

from validation.steps.trajectory_strides.core.joint_angle_cycles import create_angle_cycles, get_angle_summary
from validation.steps.trajectory_strides.core.angle_gait_plots import plot_angle_summary_grid

from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from skellymodels.managers.human import Human


class TrajectoryStridesStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = TrajectoryStridesConfig

    def calculate(self):
        
        gait_events = self.data[QUALISYS_GAIT_EVENTS.name]
        frame_range = range(*self.cfg.frame_range) if self.cfg.frame_range is not None else None

        freemocap_actor:Human = make_freemocap_actor_from_parquet(parquet_path=self.data[FREEMOCAP_PARQUET.name])
        qualisys_actor:Human = make_freemocap_actor_from_parquet(parquet_path=self.data[QUALISYS_PARQUET.name])

        heel_strikes:dict[str, list[slice]] = get_heel_strike_slices(gait_events=gait_events, frame_range=frame_range)

        for side, slices in heel_strikes.items():
            self.logger.info(f"Found {len(slices)} strides for the {side} foot")


        markers = ["hip", "knee", "ankle", "heel", "foot_index"]

        qtm = qualisys_actor.body.xyz.as_dict
        fmc = freemocap_actor.body.xyz.as_dict

        self.logger.info(f"Separating trajectories into strides")
        trajectory_per_stride = create_trajectory_cycles(
            freemocap_dict=fmc,
            qualisys_dict=qtm,
            marker_list=markers,
            gait_events=heel_strikes,
            freemocap_tracker_name = self.ctx.project_config.freemocap_tracker,
            n_points=100
        )

        self.logger.info(f"Computing summary statistics for trajectories per stride")
        trajectory_summary_stats = get_trajectory_summary(trajectory_per_stride)

        split_t_stride_dfs = {tracker: df_t for tracker, df_t in trajectory_per_stride.groupby("tracker")}
        self.outputs[FREEMOCAP_TRAJECTORY_CYCLES.name] = split_t_stride_dfs[self.ctx.project_config.freemocap_tracker]
        self.outputs[QUALISYS_TRAJECTORY_CYCLES.name] = split_t_stride_dfs["qualisys"]

        split_t_summary_dfs = {tracker: df_t for tracker, df_t in trajectory_summary_stats.groupby("tracker")}
        self.outputs[FREEMOCAP_TRAJECTORY_SUMMARY_STATS.name] = split_t_summary_dfs[self.ctx.project_config.freemocap_tracker]
        self.outputs[QUALISYS_TRAJECTORY_SUMMARY_STATS.name] = split_t_summary_dfs["qualisys"]

        self.logger.info(f"Generating trajectories per gait cycle plots")
        fig = plot_trajectory_cycles_grid(trajectory_per_stride, marker_order=markers)

        self.logger.info(f"Generating trajectory summary statistics plots")
        fig_summary = plot_trajectory_summary_grid(trajectory_summary_stats, marker_order=markers)
        
        self.outputs[TRAJECTORY_PER_STRIDE_FIG.name] = fig
        self.outputs[TRAJECTORY_MEAN_FIG.name] = fig_summary


        # if FREEMOCAP_JOINT_ANGLES.exists(self.ctx.recording_dir, **self.ctx.data_component_context) and \
        #    QUALISYS_JOINT_ANGLES.exists(self.ctx.recording_dir, **self.ctx.data_component_context):
        #     self.logger.info("Both FreeMoCap and Qualisys joint angles found, proceeding to stride separation")
            
        #     fmc_joint_angles = FREEMOCAP_JOINT_ANGLES.load(self.ctx.recording_dir, **self.ctx.data_component_context)
        #     qtm_joint_angles = QUALISYS_JOINT_ANGLES.load(self.ctx.recording_dir, **self.ctx.data_component_context)
            
        #     angles_per_stride = create_angle_cycles(freemocap_df=fmc_joint_angles,
        #                         qualisys_df=qtm_joint_angles,
        #                         gait_events=heel_strikes,
        #                         freemocap_tracker_name=self.ctx.project_config.freemocap_tracker,
        #                         n_points= 100)
            
        #     angle_summary_stats = get_angle_summary(angles_per_stride)

        #     self.save_by_tracker(angles_per_stride, "joint_angles", "joint_angles_per_stride.csv")
        #     self.save_by_tracker(angle_summary_stats, "joint_angles", "joint_angles_per_stride_summary_stats.csv")
        #     fig_angle_summary = plot_angle_summary_grid(angle_summary_stats)
        #     fig_angle_summary.write_html(self.ctx.freemocap_path / "joint_angles" / "joint_angles_summary.html")
        #     self.logger.info(f"Saved joint angle summary plot to {self.ctx.freemocap_path / 'joint_angles' / 'joint_angles_summary.html'}")
 




