from validation.pipeline.base import ValidationStep
from validation.components import QUALISYS_GAIT_EVENTS, FREEMOCAP_JOINT_ANGLES, QUALISYS_JOINT_ANGLES
from validation.steps.stride_separator.components import REQUIRES
from validation.steps.stride_separator.config import StrideSeparatorConfig
from validation.steps.stride_separator.core.stride_slices import get_heel_strike_slices
from validation.steps.stride_separator.core.trajectory_cycles import create_trajectory_cycles, get_trajectory_summary
from validation.steps.stride_separator.core.trajectory_gait_plots import plot_trajectory_cycles_grid, plot_trajectory_summary_grid

from validation.steps.stride_separator.core.joint_angle_cycles import create_angle_cycles, get_angle_summary
from validation.steps.stride_separator.core.angle_gait_plots import plot_angle_summary_grid

from validation.utils.actor_utils import make_freemocap_actor_from_parquet
from skellymodels.managers.human import Human
import pandas as pd


class StrideSeparatorStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = []
    CONFIG = StrideSeparatorConfig

    def save_by_tracker(self, df:pd.DataFrame, folder_name:str, file_name:str):
        for tracker, df_t in df.groupby('tracker'):
            tracker_path = self.ctx.recording_dir/"validation"/tracker/folder_name
            tracker_path.mkdir(parents=True, exist_ok=True)
            df_t.to_csv(tracker_path/f"{tracker}_{file_name}.csv", index=False)
            self.logger.info(f"Saved {tracker} {file_name} to {tracker_path/f'{tracker}_{file_name}.csv'}")

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
            freemocap_tracker_name = self.ctx.project_config.freemocap_tracker,
            n_points=100
        )

        trajectory_summary_stats = get_trajectory_summary(trajectory_per_stride)

        self.save_by_tracker(trajectory_per_stride, "trajectories", "trajectories_per_stride.csv")        
        self.save_by_tracker(trajectory_summary_stats, "trajectories", "trajectories_per_stride_summary_stats.csv")

        freemocap_save_path = self.ctx.freemocap_path / "trajectories"

        fig = plot_trajectory_cycles_grid(trajectory_per_stride, marker_order=markers)
        fig_summary = plot_trajectory_summary_grid(trajectory_summary_stats, marker_order=markers)

        fig.write_html(freemocap_save_path / "trajectories_per_stride.html")
        self.logger.info(f"Saved trajectory cycles plot to {freemocap_save_path / 'trajectories_per_stride.html'}")
        fig_summary.write_html(freemocap_save_path / "trajectories_summary.html")
        self.logger.info(f"Saved trajectory summary plot to {freemocap_save_path / 'trajectories_summary.html'}")

        if FREEMOCAP_JOINT_ANGLES.exists(self.ctx.recording_dir, **self.ctx.data_component_context) and \
           QUALISYS_JOINT_ANGLES.exists(self.ctx.recording_dir, **self.ctx.data_component_context):
            self.logger.info("Both FreeMoCap and Qualisys joint angles found, proceeding to stride separation")
            
            fmc_joint_angles = FREEMOCAP_JOINT_ANGLES.load(self.ctx.recording_dir, **self.ctx.data_component_context)
            qtm_joint_angles = QUALISYS_JOINT_ANGLES.load(self.ctx.recording_dir, **self.ctx.data_component_context)
            
            angles_per_stride = create_angle_cycles(freemocap_df=fmc_joint_angles,
                                qualisys_df=qtm_joint_angles,
                                gait_events=heel_strikes,
                                freemocap_tracker_name=self.ctx.project_config.freemocap_tracker,
                                n_points= 100)
            
            angle_summary_stats = get_angle_summary(angles_per_stride)

            self.save_by_tracker(angles_per_stride, "joint_angles", "joint_angles_per_stride.csv")
            self.save_by_tracker(angle_summary_stats, "joint_angles", "joint_angles_per_stride_summary_stats.csv")
            fig_angle_summary = plot_angle_summary_grid(angle_summary_stats)
            fig_angle_summary.write_html(self.ctx.freemocap_path / "joint_angles" / "joint_angles_summary.html")
            self.logger.info(f"Saved joint angle summary plot to {self.ctx.freemocap_path / 'joint_angles' / 'joint_angles_summary.html'}")
 




