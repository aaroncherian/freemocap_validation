from validation.steps.temporal_alignment.components import REQUIRES, PRODUCES
from validation.components import QUALISYS_MARKERS,QUALISYS_START_TIME, FREEMOCAP_TIMESTAMPS, QUALISYS_SYNCED_JOINT_CENTERS, QUALISYS_COM, FREEMOCAP_PRE_SYNC_JOINT_CENTERS
#revisit whether this import implementation above is worth it
from validation.steps.temporal_alignment.visualize import SynchronizationVisualizer
from validation.steps.temporal_alignment.core.temporal_synchronizer import TemporalSyncManager
from validation.steps.temporal_alignment.config import TemporalAlignmentConfig
from validation.utils.actor_utils import make_freemocap_actor_from_tracked_points, make_qualisys_actor
from skellymodels.experimental.model_redo.managers.human import Human
from validation.pipeline.base import ValidationStep
from pathlib import Path
from nicegui import ui

# from skellymodels.experimental.model_redo.tracker_info.model_info importModelInfo


class TemporalAlignmentStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = TemporalAlignmentConfig

    def calculate(self):
        self.logger.info("Starting temporal alignment")

        freemocap_timestamps   = self.data[FREEMOCAP_TIMESTAMPS.name]
        qualisys_dataframe = self.data[QUALISYS_MARKERS.name]
        qualisys_unix_start_time = self.data[QUALISYS_START_TIME.name]
        freemocap_joint_centers = self.data[FREEMOCAP_PRE_SYNC_JOINT_CENTERS.name]

        freemocap_actor = make_freemocap_actor_from_tracked_points(project_config=self.ctx.project_config,
                                               tracked_points_data=freemocap_joint_centers)
        

        manager = TemporalSyncManager(freemocap_model = freemocap_actor,
                                freemocap_timestamps= freemocap_timestamps,
                                qualisys_marker_data = qualisys_dataframe,
                                qualisys_unix_start_time = qualisys_unix_start_time,
                                start_frame = self.cfg.start_frame,
                                end_frame = self.cfg.end_frame,)
        
        (self.freemocap_lag_component,
        self.qualisys_synced_lag_component,
        self.qualisys_original_lag_component,
        ) = manager.run()

        qualisys_actor = make_qualisys_actor(project_config=self.ctx.project_config,
                                             tracked_points_data=self.qualisys_synced_lag_component.joint_center_array)

        qualisys_actor.calculate()

        self.outputs[QUALISYS_SYNCED_JOINT_CENTERS.name] = self.qualisys_synced_lag_component.joint_center_array
        self.outputs[QUALISYS_COM.name] = qualisys_actor.body.trajectories['total_body_com'].as_numpy




    def visualize(self, window_size: int = 300):
        import numpy as np
        import plotly.graph_objects as go
        joint_names = list(
            set(self.freemocap_lag_component.list_of_joint_center_names)
            & set(self.qualisys_original_lag_component.list_of_joint_center_names)
            & set(self.qualisys_synced_lag_component.list_of_joint_center_names)
        )

        traces = []
        buttons = []
        time_cache = {}

        for joint_idx, joint_name in enumerate(joint_names):
            # Get joint indices
            f_idx = self.freemocap_lag_component.list_of_joint_center_names.index(joint_name)
            o_idx = self.qualisys_original_lag_component.list_of_joint_center_names.index(joint_name)
            c_idx = self.qualisys_synced_lag_component.list_of_joint_center_names.index(joint_name)

            # Extract data
            fmc = self.freemocap_lag_component.joint_center_array[:, f_idx, :]
            orig = self.qualisys_original_lag_component.joint_center_array[:, o_idx, :]
            corr = self.qualisys_synced_lag_component.joint_center_array[:, c_idx, :]

            min_len = min(len(fmc), len(orig), len(corr))
            fmc, orig, corr = fmc[:min_len], orig[:min_len], corr[:min_len]

            # Pick window
            start = 0
            if min_len > window_size:
                vel = np.sum(np.abs(np.diff(fmc, axis=0)), axis=1)
                smoothed = np.convolve(vel, np.ones(30)/30, mode='same')
                start = max(0, np.argmax(smoothed) - window_size // 2)
            end = min(start + window_size, min_len)
            time = np.arange(start, end)
            time_cache[joint_name] = time

            data = {
                "FreeMoCap": (fmc[start:end] - np.nanmean(fmc[start:end], axis=0)) / np.nanstd(fmc[start:end], axis=0),
                "Original Qualisys": (orig[start:end] - np.nanmean(orig[start:end], axis=0)) / np.nanstd(orig[start:end], axis=0),
                "Corrected Qualisys": (corr[start:end] - np.nanmean(corr[start:end], axis=0)) / np.nanstd(corr[start:end], axis=0)
            }

            for dim, axis in enumerate(["X", "Y", "Z"]):
                for label, color in zip(data, ['blue', 'red', 'green']):
                    trace = go.Scatter(
                        x=time,
                        y=data[label][:, dim],
                        name=f"{label} ({axis})",
                        visible=(joint_idx == 0),  # only show first joint initially
                        line=dict(color=color),
                        legendgroup=label + axis,
                        showlegend=(dim == 0)  # only show legend once
                    )
                    traces.append(trace)

            # Button for this joint
            n_traces_per_joint = 9  # 3 systems × 3 dimensions
            visibility = [False] * len(joint_names) * n_traces_per_joint
            base = joint_idx * n_traces_per_joint
            for i in range(n_traces_per_joint):
                visibility[base + i] = True
            buttons.append(dict(
                label=joint_name,
                method='update',
                args=[{'visible': visibility},
                    {'title': f"Trajectories for Joint: {joint_name}"}]
            ))

        fig = go.Figure(data=traces)

        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )],
            height=700,
            title=f"Trajectories for Joint: {joint_names[0]}",
            template="plotly_white",
            margin=dict(l=40, r=40, t=80, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.show()

if __name__ in {"__main__", "__mp_main__"}:
    
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    import numpy as np
    from validation.pipeline.base import PipelineContext
    import logging 

    logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(message)s')

    path_to_recording = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2")
    path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

    ctx = PipelineContext(recording_dir=path_to_recording)
    ctx.put(FREEMOCAP_TIMESTAMPS.name,
            FREEMOCAP_TIMESTAMPS.load(path_to_recording))
    ctx.put(QUALISYS_MARKERS.name,
            QUALISYS_MARKERS.load(path_to_recording))
    ctx.put(QUALISYS_START_TIME.name,
            QUALISYS_START_TIME.load(path_to_recording))

    data = np.load(path_to_data)
    human = Human.from_numpy_array(name = 'human',
                                    model_info=MediapipeModelInfo(),
                                    tracked_points_numpy_array=data)
    
    ctx.put("freemocap_actor", human)

    step = TemporalAlignmentStep(ctx, logger=logging.getLogger('temporal_synchronization'))
    step.calculate()
    step.visualize()
    step.store()

    ui.run()

    f = 2