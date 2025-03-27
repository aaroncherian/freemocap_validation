from skellymodels.experimental.model_redo.managers.human import Human
from validation.utils.rotation import run_skellyforge_rotation
from validation.steps.temporal_alignment.core.lag_calculation import LagCalculatorComponent,LagCalculator
from validation.steps.temporal_alignment.core.qualisys_processing import QualisysMarkerData, QualisysJointCenterData, DataResampler
from validation.steps.temporal_alignment.core.markersets.full_body_weights import joint_center_weights
import pandas as pd
import numpy as np

class TemporalSyncManager:
    def __init__(self, freemocap_model: Human,
                 freemocap_timestamps: pd.DataFrame,
                 qualisys_marker_data: pd.DataFrame,
                 qualisys_unix_start_time:float):
        
        self.freemocap_model = freemocap_model
        self.freemocap_timestamps, self.framerate = self._get_timestamps(freemocap_timestamps)
        self.qualisys_marker_data = qualisys_marker_data
        self.qualisys_unix_start_time =qualisys_unix_start_time
        self.run()

    def run(self):
        self._process_freemocap_data()
        self._process_qualisys_data()

        qualisys_component = self._create_qualisys_component(lag_in_seconds=0)
        initial_lag = self._calculate_lag(qualisys_component)

        corrected_qualisys_component = self._create_qualisys_component(lag_in_seconds=initial_lag)
        final_lag = self._calculate_lag(corrected_qualisys_component)

        print('Initial lag:', initial_lag)
        print('Final lag:', final_lag)

        return self.freemocap_lag_component, corrected_qualisys_component, qualisys_component

    def _process_freemocap_data(self):
        freemocap_data = self.freemocap_model.body.trajectories['3d_xyz'].as_numpy
        landmark_names = self.freemocap_model.body.trajectories['3d_xyz'].landmark_names
        origin_aligned_freemocap_data = run_skellyforge_rotation(raw_skeleton_data=freemocap_data,
                                                                 landmark_names=landmark_names)
        self.freemocap_lag_component = LagCalculatorComponent(
            joint_center_array=origin_aligned_freemocap_data,
            list_of_joint_center_names=landmark_names
        )

    def _process_qualisys_data(self):
        self.qualisys_marker_data_holder = QualisysMarkerData(
            marker_dataframe=self.qualisys_marker_data,
            unix_start_time=self.qualisys_unix_start_time
        )

        self.qualisys_joint_center_data_holder = QualisysJointCenterData(
            marker_data_holder=self.qualisys_marker_data_holder,
            weights=joint_center_weights
        )


    def _calculate_lag(self, qualisys_lag_component: LagCalculatorComponent):
        lag_corrector = LagCalculator(
            freemocap_component=self.freemocap_lag_component, 
            qualisys_component=qualisys_lag_component, 
            framerate=self.framerate)
        
        lag_corrector.run()
        print('Median lag:', lag_corrector.median_lag)
        print('Lag in seconds:', lag_corrector.get_lag_in_seconds())
        return lag_corrector.get_lag_in_seconds()
    f = 2
        
        
    def _extract_marker_data(self) -> pd.DataFrame:
        """Extract only marker data columns."""

        columns_of_interest = self.qualisys_marker_data.columns[
            ~self.qualisys_marker_data.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]
        return self.qualisys_marker_data[columns_of_interest]


    def _get_timestamps(self, freemocap_timestamps):
        timestamps = freemocap_timestamps['timestamp_unix_seconds']
        time_diff = np.diff(timestamps)
        framerate = 1 / np.nanmean(time_diff)
        print(f"Calculated FreeMoCap framerate: {framerate}")
        return timestamps, framerate
        f = 2

    def _create_qualisys_component(self, lag_in_seconds:float = 0) -> LagCalculatorComponent: 
        joint_center_names = list(joint_center_weights.keys())
        df = self.qualisys_joint_center_data_holder.as_dataframe_with_unix_timestamps(lag_seconds=lag_in_seconds)
        resampler = DataResampler(df, self.freemocap_timestamps)
        resampler.resample()
        self.resampled_qualisys_joint_center_data = resampler.as_dataframe
        return LagCalculatorComponent(
            joint_center_array=resampler.rotated_resampled_marker_array(joint_center_names),
            list_of_joint_center_names=joint_center_names
        )



if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    from validation.steps.temporal_alignment.components import get_component


    path_to_recording = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking")
    path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

    data = np.load(path_to_data)

    human = Human.from_numpy_array(name = 'human',
                                   model_info=MediapipeModelInfo(),
                                   tracked_points_numpy_array=data)
    
    freemocap_timestamps = get_component(key = 'freemocap_timestamps')

    qualisys_markers = get_component(key = 'qualisys_markers')
    qualisys_loaded = qualisys_markers.load(base_dir=path_to_recording)
    qualisys_df = qualisys_loaded.dataframe
    qualisys_unix_start_time = qualisys_loaded.unix_start_time
    
    temp_manager = TemporalSyncManager(freemocap_model = human,
                                       freemocap_timestamps=freemocap_timestamps.load(base_dir= path_to_recording),
                                       qualisys_marker_data = qualisys_df,
                                       qualisys_unix_start_time = qualisys_unix_start_time)

    
    f = 2