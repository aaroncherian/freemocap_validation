import numpy as np
from dataclasses import dataclass

@dataclass
class LagCalculatorComponent:
    joint_center_array: np.ndarray
    list_of_joint_center_names: list

    def __post_init__(self):
        if self.joint_center_array.shape[1] != len(self.list_of_joint_center_names):
            raise ValueError(f"Number of joint centers: {self.joint_center_array.shape} must match the number of joint center names: {len(self.list_of_joint_center_names)}")
        

class LagCalculator:
    def __init__(self, freemocap_component: LagCalculatorComponent, qualisys_component: LagCalculatorComponent, framerate: float):
        self.freemocap_component = freemocap_component
        self.qualisys_component = qualisys_component
        self.framerate = framerate

    def run(self):  
        common_joint_center_names = self.get_common_joint_center_names(
            self.freemocap_component.list_of_joint_center_names,
            self.qualisys_component.list_of_joint_center_names
        )

        optimal_lag_list = self.calculate_lag_for_common_joints(
            freemocap_joint_centers_array=self.freemocap_component.joint_center_array,
            qualisys_joint_centers_array=self.qualisys_component.joint_center_array,
            freemoocap_joint_centers_names=self.freemocap_component.list_of_joint_center_names,
            qualisys_joint_centers_names=self.qualisys_component.list_of_joint_center_names,
            common_joint_centers=common_joint_center_names
        )
        
        return optimal_lag_list


    def get_common_joint_center_names(self, freemocap_joint_center_names, qualisys_joint_center_names):
        return list(set(freemocap_joint_center_names) & set(qualisys_joint_center_names))
    
    def calculate_lag_for_common_joints(self, 
                                        freemocap_joint_centers_array:np.ndarray, 
                                        qualisys_joint_centers_array:np.ndarray, 
                                        freemoocap_joint_centers_names:list, 
                                        qualisys_joint_centers_names:list, 
                                        common_joint_centers:list):
        
        optimal_lag_list = []

        for joint_center in common_joint_centers:
            qualisys_joint_idx = qualisys_joint_centers_names.index(joint_center)
            freemocap_joint_idx = freemoocap_joint_centers_names.index(joint_center)

            qualisys_joint_data = qualisys_joint_centers_array[:,qualisys_joint_idx,:]
            freemocap_joint_data = freemocap_joint_centers_array[:,freemocap_joint_idx,:]

            lags_for_joint = self.calculate_lag(freemocap_joint_data, qualisys_joint_data)
            print(f"Lags per dimension for joint center {joint_center}: {lags_for_joint}")
            optimal_lag_list.append(lags_for_joint) 

        return optimal_lag_list

    def calculate_lag(self,
                      freemocap_joint_centers_array:np.ndarray,
                      qualisys_joint_centers_array:np.ndarray):
        """
        Calculate the optimal lag for a single marker across all three dimensions (X, Y, Z).

        Parameters:
            freemocap_joint_centers_array (np.ndarray): FreeMoCap data of shape (frames, 1, 3) for a single marker.
            qualisys_joint_centers_array (np.ndarray): Qualisys data of shape (frames, 1, 3) for a single marker.

        Returns:
            np.ndarray: Optimal lags for each dimension (X, Y, Z).
        """

        optimal_lags = []
        for dim in range(3):  # Loop over X, Y, Z
            freemocap_dim = freemocap_joint_centers_array[:, dim]
            qualisys_dim = qualisys_joint_centers_array[:, dim]

            # Ensure the signals are the same length
            min_length = min(len(freemocap_dim), len(qualisys_dim))
            freemocap_dim = freemocap_dim[:min_length]
            qualisys_dim = qualisys_dim[:min_length]

            # Normalize the data
            normalized_freemocap = self.normalize(freemocap_dim)
            normalized_qualisys = self.normalize(qualisys_dim)

            # Compute cross-correlation
            cross_corr = np.correlate(normalized_freemocap, normalized_qualisys, mode='full')

            # Find the lag that maximizes the cross-correlation
            optimal_lag = np.argmax(cross_corr) - (len(normalized_qualisys) - 1)
            optimal_lags.append(optimal_lag)

            self.optimal_lags = optimal_lags
        return np.array(optimal_lags)


    def normalize(self, 
                  signal: np.ndarray) -> np.ndarray:
            """
            Normalize a signal to have zero mean and unit variance.
            
            Parameters:
                signal (np.ndarray): The signal to normalize.

            Returns:
                np.ndarray: The normalized signal.
            """
            return (signal - signal.mean()) / signal.std()

    @property
    def median_lag(self):
        return int(np.median(self.optimal_lags))
    
    def get_lag_in_seconds(self, lag=None):
        """Calculate lag in seconds using median lag by default."""
        if lag is None:
            lag = self.median_lag
        return lag / self.framerate
