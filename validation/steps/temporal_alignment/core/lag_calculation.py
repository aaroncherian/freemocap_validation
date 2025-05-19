import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List

@dataclass
class LagCalculatorComponent:
    joint_center_array: np.ndarray
    list_of_joint_center_names: list

    def __post_init__(self):
        if self.joint_center_array.shape[1] != len(self.list_of_joint_center_names):
            raise ValueError(f"Number of joint centers: {self.joint_center_array.shape} must match the number of joint center names: {len(self.list_of_joint_center_names)}")


def plot_radius_of_gyration(freemocap_rg, qualisys_rg, framerate):
    time_fmc = np.arange(len(freemocap_rg)) / framerate
    time_qual = np.arange(len(qualisys_rg)) / framerate

    plt.figure(figsize=(10, 4))
    plt.plot(time_fmc, freemocap_rg, label="FreeMoCap", alpha=0.8)
    plt.plot(time_qual, qualisys_rg, label="Qualisys", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Radius of Gyration")
    plt.title("Radius of Gyration Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

def pairwise_pca_signal(joint_centers_array: np.ndarray, n_components=1):
    """
    joint_centers_array: (frames, markers, 3)
    Returns: (frames,) first PCA component of the pairwise distances
    """
    pairwise_vectors = []

    for frame in range(joint_centers_array.shape[0]):
        frame_data = joint_centers_array[frame]
        if np.isnan(frame_data).all():
            pairwise_vectors.append(np.full((joint_centers_array.shape[1] * (joint_centers_array.shape[1] - 1)) // 2, np.nan))
            continue
        # Flattened upper-triangular of pairwise distance matrix
        pairwise_vec = pdist(frame_data, metric='euclidean')  # (N(N-1)/2,)
        pairwise_vectors.append(pairwise_vec)

    D = np.array(pairwise_vectors)  # shape (frames, pairwise_distances)

    # Remove frames with NaNs
    valid_mask = ~np.isnan(D).any(axis=1)
    D_valid = D[valid_mask]

    # PCA
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(D_valid)[:, 0]

    # Reinsert NaNs where frames were dropped
    full_proj = np.full(D.shape[0], np.nan)
    full_proj[valid_mask] = proj

    return full_proj
from scipy.signal import butter, filtfilt

def lowpass(sig, fs, cutoff_hz=2.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype='low', analog=False)
    return filtfilt(b, a, sig)  

import matplotlib.pyplot as plt
from scipy.signal import correlate

def show_xcorr(a, b, fps, max_lag_s=3):
    a = (a - np.nanmean(a)) / np.nanstd(a)
    b = (b - np.nanmean(b)) / np.nanstd(b)

    max_lag = int(max_lag_s * fps)
    cc = correlate(a, b, mode='full')
    lags = np.arange(-len(a)+1, len(a))
    centre = len(cc) // 2
    cc = cc[centre-max_lag : centre+max_lag+1]
    lags = lags[centre-max_lag : centre+max_lag+1]

    plt.figure(figsize=(6,3))
    plt.plot(lags / fps, cc)
    plt.axvline(0, c='k', lw=0.5)
    plt.xlabel('Lag (s)'); plt.ylabel('Cross-corr')
    plt.title('Raw (time-domain) x-corr')
    plt.tight_layout(); plt.show()


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

        qualisys_joint_to_centroid = self.calculate_jointwise_distances_to_centroid(self.qualisys_component.joint_center_array)
        freemocap_joint_to_centroid = self.calculate_jointwise_distances_to_centroid(self.freemocap_component.joint_center_array)
        
        optimal_lag_list = self.calculate_lag_for_common_joints(
            freemocap_joint_centers_array = freemocap_joint_to_centroid,
            qualisys_joint_centers_array=qualisys_joint_to_centroid,
            freemoocap_joint_centers_names=self.freemocap_component.list_of_joint_center_names,
            qualisys_joint_centers_names=self.qualisys_component.list_of_joint_center_names,
            common_joint_centers=common_joint_center_names
        )
        
        return optimal_lag_list
    
    def calculate_jointwise_distances_to_centroid(self, joint_centers_array: np.ndarray):
        """
        Returns: (frames, joints) array of distances from each joint to the frame's centroid.
        """
        centroid = np.nanmean(joint_centers_array, axis=1, keepdims=True)   # (frames, 1, 3)
        diffs = joint_centers_array - centroid                              # (frames, joints, 3)
        dists = np.linalg.norm(diffs, axis=2)                               # (frames, joints)
        return dists

    def get_common_joint_center_names(self, freemocap_joint_center_names, qualisys_joint_center_names):
        return list(set(freemocap_joint_center_names) & set(qualisys_joint_center_names))
        from scipy.signal import correlate

    def calculate_lag_for_common_joints(
        self,
        freemocap_joint_centers_array: np.ndarray,  # shape: (frames, joints)
        qualisys_joint_centers_array: np.ndarray,   # shape: (frames, joints)
        freemoocap_joint_centers_names: List[str],
        qualisys_joint_centers_names: List[str],
        common_joint_centers: List[str],
        window_size: int = 300,
        stride: int = 100,
        max_lag: int = 200
    ) -> List[int]:
        from collections import Counter

        def normalize(sig):
            return (sig - np.nanmean(sig)) / (np.nanstd(sig) + 1e-8)

        def lowpass(signal, fs, cutoff=14.0, order=4):
            nyq = 0.5 * fs
            b, a = butter(order, cutoff / nyq, btype='low')
            return filtfilt(b, a, signal)

        def xcorr_lag(sig_a, sig_b, max_lag):
            corr = correlate(sig_a, sig_b, mode='full')
            lags = np.arange(-len(sig_a) + 1, len(sig_a))
            valid = (lags >= -max_lag) & (lags <= max_lag)
            return lags[valid][np.argmax(corr[valid])]

        fmc_index_map = {name: i for i, name in enumerate(freemoocap_joint_centers_names)}
        qls_index_map = {name: i for i, name in enumerate(qualisys_joint_centers_names)}

        optimal_lags = []

        for joint_name in common_joint_centers:
            f_idx = fmc_index_map[joint_name]
            q_idx = qls_index_map[joint_name]

            f_sig = normalize(lowpass(freemocap_joint_centers_array[:, f_idx], self.framerate))
            q_sig = normalize(lowpass(qualisys_joint_centers_array[:, q_idx], self.framerate))

            min_len = min(len(f_sig), len(q_sig))
            f_sig, q_sig = f_sig[:min_len], q_sig[:min_len]

            lags = []
            for start in range(0, min_len - window_size + 1, stride):
                f_win = f_sig[start:start + window_size]
                q_win = q_sig[start:start + window_size]

                if np.std(f_win) < 1e-5 or np.std(q_win) < 1e-5:
                    continue  # skip flat windows

                lag = xcorr_lag(f_win, q_win, max_lag=max_lag)
                lags.append(lag)

            if lags:
                best_lag = Counter(lags).most_common(1)[0][0]
            else:
                best_lag = 0  # fallback

            optimal_lags.append(best_lag)

        self.optimal_lags = optimal_lags
        return optimal_lags

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
            # plot_radius_of_gyration(normalized_freemocap, normalized_qualisys, self.framerate)

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
