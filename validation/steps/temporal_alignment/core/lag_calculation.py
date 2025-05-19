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
    def __init__(self, freemocap_component: LagCalculatorComponent, 
                 qualisys_component: LagCalculatorComponent, 
                 framerate: float,
                 start_frame:int,
                 end_frame:int):
        self.freemocap_component = freemocap_component
        self.qualisys_component = qualisys_component
        self.framerate = framerate
        self.start = start_frame
        self.end = end_frame

    def run(self):


        optimal_lag_list = self.run_pca_based_sync()
        self.optimal_lags = optimal_lag_list
        # For PCA, use weighted consensus based on explained variance
        self._median_lag = self.compute_pca_consensus_lag()
        
        return optimal_lag_list
    

    def get_common_joint_center_names(self, freemocap_joint_center_names, qualisys_joint_center_names):
        return list(set(freemocap_joint_center_names) & set(qualisys_joint_center_names))


    def run_pca_based_sync(self):
        """
        Run PCA-based synchronization as an alternative to centroid-based method.
        Returns a list of optimal lags just like calculate_lag_for_common_joints.
        """
        # Get raw joint data for both systems
        freemocap_data = self.freemocap_component.joint_center_array  # (frames, joints, 3)
        qualisys_data = self.qualisys_component.joint_center_array    # (frames, joints, 3)
        
        # Get common joint names and indices
        common_joint_center_names = self.get_common_joint_center_names(
            self.freemocap_component.list_of_joint_center_names,
            self.qualisys_component.list_of_joint_center_names
        )
        
        fmc_indices = [self.freemocap_component.list_of_joint_center_names.index(name) 
                    for name in common_joint_center_names]
        qls_indices = [self.qualisys_component.list_of_joint_center_names.index(name) 
                    for name in common_joint_center_names]
        
        # Get data for common joints only
        fmc_data = freemocap_data[self.start:self.end, fmc_indices, :]  # (frames, common_joints, 3)
        qls_data = qualisys_data[self.start:self.end, qls_indices, :]   # (frames, common_joints, 3)
        
        # Plot original data for selected joints for visual comparison
        self._plot_joint_trajectories(fmc_data, qls_data, common_joint_center_names)
        
        # Reshape for PCA: (frames, joints*3)
        fmc_frames, fmc_joints, _ = fmc_data.shape
        qls_frames, qls_joints, _ = qls_data.shape
        
        fmc_reshaped = fmc_data.reshape(fmc_frames, -1)  # (frames, joints*3)
        qls_reshaped = qls_data.reshape(qls_frames, -1)  # (frames, joints*3)
        
        # Handle NaNs by filling with column means
        for col in range(fmc_reshaped.shape[1]):
            mask = np.isnan(fmc_reshaped[:, col])
            if mask.any():
                col_mean = np.nanmean(fmc_reshaped[:, col])
                fmc_reshaped[mask, col] = col_mean if not np.isnan(col_mean) else 0
        
        for col in range(qls_reshaped.shape[1]):
            mask = np.isnan(qls_reshaped[:, col])
            if mask.any():
                col_mean = np.nanmean(qls_reshaped[:, col])
                qls_reshaped[mask, col] = col_mean if not np.isnan(col_mean) else 0
        
        # Apply PCA to both datasets
        n_components = min(3, min(fmc_reshaped.shape[0], qls_reshaped.shape[0], fmc_reshaped.shape[1]) - 1)
        
        fmc_pca = PCA(n_components=n_components)
        qls_pca = PCA(n_components=n_components)
        
        fmc_components = fmc_pca.fit_transform(fmc_reshaped)
        qls_components = qls_pca.fit_transform(qls_reshaped)
        
        # Print explained variance to understand data quality
        print(f"FreeMoCap explained variance: {fmc_pca.explained_variance_ratio_}")
        print(f"Qualisys explained variance: {qls_pca.explained_variance_ratio_}")
        
        # Plot PCA components
        self._plot_pca_components(fmc_components, qls_components, fmc_pca.explained_variance_ratio_)
        
        # Store component variance ratios for weighted consensus
        self._pca_variance_ratios = fmc_pca.explained_variance_ratio_
        
        # Cross-correlate each component pair
        optimal_lags = []
        max_lag_frames = int(3.0 * self.framerate)  # 3 second max lag
        correlation_strengths = []  # Track correlation strength for each component
        
        for i in range(n_components):
            fmc_comp = self.normalize(fmc_components[:, i])
            qls_comp = self.normalize(qls_components[:, i])
            
            # Apply low-pass filter to reduce noise
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 10 / (self.framerate/2), 'low')
            fmc_comp_filtered = filtfilt(b, a, fmc_comp)
            qls_comp_filtered = filtfilt(b, a, qls_comp)
            
            # Ensure signals are same length
            min_length = min(len(fmc_comp_filtered), len(qls_comp_filtered))
            fmc_comp_filtered = fmc_comp_filtered[:min_length]
            qls_comp_filtered = qls_comp_filtered[:min_length]
            
            # Check correlation direction
            direct_corr = np.corrcoef(fmc_comp_filtered[:min_length], qls_comp_filtered[:min_length])[0, 1]
            
            # If signals are negatively correlated, flip one for correlation calculation
            use_flipped = False
            if direct_corr < 0:
                print(f"PC{i+1}: Detected negative correlation ({direct_corr:.2f}), flipping signal for calculation")
                qls_comp_flipped = -qls_comp_filtered
                use_flipped = True
            else:
                qls_comp_flipped = qls_comp_filtered
            
            # Compute cross-correlation with appropriate signal (original or flipped)
            if use_flipped:
                cross_corr = correlate(fmc_comp_filtered, qls_comp_flipped, mode='full')
            else:
                cross_corr = correlate(fmc_comp_filtered, qls_comp_filtered, mode='full')
                
            lags_frames = np.arange(-len(fmc_comp_filtered)+1, len(fmc_comp_filtered))
            
            # Find lag with max correlation within reasonable range
            mask = (lags_frames >= -max_lag_frames) & (lags_frames <= max_lag_frames)
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) > 0:
                max_corr_idx = np.argmax(np.abs(cross_corr[valid_indices]))
                max_idx = valid_indices[max_corr_idx]
                lag = lags_frames[max_idx]
                max_corr_value = cross_corr[max_idx]
                
                # Store the lag and correlation strength
                optimal_lags.append(lag)
                correlation_strengths.append(np.abs(max_corr_value))
                
                print(f"PC{i+1}: Lag = {lag} frames ({lag/self.framerate:.3f} s), " 
                    f"Correlation = {max_corr_value:.2f}, "
                    f"Variance = {fmc_pca.explained_variance_ratio_[i]:.2%}")
                
                # Visualize correlation and lag for each component
                # Use original signals for visualization, but indicate if flipping was used
                self._plot_component_correlation_with_phase(
                    fmc_comp_filtered, qls_comp_filtered, qls_comp_flipped if use_flipped else None,
                    cross_corr[valid_indices], lags_frames[valid_indices], 
                    lag, i, fmc_pca.explained_variance_ratio_[i], use_flipped)
        
        # Store lags and correlation strengths
        self.optimal_lags = optimal_lags
        return optimal_lags
    def _plot_component_correlation_with_phase(self, fmc_comp, qls_comp, qls_comp_flipped, 
                                            cross_corr, lags, best_lag, comp_idx, 
                                            explained_variance, used_flip=False):
        """Plot component signals and their cross-correlation with best lag, showing phase handling."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Component time series
        plt.subplot(2, 1, 1)
        t = np.arange(len(fmc_comp)) / self.framerate
        plt.plot(fmc_comp, 'b-', label='FreeMoCap')
        plt.plot(qls_comp, 'r-', label='Qualisys')
        
        if qls_comp_flipped is not None:
            plt.plot(qls_comp_flipped, 'g--', label='Qualisys (flipped for calculation)', alpha=0.5)
        
        plt.legend()
        phase_info = " [Phase-inverted]" if used_flip else ""
        plt.title(f"PC{comp_idx+1} Time Series{phase_info} (Explained Variance: {explained_variance:.2%})")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Value")
        
        # Plot 2: Cross-correlation
        plt.subplot(2, 1, 2)
        lag_times = lags 
        plt.plot(lag_times, cross_corr)
        plt.axvline(best_lag/self.framerate, color='r', linestyle='--', 
                    label=f'Best lag: {best_lag/self.framerate:.3f}s')
        corr_method = "Phase-aware correlation" if used_flip else "Standard correlation"
        plt.title(f"Cross-correlation for PC{comp_idx+1} ({corr_method})")
        plt.xlabel("Lag (s)")
        plt.ylabel("Correlation")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def _plot_joint_trajectories(self, fmc_data, qls_data, joint_names, max_joints=3):
        """Plot sample joint trajectories to visualize the data quality and alignment."""
        import matplotlib.pyplot as plt
        
        # Select a few joints to plot
        plot_joints = min(max_joints, len(joint_names))
        
        plt.figure(figsize=(12, 4 * plot_joints))
        
        for i in range(plot_joints):
            # Plot X, Y, Z for one joint
            for dim, dim_name in enumerate(['X', 'Y', 'Z']):
                plt.subplot(plot_joints, 3, i*3 + dim + 1)
                
                # Time vectors
                time_fmc = np.arange(fmc_data.shape[0]) / self.framerate
                time_qls = np.arange(qls_data.shape[0]) / self.framerate
                
                # Plot trajectories
                plt.plot(time_fmc, fmc_data[:, i, dim], 'b-', alpha=0.7, label='FreeMoCap')
                plt.plot(time_qls, qls_data[:, i, dim], 'r-', alpha=0.7, label='Qualisys')
                
                plt.title(f"{joint_names[i]} - {dim_name}")
                plt.xlabel("Time (s)")
                plt.ylabel("Position")
                if i == 0 and dim == 0:
                    plt.legend()
        
        plt.tight_layout()
        plt.show()

    def _plot_pca_components(self, fmc_components, qls_components, explained_variance):
        """Plot PCA components from both systems to visually compare them."""
        import matplotlib.pyplot as plt
        
        n_components = min(3, fmc_components.shape[1])
        plt.figure(figsize=(12, 4 * n_components))
        
        for i in range(n_components):
            plt.subplot(n_components, 1, i+1)
            
            # Time vectors
            time_fmc = np.arange(fmc_components.shape[0]) / self.framerate
            time_qls = np.arange(qls_components.shape[0]) / self.framerate
            
            # Normalize for better visual comparison
            fmc_comp = self.normalize(fmc_components[:, i])
            qls_comp = self.normalize(qls_components[:, i])
            
            plt.plot(time_fmc, fmc_comp, 'b-', alpha=0.7, label='FreeMoCap')
            plt.plot(time_qls, qls_comp, 'r-', alpha=0.7, label='Qualisys')
            
            plt.title(f"PC{i+1} (Explained Variance: {explained_variance[i]:.2%})")
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Component Value")
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def compute_pca_consensus_lag(self):
        """
        Compute a weighted consensus lag based on PCA component correlations.
        Weights are based on explained variance of each component.
        """
        if not hasattr(self, 'optimal_lags') or not self.optimal_lags:
            return 0
        
        # If we only have one lag, return it
        if len(self.optimal_lags) == 1:
            return self.optimal_lags[0]
        
        # Get explained variance ratios if available
        if hasattr(self, '_pca_variance_ratios'):
            weights = self._pca_variance_ratios[:len(self.optimal_lags)]
        else:
            # Default to equal weights
            weights = np.ones(len(self.optimal_lags)) / len(self.optimal_lags)
        
        # Compute weighted average
        weighted_lag = np.sum(np.array(self.optimal_lags) * weights) / np.sum(weights)
        
        # Convert to integer lag
        return int(np.round(weighted_lag))
    
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
