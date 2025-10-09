import numpy as np
import matplotlib.pyplot as plt

def plot_gait_events_over_time(
    q_hs:np.ndarray, 
    q_to:np.ndarray, 
    fmc_hs:np.ndarray, 
    fmc_to:np.ndarray,
    sampling_rate:float,
    title="Gait events over time (one foot)",
    xlim=None,
    separation=0.03,
):
    """
    Plot Qualisys and FreeMoCap gait events grouped by event type.
    
    Heel strikes are plotted around y=1, toe offs around y=0.
    Qualisys markers are placed slightly above FreeMoCap markers for each event type.
    
    Parameters
    ----------
    q_hs, q_to : array-like
        Qualisys heel strike and toe off times
    fmc_hs, fmc_to : array-like
        FreeMoCap heel strike and toe off times
    title : str
        Plot title
    xlim : tuple, optional
        X-axis limits (min, max)
    separation : float
        Vertical separation between Qualisys and FreeMoCap markers
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    frame_interval = 1 / sampling_rate
    q_hs, q_to = np.asarray(q_hs), np.asarray(q_to)
    fmc_hs, fmc_to = np.asarray(fmc_hs), np.asarray(fmc_to)

    fig, ax = plt.subplots(figsize=(15, 6))

    # Colors per system
    qual_color = "#d62728"  # blue
    fmc_color  = "#1f77b4"  # orange

    # Y-positions: group by event type with small separation between systems
    # Heel strikes around y=1
    q_hs_y = np.full(len(q_hs), 0 + separation/2)      # Qualisys HS slightly above 1
    fmc_hs_y = np.full(len(fmc_hs), 0 - separation/2)  # FreeMoCap HS slightly below 1

    # Toe offs around y=0
    q_to_y = np.full(len(q_to), .3 + separation/2)      # Qualisys TO slightly above 0
    fmc_to_y = np.full(len(fmc_to), .3 - separation/2)  # FreeMoCap TO slightly below 0

    # Plot heel strikes (x markers)
    ax.scatter(q_hs*frame_interval, q_hs_y, marker='x', s=60, label="Qualisys HS", c=qual_color)
    ax.scatter(fmc_hs*frame_interval, fmc_hs_y, marker='x', s=60, label="FreeMoCap HS", c=fmc_color)

    # Plot toe offs (circle markers)
    ax.scatter(q_to*frame_interval, q_to_y, marker='o', s=60, label="Qualisys TO", 
               c=qual_color, facecolors='none', edgecolors=qual_color)
    ax.scatter(fmc_to*frame_interval, fmc_to_y, marker='o', s=60, label="FreeMoCap TO", 
               c=fmc_color, facecolors='none', edgecolors=fmc_color)

    # Guide lines at the center of each event group
    ax.axhline(.5, color='gray', lw=0.8, alpha=0.3, zorder=0)
    ax.axhline(0, color='gray', lw=0.8, alpha=0.3, zorder=0)

    # Set y-axis with proper labels
    ax.set_yticks([.3, 0])
    ax.set_yticklabels(["Heel Strike", "Toe Off"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    
    # Legend with 2 columns
    ax.legend(ncols=2, frameon=False, loc='upper right')
    
    # Set y-limits with padding
    ax.set_ylim(-0.3, .6)
    
    # Apply x-limits if provided
    if xlim: 
        ax.set_xlim(*xlim)
    
    # Optional: add subtle grid
    ax.grid(True, axis='x', alpha=0.2)
    
    fig.tight_layout()
    return fig, ax
