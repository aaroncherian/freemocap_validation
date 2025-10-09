
import numpy as np
import pandas as pd

def normalize_gait_cycle(gait_cycle_data:np.ndarray, n_points: int) -> np.ndarray:
    num_frames = gait_cycle_data.shape[0]
    
    x = np.linspace(0, 1.0, num=num_frames)
    x_new = np.linspace(0, 1.0, num=n_points)

    result = np.empty((n_points,3), dtype = float)
    for dimension in range(3):
        result[:,dimension] = np.interp(x_new, x, gait_cycle_data[:,dimension])
    
    return result


def get_cycles_for_tracker(trajectory_dict:dict[str, np.ndarray],
                                marker_list: list[str],
                                gait_events: dict[str, list[slice]],
                                tracker:str,
                                n_points:int = 100) -> pd.DataFrame:
    dfs = []
    for side, slices in gait_events.items():
        for marker in marker_list:
            m = f"{side}_{marker}"

            gait_cycles = np.stack([normalize_gait_cycle(trajectory_dict[m][s], n_points=n_points) for s in slices])

            num_cycles = gait_cycles.shape[0]
            num_points = gait_cycles.shape[1]

            x = gait_cycles[...,0].ravel()
            y = gait_cycles[...,1].ravel()
            z = gait_cycles[...,2].ravel()

            cycle_count = np.repeat(np.arange(1, num_cycles+1), num_points)
            cycle_percent = np.tile(np.arange(num_points), num_cycles)

            cycle_df = pd.DataFrame({
                "marker": m,
                "x": x,
                "y": y,
                "z": z,
                "cycle": cycle_count,
                "percent_gait_cycle": cycle_percent,
                "system": tracker
            })

            dfs.append(cycle_df)

    return pd.concat(dfs, ignore_index=True)

def create_trajectory_cycles(freemocap_dict:dict[str, np.ndarray],
                             qualisys_dict:dict[str, np.ndarray],
                             marker_list: list[str],
                             gait_events: dict[str, list[slice]],
                             freemocap_tracker_name:str,
                             n_points:int = 100) -> pd.DataFrame:
    
    freemocap_cycles = get_cycles_for_tracker(
        trajectory_dict=freemocap_dict,
        marker_list=marker_list,
        gait_events=gait_events,
        tracker=freemocap_tracker_name,
        n_points=n_points
    )

    qualisys_cycles = get_cycles_for_tracker(
        trajectory_dict=qualisys_dict,
        marker_list=marker_list,
        gait_events=gait_events,
        tracker="qualisys",
        n_points=n_points
    )

    return pd.concat([freemocap_cycles, qualisys_cycles], ignore_index=True)