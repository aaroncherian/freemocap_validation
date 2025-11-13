import pandas as pd
import numpy as np

def rmse(error: np.ndarray):
    return np.sqrt(np.mean(error**2))

def calculate_rmse(freemocap_df: pd.DataFrame, qualisys_df: pd.DataFrame):

    rmse_rows = []

    markers = freemocap_df["marker"].unique()  # your exact list

    for cycle in sorted(freemocap_df["cycle"].unique()):
        fmc_cycle = freemocap_df.query("cycle == @cycle")
        qual_cycle = qualisys_df.query("cycle == @cycle")

        for marker in markers:
            fmc_marker = (
                fmc_cycle.query("marker == @marker")
                .sort_values("percent_gait_cycle")
            )
            qual_marker = (
                qual_cycle.query("marker == @marker")
                .sort_values("percent_gait_cycle")
            )

            # --- safety: skip if marker missing in this stride for one system ---
            if fmc_marker.empty or qual_marker.empty:
                # Optional: uncomment to see what's missing
                print(f"Skipping cycle {cycle}, marker {marker} (empty in one system)")
                continue

            diff = fmc_marker[["x", "y", "z"]].values - qual_marker[["x", "y", "z"]].values

            rmse_x = rmse(diff[:, 0])
            rmse_y = rmse(diff[:, 1])
            rmse_z = rmse(diff[:, 2])

            err_3d = np.linalg.norm(diff, axis=1)
            rmse_3d = rmse(err_3d)

            rmse_rows.append({
                "cycle": cycle,
                "marker": marker,
                "rmse_x": rmse_x,
                "rmse_y": rmse_y,
                "rmse_z": rmse_z,
                "rmse_3d": rmse_3d,
            })

    freemocap_rmse = pd.DataFrame(rmse_rows)
    return freemocap_rmse
