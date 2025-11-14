import pandas as pd
import numpy as np

def rmse(err):
    return np.sqrt(np.mean(err**2))

def calculate_rmse(freemocap_df: pd.DataFrame, qualisys_df: pd.DataFrame):

    rmse_rows = []

    # Unique combinations we want RMSE for
    combos = (
        freemocap_df[["joint", "side", "component"]]
        .drop_duplicates()
    )

    for cycle in sorted(freemocap_df["cycle"].unique()):
        
        fmc_cycle = freemocap_df.query("cycle == @cycle")
        qtm_cycle = qualisys_df.query("cycle == @cycle")

        for _, row in combos.iterrows():
            joint = row["joint"]
            side = row["side"]
            comp = row["component"]

            # Extract the angle series for this DOF
            fmc_sub = (
                fmc_cycle
                .query("joint == @joint and side == @side and component == @comp")
                .sort_values("percent_gait_cycle")
            )
            qtm_sub = (
                qtm_cycle
                .query("joint == @joint and side == @side and component == @comp")
                .sort_values("percent_gait_cycle")
            )

            # Skip if missing in this stride
            if fmc_sub.empty or qtm_sub.empty:
                # print(f"Skip cycle {cycle}, {side} {joint} {comp}")
                continue

            err = fmc_sub["angle"].values - qtm_sub["angle"]
            rmse_angle = rmse(err)

            rmse_rows.append({
                "cycle": cycle,
                "joint": joint,
                "side": side,
                "component": comp,     # angle DOF
                "rmse_deg": rmse_angle
            })

    return pd.DataFrame(rmse_rows)
