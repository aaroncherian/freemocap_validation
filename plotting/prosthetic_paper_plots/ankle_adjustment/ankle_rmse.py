from pathlib import Path
import pandas as pd


conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }

tracker = "rtmpose_dlc"
joint_to_extract = {"ankle":"dorsi_plantar", "knee":"flex_ext"}

all_dfs = []

for condition, rec_path in conditions.items():
    if not Path(rec_path).is_dir():
        raise FileNotFoundError(f"Condition path does not exist or is not a directory: {rec_path}")
    
    path_to_rmses = Path(rec_path)/'validation'/tracker/'joint_angles'/'joint_angles_per_stride_rmse_stats.csv'

    df_rmse = pd.read_csv(path_to_rmses)

    for joint, component in joint_to_extract.items():
        df_rmse_joint = df_rmse.query(f"joint == '{joint}' & side == 'right' and component == '{component}'")
        df_mean = pd.DataFrame({
            "mean": [df_rmse_joint["rmse_deg"].mean()],
            "std":  [df_rmse_joint["rmse_deg"].std()],
            "condition": [condition],
            "joint" : [joint]
        })
        all_dfs.append(df_mean)
    

df_mean_rmse = pd.concat(all_dfs, ignore_index=True)

values_df = df_mean_rmse.groupby("joint").agg(mean = ("mean", "mean"), std = ("mean", "std"))

for joint in values_df.index:
    print(f"Mean RMSE across conditions for {joint}: {values_df.loc[joint]['mean']:.2f} degrees (std: {values_df.loc[joint]['std']:.2f} degrees)")

f = 2