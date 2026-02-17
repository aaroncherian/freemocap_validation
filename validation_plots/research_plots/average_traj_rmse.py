from pathlib import Path
import pandas as pd
import re

def load_rmse_speed_table(csv_path: Path) -> pd.DataFrame:
    """
    Reads a CSV shaped like:
      Speed (m/s), 0.5, 1, 1.5, 2, 2.5
      Mediapipe,   5.4, 4.6, ...
      Rtmpose,     4.6, 4.0, ...
    Returns long-form:
      tracker, speed, rmse
    """
    df = pd.read_csv(csv_path)

    # First column contains tracker names (despite being labeled "Speed (m/s)")
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "tracker"})

    # Melt speeds into rows
    long_df = df.melt(
        id_vars=["tracker"],
        var_name="speed",
        value_name="rmse",
    )

    # Clean types
    long_df["tracker"] = long_df["tracker"].astype(str).str.strip()
    long_df["speed"] = pd.to_numeric(long_df["speed"], errors="coerce")
    long_df["rmse"] = pd.to_numeric(long_df["rmse"], errors="coerce")

    # Drop any junk rows/cols that didn't parse
    long_df = long_df.dropna(subset=["tracker", "speed", "rmse"])

    return long_df


def summarize_folder(folder: Path) -> pd.DataFrame:
    """
    Loops files like: ankle_x_trajectory_rmse_table.csv
    Produces mean RMSE per tracker per joint per dimension (averaged across speeds).
    """
    files = sorted(folder.glob("*_trajectory_rmse_table*.csv"))
    out = []

    for f in files:
        m = re.match(r"(.*?)_([xyz])_trajectory_rmse_table", f.stem)
        if not m:
            # skip anything that doesn't match joint_dim pattern
            continue

        joint, dim = m.groups()

        long_df = load_rmse_speed_table(f)

        # Average across speeds within this file
        avg = (
            long_df.groupby("tracker", as_index=False)["rmse"]
            .mean()
            .rename(columns={"rmse": "mean_rmse_over_speeds"})
        )
        avg["joint"] = joint
        avg["dimension"] = dim
        avg["source_file"] = f.name  # handy for debugging

        out.append(avg)

    if not out:
        return pd.DataFrame(columns=["tracker", "joint", "dimension", "mean_rmse_over_speeds", "source_file"])

    final = pd.concat(out, ignore_index=True)

    # Nice ordering
    final = final[["tracker", "joint", "dimension", "mean_rmse_over_speeds", "source_file"]]
    final = final.sort_values(["joint", "dimension", "tracker"]).reset_index(drop=True)

    return final


# -----------------------
# Use it
# -----------------------
folder = Path(r"D:\validation")  # <- change this
final_df = summarize_folder(folder)

print(final_df)

# Save
final_df.to_csv(folder / "avg_rmse_per_tracker_joint_dimension.csv", index=False)
