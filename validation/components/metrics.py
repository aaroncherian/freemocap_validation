from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import load_csv, load_numpy, save_numpy, save_csv

POSITIONRMSE = DataComponent(
    name = "position_rmse",
    filename= "position_rmse.csv",
    relative_path= "validation/mediapipe/metrics",
    saver = save_csv,
    loader= load_csv
)

POSITIONABSOLUTEERROR = DataComponent(
    name = "position_absolute_error",
    filename = "position_absolute_error.csv",
    relative_path= "validation/mediapipe/metrics",
    saver=save_csv,
    loader=load_csv
)

VELOCITYRMSE = DataComponent(
    name = "velocity_rmse",
    filename= "velocity_rmse.csv",
    relative_path= "validation/mediapipe/metrics",
    saver = save_csv,
    loader= load_csv
)

VELOCITYABSOLUTEERROR = DataComponent(
    name = "velocity_absolute_error",
    filename= "velocity_absolute_error.csv",
    relative_path= "validation/mediapipe/metrics",
    saver=save_csv,
    loader= load_csv
)