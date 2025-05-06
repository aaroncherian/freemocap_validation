from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import load_csv, load_numpy, save_numpy, save_csv

PositionRMSE = DataComponent(
    name = "position_rmse",
    filename= "position_rmse.csv",
    saver = save_csv,
    loader= load_csv
)

PositionAbsoluteError = DataComponent(
    name = "position_absolute_error",
    file_name = "position_absolute_error.csv",
    saver=save_csv,
    loader=load_csv
)