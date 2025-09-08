from database.database_helpers import DatabaseHelper
import pandas as pd
from pathlib import Path
import json 
import plotly.express as px

db:DatabaseHelper = DatabaseHelper(r"mydb3.sqlite")

EYES_OPEN_SOLID_GROUND = "Eyes Open/Solid Ground"
EYES_CLOSED_SOLID_GROUND = "Eyes Closed/Solid Ground"
EYES_OPEN_FOAM = "Eyes Open/Foam"
EYES_CLOSED_FOAM = "Eyes Closed/Foam"


condition_order = [
    EYES_OPEN_SOLID_GROUND,
    EYES_CLOSED_SOLID_GROUND,
    EYES_OPEN_FOAM,
    EYES_CLOSED_FOAM
]

def get_nih_data(database:DatabaseHelper, tracker:str):
    df = pd.read_sql_query(db.sql_view, db.con,
                       params=["balance", tracker, "path_length"])

    df['file_path'] = df.apply(
        lambda row: Path(row['data_root'])/row['trial_name']/'validation'/row['tracker']/'path_length_analysis'/row['folder_name'], axis = 1
    )

    return df


def read_condition_values(file_path:Path):
    json_file_path = file_path/'condition_data.json'
    try:
        with open (json_file_path, 'r') as f:
            data = json.load(f)
        return data.get("Path Lengths:", None)
    except FileNotFoundError:
        return None

df_qual = get_nih_data(db, "qualisys")
df_mediapipe = get_nih_data(db, "mediapipe")

df = pd.concat([df_qual, df_mediapipe], ignore_index=True)

df["conditions_dict"] = df["file_path"].apply(read_condition_values)

# Expand dict to columns, then melt to long
conditions_wide = df["conditions_dict"].apply(pd.Series)
id_cols = [c for c in df.columns if c != "conditions_dict"]  # keep your metadata columns

df_long = pd.concat([df[id_cols], conditions_wide], axis=1) \
            .melt(
                id_vars=id_cols,
                var_name="condition",
                value_name="path_length"
            ) \
            .dropna(subset=["path_length"])  # drop rows for missing values


df_long[(df_long['condition'] == EYES_OPEN_SOLID_GROUND) & (df_long['tracker'] == "mediapipe")]

mean_std_df = (
    df_long
    .groupby(["tracker", "condition"], as_index=False)["path_length"]
    .agg(["mean", "std"])
    .reset_index()
)

mean_std_df['condition'] = pd.Categorical(mean_std_df['condition'], categories=condition_order, ordered=True)
mean_std_df = mean_std_df.sort_values(
    ["tracker", "condition"], 
    key=lambda col: col.cat.codes if col.name == "condition" else col
)


fig = px.line(
    mean_std_df,
    x="condition",
    y="mean",
    color="tracker",
    error_y="std",
    markers=True,
    category_orders={"condition": condition_order},
    line_group="tracker"   # explicit grouping, prevents stray traces
)
fig.update_layout(
    xaxis_title="Condition",
    yaxis_title="Path Length",
    title="Mean ± SD Path Length per Condition (by Tracker)"
)
fig.show()
f = 2
