"""
Paper figure: per-segment length error vs Qualisys.

Pared from the full exploratory module to the single view that supports the
scaling-factor figure in the main text, and laid out as a two-panel figure.

What this figure is for
-----------------------
The main-text scaling-factor box shows ViTPose ~2.5% below 1.0 while MediaPipe
and RTMPose sit near 1.0. Because the isotropic scale is fit to *positions*, it
is effectively a longitudinal/stature fit dominated by the markers farthest
from the body centroid. This figure shows where the per-segment biases live and
makes two points read at a glance:

  A. Upper body (spine, shoulder, upper arm, forearm), proximal->distal down
     the arm. The large shoulder bias here is SHARED across all three trackers
     (2D keypoint conventions / topology), not a ViTPose effect.
  B. Lower body (pelvis, thigh, shank, foot), proximal->distal down the leg.
     The shank is the ONE segment where ViTPose departs from the other two
     (~+6% vs ~0); via the kinematic chain an over-long shank displaces the
     distal foot complex outward, which is what pulls the positional scale
     down. The large pelvis bias is, again, shared across trackers.

head and heel are dropped: their keypoints are the least reliable in this rig
and add a distracting failure mode unrelated to the scaling story.

Style note
----------
Print-style tokens (font, sizes, colours, panel dimensions) are constants at
the top so the figure can be snapped to the house style of the other figures in
one place. `plot_segment_error_figure` is the paper output; `plot_segment_error`
is kept as a quick per-side QA view (set collapse_sides=False to confirm
left/right symmetry rather than assume it).
"""

import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skellymodels.managers.human import Human

DB_PATH = "validation.db"

TRACKER_ORDER = ["rtmpose", "mediapipe", "vitpose"]
TRACKER_COLORS = {
    "mediapipe": "#054f85",
    "rtmpose": "#d36600",
    "vitpose": "#029131",
}
TRACKER_DISPLAY = {
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}
TRACKER_SYMBOLS = {
    "mediapipe": "square",
    "rtmpose": "circle",
    "vitpose": "diamond",
}

# Proximal -> distal so plots read anatomically.
SEGMENT_TYPE_ORDER = [
    "pelvis", "spine",
    "clavicle", "shoulder", "upper_arm", "forearm", "hand",
    "thigh", "shank", "foot",
]
SEGMENT_DISPLAY = {
    "pelvis": "Pelvis", "spine": "Spine", "shoulder": "Shoulder",
    "upper_arm": "Upper arm", "forearm": "Forearm", "thigh": "Thigh",
    "shank": "Shank", "foot": "Foot",
}

# Segments dropped from the figure (least reliable keypoints).
EXCLUDE_SEGMENTS = ("head", "heel")

# Two-panel layout: (panel label, group name, segments proximal->distal).
# ROW_GROUPS = [
#     ("A", "Upper body", ["spine", "shoulder", "upper_arm", "forearm"]),
#     ("B", "Lower body", ["pelvis", "thigh", "shank", "foot"]),
# ]
# Scale-preserving alternative: keep both big shared biases (pelvis & shoulder)
# on the top row so the lower row stays zoomed and the shank's +6% pops. Also
# keeps forearm with upper_arm. Swap in if the shank reads too small in row B.
ROW_GROUPS = [
    ("A", "Trunk and arms", ["pelvis", "spine", "shoulder", "upper_arm", "forearm"]),
    ("B", "Lower limb",     ["thigh", "shank", "foot"]),
]

# ---- print-style tokens (matched to scaling_factor_boxplot.py) ------------
FONT_FAMILY = None        # inherit template default, as the scaling figure does
FONT_SIZE = 12
AXIS_TITLE_SIZE = 12
TICK_SIZE = 11
PANEL_LABEL_SIZE = 14
FIG_WIDTH = 1000
FIG_HEIGHT = 640
GRID_COLOR = "#E6E6E6"
AXIS_COLOR = "#9AA0A6"
ZERO_LINE_COLOR = "black"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def get_segment_length_dataframe(database_path: str) -> pd.DataFrame:
    query = """
    SELECT
        t.participant_code,
        t.trial_name,
        a.path,
        a.component_name,
        a.condition,
        a.tracker
    FROM artifacts AS a
    JOIN trials AS t
        ON a.trial_id = t.id
    WHERE a.category = 'synced_data'
      AND a.tracker IN ('mediapipe', 'rtmpose', 'vitpose', 'qualisys')
      AND a.file_exists = 1
    ORDER BY t.trial_name, a.path
    """

    with sqlite3.connect(database_path) as conn:
        path_df = pd.read_sql_query(query, conn)

    records = []

    for row in path_df.itertuples(index=False):
        human: Human = Human.from_parquet(row.path)

        segment_connections = (
            human.body.anatomical_structure.segment_connections
        )

        joint_names = human.body.xyz.as_dict.keys()
        xyz = human.body.xyz.as_array

        joint_index = {
            joint_name: index
            for index, joint_name in enumerate(joint_names)
        }

        valid_segments = [
            (segment_name, connection["proximal"], connection["distal"])
            for segment_name, connection in segment_connections.items()
            if (
                connection["proximal"] in joint_index
                and connection["distal"] in joint_index
            )
        ]

        segment_names = [segment[0] for segment in valid_segments]
        proximal_names = [segment[1] for segment in valid_segments]
        distal_names = [segment[2] for segment in valid_segments]

        proximal_indices = np.fromiter(
            (joint_index[name] for name in proximal_names), dtype=int
        )
        distal_indices = np.fromiter(
            (joint_index[name] for name in distal_names), dtype=int
        )

        proximal_xyz = xyz[:, proximal_indices, :]
        distal_xyz = xyz[:, distal_indices, :]

        lengths = np.linalg.norm(distal_xyz - proximal_xyz, axis=-1)

        medians = np.nanmedian(lengths, axis=0)
        stdevs = np.nanstd(lengths, axis=0)

        records.extend(
            {
                "participant_code": row.participant_code,
                "trial_name": row.trial_name,
                "condition": row.condition,
                "tracker": row.tracker,
                "component_name": row.component_name,
                "path": row.path,
                "segment": segment_name,
                "proximal": proximal,
                "distal": distal,
                "median": median,
                "stdev": stdev,
            }
            for (segment_name, proximal, distal, median, stdev) in zip(
                segment_names, proximal_names, distal_names, medians, stdevs
            )
        )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_segment_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Split `segment` into a side and a side-agnostic segment_type."""
    df = df.copy()
    df["side"] = np.select(
        [df["segment"].str.startswith("left_"),
         df["segment"].str.startswith("right_")],
        ["left", "right"],
        default="midline",
    )
    df["segment_type"] = df["segment"].str.replace(
        r"^(left|right)_", "", regex=True
    )
    return df


def ordered_segments(present) -> list[str]:
    """Order raw `segment` names proximal->distal, left then right."""
    present = set(present)
    order = []
    for seg in SEGMENT_TYPE_ORDER:
        for name in (f"left_{seg}", f"right_{seg}", seg):
            if name in present:
                order.append(name)
    order += [s for s in present if s not in order]
    return order


def compute_paired_differences(
    df: pd.DataFrame,
    reference: str = "qualisys",
    value: str = "median",
) -> pd.DataFrame:
    """
    One row per matched (recording, segment, tracker) with the tracker length,
    the Qualisys length, and the signed difference in mm and percent. Percent
    removes body size and segment size, so a per-segment scaling bias shows up
    as a roughly constant offset regardless of subject or limb size.
    """
    df = add_segment_parts(df)

    index_columns = [
        "participant_code", "trial_name", "condition",
        "segment", "segment_type", "side",
    ]

    wide = (
        df.pivot_table(
            index=index_columns,
            columns="tracker",
            values=value,
            aggfunc="first",
        )
        .reset_index()
    )

    if reference not in wide.columns:
        raise ValueError(f"Reference tracker {reference!r} not found.")

    frames = []
    for tracker in (t for t in TRACKER_ORDER if t in wide.columns):
        valid = wide[[tracker, reference]].notna().all(axis=1)
        sub = wide.loc[valid, index_columns].copy()
        sub["tracker"] = tracker
        sub["tracker_length_mm"] = wide.loc[valid, tracker].to_numpy()
        sub["reference_length_mm"] = wide.loc[valid, reference].to_numpy()
        sub["difference_mm"] = (
            sub["tracker_length_mm"] - sub["reference_length_mm"]
        )
        sub["percent_difference"] = (
            100 * sub["difference_mm"] / sub["reference_length_mm"]
        )
        frames.append(sub)

    if not frames:
        raise ValueError("No comparison trackers matched the reference.")

    return pd.concat(frames, ignore_index=True)


def _round_range(values, pad=3.0, step=5.0):
    """Round a data range out to clean tick-friendly bounds."""
    lo = np.floor((np.nanmin(values) - pad) / step) * step
    hi = np.ceil((np.nanmax(values) + pad) / step) * step
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Paper figure: two panels, proximal (large/shared) over distal (zoomed)
# ---------------------------------------------------------------------------
def plot_segment_error_figure(
    paired: pd.DataFrame,
    exclude=EXCLUDE_SEGMENTS,
):
    data = paired[~paired["segment_type"].isin(exclude)].copy()
    trackers = [t for t in TRACKER_ORDER if t in data["tracker"].unique()]

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16)

    for row_idx, (panel, group_name, segs) in enumerate(ROW_GROUPS, start=1):
        segs_present = [s for s in segs if s in set(data["segment_type"])]
        cats = [SEGMENT_DISPLAY.get(s, s) for s in segs_present]
        row_data = data[data["segment_type"].isin(segs_present)]

        for tracker in trackers:
            sub = row_data[row_data["tracker"] == tracker]
            x = sub["segment_type"].map(SEGMENT_DISPLAY).fillna(
                sub["segment_type"]
            )
            fig.add_trace(
                go.Box(
                    x=x,
                    y=sub["percent_difference"],
                    name=TRACKER_DISPLAY.get(tracker, tracker),
                    legendgroup=tracker,
                    showlegend=(row_idx == 1),
                    offsetgroup=tracker,
                    alignmentgroup="trackers",
                    marker={
                        "size": 4,
                        "opacity": 0.6,
                        "color": TRACKER_COLORS[tracker],
                    },
                    marker_color=TRACKER_COLORS[tracker],
                    line={"color": TRACKER_COLORS[tracker], "width": 1},
                    opacity=0.75,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=0,
                ),
                row=row_idx, col=1,
            )

        lo, hi = _round_range(row_data["percent_difference"])
        fig.add_hline(
            y=0, line_dash="dash", line_width=1,
            line_color=ZERO_LINE_COLOR, row=row_idx, col=1,
        )
        fig.update_xaxes(
            categoryorder="array", categoryarray=cats,
            tickfont={"size": TICK_SIZE},
            showgrid=False, showline=True, linecolor=AXIS_COLOR,
            row=row_idx, col=1,
        )
        fig.update_yaxes(
            range=[lo, hi],
            dtick=10 if (hi - lo) > 30 else 5,
            title_text="Difference from Qualisys (%)",
            title_font={"size": AXIS_TITLE_SIZE},
            tickfont={"size": TICK_SIZE},
            gridcolor=GRID_COLOR, zeroline=False,
            showline=True, linecolor=AXIS_COLOR,
            row=row_idx, col=1,
        )
        # Panel label + group name, top-left of each subplot.
        fig.add_annotation(
            text=f"<b>{panel}</b>  {group_name}",
            xref="x domain", yref="y domain",
            x=0.0, y=1.10, xanchor="left", yanchor="bottom",
            showarrow=False,
            font={"size": PANEL_LABEL_SIZE, "family": FONT_FAMILY},
            row=row_idx, col=1,
        )

    fig.update_layout(
        template="plotly_white",
        boxmode="group",
        boxgap=0.35,
        boxgroupgap=0.2,
        font={"family": FONT_FAMILY, "size": FONT_SIZE},
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        margin={"l": 70, "r": 150, "t": 50, "b": 40},
        legend={
            "title_text": "Tracker",
            "orientation": "v",
            "yanchor": "top", "y": 1.0,
            "xanchor": "left", "x": 1.02,
        },
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# QA view: per-side box plot (set collapse_sides=False to inspect symmetry)
# ---------------------------------------------------------------------------
def plot_segment_error(paired: pd.DataFrame, collapse_sides: bool = True):
    trackers = [t for t in TRACKER_ORDER if t in paired["tracker"].unique()]

    if collapse_sides:
        x_col = "segment_type"
        present = paired["segment_type"].unique()
        x_order = [s for s in SEGMENT_TYPE_ORDER if s in present]
        x_order += [s for s in present if s not in x_order]
        x_label = "Segment"
    else:
        x_col = "segment"
        x_order = ordered_segments(paired["segment"].unique())
        x_label = "Side and segment"

    fig = px.box(
        paired,
        x=x_col,
        y="percent_difference",
        color="tracker",
        points="all",
        category_orders={x_col: x_order, "tracker": trackers},
        color_discrete_map=TRACKER_COLORS,
        labels={
            x_col: x_label,
            "percent_difference": "Difference from Qualisys (%)",
            "tracker": "Tracker",
        },
        title="Segment-length error vs Qualisys (%)",
    )
    fig.update_traces(
        boxmean=True, marker={"size": 4, "opacity": 0.35},
        jitter=0.3, pointpos=0,
    )
    fig.add_hline(y=0, line_dash="dash", line_width=1)
    fig.update_layout(
        template="plotly_white", boxmode="group", height=600,
        hovermode="closest", legend_title_text="Tracker",
    )
    return fig


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    segment_length_df = get_segment_length_dataframe(DB_PATH)
    paired_df = compute_paired_differences(segment_length_df)

    fig = plot_segment_error_figure(paired_df)
    fig.show()
    # Export for the manuscript (needs kaleido):
    # fig.write_image("segment_error_supplement.svg",
    #                 width=FIG_WIDTH, height=FIG_HEIGHT, scale=2)