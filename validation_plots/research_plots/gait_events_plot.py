import sqlite3
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EventMatchResult:
    differences: int
    false_positives: int
    false_negatives: int

conn = sqlite3.connect("validation.db")

query = """
SELECT t.participant_code,
        t.trial_name,
        a.path,
        a.component_name,
        a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "treadmill"
    AND a.category = "gait_events"
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE "%gait_events"
ORDER BY t.trial_name, a.path
"""

reference_system = "qualisys"
TRACKERS = ["mediapipe"]


def find_closest_pair(reference_frame:int, tracker_frames:list, tolerance):
    tracker_frames = np.array(tracker_frames, dtype = int)
    closest_points = tracker_frames[(tracker_frames >= (reference_frame-tolerance)) & (tracker_frames <= (reference_frame + tolerance))]
    
    if len(closest_points) == 1:
        return closest_points[0]
    elif len(closest_points) == 0:
        return None
    elif len(closest_points) > 1:
        differences = closest_points - reference_frame
        return closest_points[np.argmin(np.abs(differences))]  #find the first closest point

   

def match_events(reference_frames:list, tracker_frames:list, tolerance: int = 2):
    reference_frames = sorted(reference_frames)
    tracker_frames = sorted(tracker_frames)
    
    differences = []
    num_ref_frames = len(reference_frames)
    num_tracker_frames = len(tracker_frames)

    if num_ref_frames > num_tracker_frames:
        print("More reference frames than tracker frames")
    elif num_ref_frames < num_tracker_frames:
        print("More tracker frames than reference frames")

    remaining_rframes = reference_frames.copy()
    remaining_tframes = tracker_frames.copy()
    for rframe in reference_frames:
        closest_frame = find_closest_pair(reference_frame=rframe, tracker_frames=remaining_tframes, tolerance=tolerance)
        
        if closest_frame is not None:
            remaining_rframes.pop(remaining_rframes.index(rframe))
            remaining_tframes.pop(remaining_tframes.index(closest_frame))
            
            differences.append(closest_frame - rframe)
        
    false_positives = len(remaining_tframes)
    false_negatives = len(remaining_rframes)

    return EventMatchResult(
        differences=differences,
        false_positives=false_positives,
        false_negatives=false_negatives
    )
            
    f= 2


path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"].lower()
    sub['tracker'] = row['tracker']
    dfs.append(sub)

df:pd.DataFrame = pd.concat(dfs, ignore_index=True)

differences_per_tracker = defaultdict(list)
fp_per_tracker = defaultdict(int)
fn_per_tracker = defaultdict(int)
for trial in df['trial_name'].unique():
    df_trial = df[df["trial_name"] == trial
                  ]
    for foot in df_trial['foot'].unique():
        df_foot = df_trial[df_trial['foot'] == foot]
        tracker_frames = {}
        for event in df_foot['event'].unique():

            sub_df = df_foot[df_foot['event'] == event]

            reference_frames = list(sub_df.groupby('tracker').get_group(reference_system)['frame'])

            for tracker in TRACKERS:
                tracker_frames = list(sub_df.groupby('tracker').get_group(tracker)['frame'])
                res:EventMatchResult = match_events(reference_frames, tracker_frames, tolerance=2)
                differences_per_tracker[(tracker, event)].extend(res.differences)
                fp_per_tracker[(tracker, event)] += res.false_positives
                fn_per_tracker[(tracker, event)] += res.false_negatives

            

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def add_frame_histogram_panel(
    fig: go.Figure,
    diffs_frames,
    *,
    row: int,
    col: int,
    ncols: int,
    title: str,
    max_frames: int = 3,
    show_ylabel: bool = True,
    fill_color: str = "#1f77b4",
    edge_color: str = "#0a3f64",
    fill_alpha: float = 0.45,
) -> None:
    x = np.asarray(diffs_frames, dtype=float)
    x = x[np.isfinite(x)]

    edges = np.arange(-max_frames - 0.5, max_frames + 0.5 + 1e-12, 1.0)
    counts, bin_edges = np.histogram(x, bins=edges)

    ymax = int(counts.max()) if counts.size else 1
    fig.update_yaxes(range=[0, ymax * 1.08], row=row, col=col)

    lefts = bin_edges[:-1]
    rights = bin_edges[1:]

    xs = np.column_stack([lefts, rights]).ravel()
    ys = np.column_stack([counts, counts]).ravel()

    xs_poly = np.concatenate(([xs[0]], xs, [xs[-1]]))
    ys_poly = np.concatenate(([0], ys, [0]))

    fig.add_trace(
        go.Scatter(
            x=xs_poly, y=ys_poly,
            mode="lines",
            fill="tozeroy",
            line=dict(width=0),
            fillcolor=hex_to_rgba(fill_color, fill_alpha),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=xs_poly, y=ys_poly,
            mode="lines",
            line=dict(color=edge_color, width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row, col=col
    )

    fig.add_vline(
        x=0.0,
        line=dict(color=edge_color, width=1.2, dash="dash"),
        opacity=0.4,
        row=row, col=col
    )

    # Bold subplot title
    annotation_idx = (row - 1) * ncols + (col - 1)
    fig.layout.annotations[annotation_idx].update(text=f"<b>{title}</b>")

    tickvals = list(range(-max_frames, max_frames + 1))
    fig.update_xaxes(
        range=[-max_frames - 0.5, max_frames + 0.5],
        tickmode="array",
        tickvals=tickvals,
        ticktext=[("0" if v == 0 else f"{v:+d}") for v in tickvals],
        row=row, col=col,
    )

    fig.update_yaxes(title_text="<b>Count</b>" if show_ylabel else "", row=row, col=col)

    mu = float(np.mean(x)) if x.size else float("nan")
    sd = float(np.std(x, ddof=1)) if x.size >= 2 else float("nan")
    label = f"μ = {mu:+.2f} ± {sd:.2f} frames" if np.isfinite(mu) and np.isfinite(sd) else "μ = n/a"

    axis_suffix = "" if (row == 1 and col == 1) else str((row - 1) * ncols + col)
    fig.add_annotation(
        x=0.03, y=0.93,
        xref=f"x{axis_suffix} domain",
        yref=f"y{axis_suffix} domain",
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=11),
    )

    fig.update_yaxes(showgrid=False, row=row, col=col)
    fig.update_xaxes(showgrid=False, row=row, col=col)

# --------------------------
# Build HS vs TO histograms
# --------------------------

tracker = "mediapipe"   # or loop over TRACKERS if you want multiple figures


# Collect diffs for HS and TO across whatever event labels you have
hs_diffs = []
to_diffs = []

for (trk, evt), diffs in differences_per_tracker.items():
    if trk != tracker:
        continue

    if evt == "heel_strike":
        hs_diffs.extend(diffs)
    elif evt == "toe_off":
        to_diffs.extend(diffs)

# print("HS event labels found:", sorted({evt for (trk, evt) in differences_per_tracker if trk == tracker and is_hs(evt)}))
# print("TO event labels found:", sorted({evt for (trk, evt) in differences_per_tracker if trk == tracker and is_to(evt)}))

# Make the figure
ncols = 2
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Heel strike", "Toe off"),
    horizontal_spacing=0.06,
)

add_frame_histogram_panel(
    fig, hs_diffs,
    row=1, col=1, ncols=ncols,
    title="Heel strike",
    max_frames=3,
    show_ylabel=True,
)

add_frame_histogram_panel(
    fig, to_diffs,
    row=1, col=2, ncols=ncols,
    title="Toe off",
    max_frames=3,
    show_ylabel=False,
)

fig.update_layout(
    width=500,
    height=300,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=70),
)

fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, title_text="<b>Error (frames)</b>")
fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

fig.show()