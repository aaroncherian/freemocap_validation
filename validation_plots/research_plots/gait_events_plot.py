import sqlite3
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EventMatchResult:
    differences: list[int]
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
    AND a.tracker IN ("mediapipe", "qualisys", "rtmpose", "vitpose")
    AND a.file_exists = 1
    AND a.component_name LIKE "%gait_events"
ORDER BY t.trial_name, a.path
"""

reference_system = "qualisys"
TRACKERS = ["mediapipe", "rtmpose", "vitpose"]

def find_closest_pair(reference_frame: int, tracker_frames: list, tolerance: int):
    tracker_frames = np.array(tracker_frames, dtype=int)
    closest_points = tracker_frames[
        (tracker_frames >= (reference_frame - tolerance)) &
        (tracker_frames <= (reference_frame + tolerance))
    ]

    if len(closest_points) == 1:
        return closest_points[0]
    elif len(closest_points) == 0:
        return None
    else:
        differences = closest_points - reference_frame
        return closest_points[np.argmin(np.abs(differences))]

def match_events(reference_frames: list, tracker_frames: list, tolerance: int = 2) -> EventMatchResult:
    reference_frames = sorted(reference_frames)
    tracker_frames = sorted(tracker_frames)

    differences = []
    remaining_rframes = reference_frames.copy()
    remaining_tframes = tracker_frames.copy()

    for rframe in reference_frames:
        closest_frame = find_closest_pair(
            reference_frame=rframe,
            tracker_frames=remaining_tframes,
            tolerance=tolerance
        )
        if closest_frame is not None:
            remaining_rframes.pop(remaining_rframes.index(rframe))
            remaining_tframes.pop(remaining_tframes.index(closest_frame))
            differences.append(int(closest_frame - rframe))

    false_positives = len(remaining_tframes)
    false_negatives = len(remaining_rframes)

    return EventMatchResult(
        differences=differences,
        false_positives=false_positives,
        false_negatives=false_negatives
    )

# ------------------------
# Load gait event CSVs
# ------------------------
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"].lower()
    sub["tracker"] = row["tracker"]
    # normalize event strings so comparisons work reliably
    sub["event"] = sub["event"].astype(str).str.lower()
    sub["foot"] = sub["foot"].astype(str).str.lower()
    dfs.append(sub)

df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

# ------------------------
# Compute differences per (tracker,event)
# ------------------------
differences_per_tracker = defaultdict(list)
fp_per_tracker = defaultdict(int)
fn_per_tracker = defaultdict(int)

for trial in df["trial_name"].unique():
    df_trial = df[df["trial_name"] == trial]

    for foot in df_trial["foot"].unique():
        df_foot = df_trial[df_trial["foot"] == foot]

        for event in df_foot["event"].unique():
            sub_df = df_foot[df_foot["event"] == event]

            # skip if reference system missing in this slice
            if reference_system not in sub_df["tracker"].unique():
                continue

            reference_frames = list(sub_df[sub_df["tracker"] == reference_system]["frame"])

            for tracker in TRACKERS:
                if tracker not in sub_df["tracker"].unique():
                    continue

                tracker_frames = list(sub_df[sub_df["tracker"] == tracker]["frame"])
                res = match_events(reference_frames, tracker_frames, tolerance=2)
                differences_per_tracker[(tracker, event)].extend(res.differences)
                fp_per_tracker[(tracker, event)] += res.false_positives
                fn_per_tracker[(tracker, event)] += res.false_negatives


# ------------------------
# Plot helpers
# ------------------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _add_hist_polygon(
    fig: go.Figure,
    diffs_frames,
    *,
    row: int,
    col: int,
    max_frames: int,
    fill_color: str,
    edge_color: str,
    fill_alpha: float,
    name: str | None,
    showlegend: bool,
    fps: int = 30,
):
    x = np.asarray(diffs_frames, dtype=float)
    x_ms = x * (1000.0 / fps)
    x_ms = x_ms[np.isfinite(x_ms)]

    frame_ms = 1000.0 / fps
    edges = np.arange(-max_frames - 0.5, max_frames + 0.5 + 1e-12, 1.0) * frame_ms
    counts, bin_edges = np.histogram(x_ms, bins=edges)

    lefts = bin_edges[:-1]
    rights = bin_edges[1:]
    xs = np.column_stack([lefts, rights]).ravel()
    ys = np.column_stack([counts, counts]).ravel()
    xs_poly = np.concatenate(([xs[0]], xs, [xs[-1]]))
    ys_poly = np.concatenate(([0], ys, [0]))

    # fill (legend entry comes from THIS trace)
    fig.add_trace(
        go.Scatter(
            x=xs_poly, y=ys_poly,
            mode="lines",
            fill="tozeroy",
            line=dict(width=0),
            fillcolor=hex_to_rgba(fill_color, fill_alpha),
            name=name if name else None,
            showlegend=showlegend,
            hoverinfo="skip",
        ),
        row=row, col=col
    )

    # outline (no legend)
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

    return x, counts

def add_frame_histogram_panel_overlay(
    fig: go.Figure,
    diffs_by_tracker: dict[str, list[int]],
    *,
    row: int,
    col: int,
    ncols: int,
    title: str,
    max_frames: int = 3,
    show_ylabel: bool = True,
    style_by_tracker: dict[str, dict] | None = None,
    showlegend: bool = True,
    fps: int = 30,
) -> None:
    if style_by_tracker is None:
        style_by_tracker = {}

    # add vline at 0 once
    fig.add_vline(
        x=0.0,
        line=dict(color="black", width=1.2, dash="dash"),
        opacity=0.35,
        row=row, col=col
    )

    # overlay polygons
    all_counts_max = 1
    stats_lines = []

    for i, (trk, diffs) in enumerate(diffs_by_tracker.items()):
        style = style_by_tracker.get(trk, {})
        fill_color = style.get("fill_color", "#1f77b4")
        edge_color = style.get("edge_color", "#0a3f64")
        fill_alpha = style.get("fill_alpha", 0.35)

        x, counts = _add_hist_polygon(
            fig, diffs,
            row=row, col=col,
            max_frames=max_frames,
            fill_color=fill_color,
            edge_color=edge_color,
            fill_alpha=fill_alpha,
            name=style.get("name", trk),
            showlegend=showlegend,
            fps=fps,
        )

        if counts.size:
            all_counts_max = max(all_counts_max, int(counts.max()))

        x_ms = x * (1000.0 / fps)
    
        mu = float(np.mean(x_ms)) if x.size else float("nan")
        sd = float(np.std(x_ms, ddof=1)) if x.size >= 2 else float("nan")
        # if np.isfinite(mu) and np.isfinite(sd):
        #     stats_lines.append(f"{style.get('name', trk)}: μ={mu:+.2f}±{sd:.2f}")
        # else:
        #     stats_lines.append(f"{style.get('name', trk)}: μ=n/a")
        
        label = f"{style.get('name', trk)}: μ={mu:+.1f}±{sd:.1f}" if np.isfinite(mu) else f"{style.get('name', trk)}: μ=n/a"
        stats_lines.append(label)

    

    print(f"\n[{title}]")
    for line in stats_lines:
        print("  ", line)

    # axis & title
    fig.update_yaxes(range=[0, all_counts_max * 1.08], row=row, col=col)
    fig.update_yaxes(title_text="<b>Count</b>" if show_ylabel else "", row=row, col=col)
    frame_ms = 1000.0 / fps
    tickvals = [v * frame_ms for v in range(-max_frames, max_frames + 1)]
    ticktext = [(f"{v:.0f}" if v == 0 else f"{v:+.0f}") for v in tickvals]

    fig.update_xaxes(
        range=[(-max_frames - 0.5) * frame_ms, (max_frames + 0.5) * frame_ms],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        row=row, col=col,
    )

    # Bold subplot title
    annotation_idx = (row - 1) * ncols + (col - 1)
    fig.layout.annotations[annotation_idx].update(text=f"<b>{title}</b>")

    fig.update_yaxes(showgrid=False, row=row, col=col)
    fig.update_xaxes(showgrid=False, row=row, col=col)


def collect_event_diffs(differences_per_tracker, trackers: list[str], event_name: str) -> dict[str, list[int]]:
    out = {}
    for trk in trackers:
        diffs = []
        for (t, evt), d in differences_per_tracker.items():
            if t != trk:
                continue
            if str(evt).lower() == event_name:
                diffs.extend(d)
        out[trk] = diffs
    return out


# ------------------------
# FIGURE 1: mediapipe only
# ------------------------
STYLE = {
    "mediapipe": dict(name="MediaPipe", fill_color="#1f77b4", edge_color="#0a3f64", fill_alpha=0.45),
    "rtmpose":   dict(name="RTMPose",   fill_color="#ff7f0e", edge_color="#b35a00", fill_alpha=0.35),
    "vitpose":   dict(name="ViTPose",   fill_color="#006D43", edge_color="#004d29", fill_alpha=0.35),
}

fig1 = make_subplots(
    rows=1, cols=2,
    shared_yaxes=True,
    subplot_titles=("Heel strike", "Toe off"),
    horizontal_spacing=0.06,
)

hs = collect_event_diffs(differences_per_tracker, ["mediapipe"], "heel_strike")
to = collect_event_diffs(differences_per_tracker, ["mediapipe"], "toe_off")

add_frame_histogram_panel_overlay(fig1, hs, row=1, col=1, ncols=2, title="Heel strike", max_frames=3, show_ylabel=True,  style_by_tracker=STYLE, showlegend=False)
add_frame_histogram_panel_overlay(fig1, to, row=1, col=2, ncols=2, title="Toe off",     max_frames=3, show_ylabel=False, style_by_tracker=STYLE, showlegend=False)

fig1.update_layout(
    width=500,
    height=300,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=70),
)
fig1.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, title_text="<b>Error (frames)</b>")
fig1.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

fig1.show()


# ------------------------
# FIGURE 2: overlay (TRACKERS)
# ------------------------
# set the overlay trackers you want here:
OVERLAY_TRACKERS = ["mediapipe", "rtmpose", "vitpose"]  # add more if needed

fig2 = make_subplots(
    rows=1, cols=2,
    shared_yaxes=True,
    subplot_titles=("Heel strike", "Toe off"),
    horizontal_spacing=0.06,
)

hs2 = collect_event_diffs(differences_per_tracker, OVERLAY_TRACKERS, "heel_strike")
to2 = collect_event_diffs(differences_per_tracker, OVERLAY_TRACKERS, "toe_off")

add_frame_histogram_panel_overlay(fig2, hs2, row=1, col=1, ncols=2, title="Heel strike", max_frames=3, show_ylabel=True,  style_by_tracker=STYLE, showlegend=True)
add_frame_histogram_panel_overlay(fig2, to2, row=1, col=2, ncols=2, title="Toe off",     max_frames=3, show_ylabel=False, style_by_tracker=STYLE, showlegend=False)

fig2.update_layout(
    width=560,
    height=320,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=110),
    legend=dict(
        x=0.5,
        y=-0.32,          # pushes legend below x-axis title
        xanchor="center",
        yanchor="top",
        orientation="h",
        title_text="",
        font=dict(size=11)
    )
)
fig2.update_xaxes(title_font = dict(size = 12),
                  tickfont = dict(size = 10), 
                  showline=True, linewidth=1, linecolor="black", mirror=True, title_text="<b>Error (ms)</b>")
fig2.update_yaxes(title_font = dict(size = 12),
                    tickfont = dict(size = 10), 
                  showline=True, linewidth=1, linecolor="black", mirror=True)

fig2.show()

fig2.write_image(r"D:/validation/gait_events_overlay.png", scale=3)