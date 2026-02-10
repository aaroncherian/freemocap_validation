import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

conn = sqlite3.connect("validation.db")
import numpy as np

query = """
SELECT t.participant_code, 
        t.trial_name,
        a.path,
        a.component_name,
        a.condition,
        a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "balance"
    AND a.category = "com_analysis"
    AND a.tracker IN ("mediapipe", "qualisys", "rtmpose")
    AND a.file_exists = 1
    AND a.component_name LIKE '%path_length_com'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)
REFERENCE_SYSTEM = "qualisys"
TRACKERS = ["mediapipe"]

dfs = []

for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    condition = row.get("condition") or ""  # handle None/empty
    participant = row["participant_code"]
    trial = row["trial_name"]

    # Load file — autodetect type
    sub_df = pd.read_json(path)
    sub_df = sub_df.rename(columns={"Frame Intervals": "frame_interval",
                                    "Path Lengths:": "path_length"}).reset_index()
    sub_df = sub_df.rename(columns={"index": "condition"})
    # Add metadata columns
    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker

    dfs.append(sub_df)

# Concatenate all into one tidy DataFrame
combined_df = pd.concat(dfs, ignore_index=True)


condition_order = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]
display_x = [c.replace('/', '<br>') for c in condition_order]


def manipulation_eyes_effect(df:pd.DataFrame, surface:str):
    if surface == "solid":
        surface_df = df.query("surface == 'Solid Ground'").copy()
    elif surface == "foam":
        surface_df = df.query("surface == 'Foam'").copy()
    else:
        raise ValueError(f"Unknown surface '{surface}' for manipulation_eyes_effect")

    id_cols = [
        "participant_code",
        "trial_name",
        "tracker"
    ]

    surface_wide = (
        surface_df.pivot_table(
            index = id_cols,
            columns = "eyes",
            values = "path_length",
            aggfunc = "first"
        )
        .reset_index()
    )

    surface_wide["difference"] = surface_wide["Closed"] - surface_wide["Open"]
    surface_wide["manipulation"] = f"eyes_on_{surface}"

    return surface_wide[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]


def manipulation_foam_effect(df: pd.DataFrame, eyes:str):
    if eyes == "open":
        eyes_df = df.query("eyes == 'Open'").copy()
    elif eyes == "closed":
        eyes_df = df.query("eyes == 'Closed'").copy()
    else:
        raise ValueError(f"Unknown eye condition '{eyes}")
    
    id_cols = [
        "participant_code",
        "trial_name",
        "tracker"
    ]

    eyes_wide = (
        eyes_df.pivot_table(
            index = id_cols,
            columns = "surface",
            values = "path_length",
            aggfunc = "first"
        )
        .reset_index()
    )

    eyes_wide["difference"] = eyes_wide["Foam"] - eyes_wide["Solid Ground"]
    eyes_wide["manipulation"] = f"foam_with_{eyes}"
    return eyes_wide[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]
    f = 2

def manipulation_hardest_easiest(df: pd.DataFrame):

    easy_condition = df.query(("eyes == 'Open' & surface == 'Solid Ground'")).copy()
    hard_condition = df.query(("eyes == 'Closed' & surface == 'Foam'")).copy()

    contrast_df = easy_condition.merge(
        hard_condition,
        on = ["participant_code", "trial_name", "tracker"],
        suffixes=("_easy", "_hard")
    )

    contrast_df["difference"] = contrast_df["path_length_hard"] - contrast_df["path_length_easy"]
    contrast_df["manipulation"] = "hardest_vs_easiest"
    return contrast_df[["participant_code", "trial_name", "tracker", "difference", "manipulation"]]
    f = 2

id_cols = [
    "participant_code",
    "condition",
    "trial_name",
]


cond = combined_df["condition"].str.extract(r"Eyes\s+(Open|Closed)\s*/\s*(Solid Ground|Foam)")
combined_df["eyes"] = cond[0]
combined_df["surface"] = cond[1]


eyes_on_solid_effect = manipulation_eyes_effect(combined_df, "solid")
eyes_on_foam_effect = manipulation_eyes_effect(combined_df, "foam")
foam_with_eyes_open = manipulation_foam_effect(combined_df, eyes = "open")
foam_with_eyes_closed = manipulation_foam_effect(combined_df, eyes = "closed")
hardest_vs_easiest = manipulation_hardest_easiest(combined_df)


manipulation_df = pd.concat(
    [
        eyes_on_solid_effect,
        foam_with_eyes_open,
        hardest_vs_easiest
    ],
    ignore_index=True
)

manip_order = [
    "eyes_on_solid",
    "foam_with_open",
    "hardest_vs_easiest",
]
f = 2

MANIP_TITLES = {
    "eyes_on_solid": "Eyes Closed − Eyes Open (Solid Ground)",
    "foam_with_open": "Eyes Open (Foam) - Eyes Open (Solid Ground)",
    "hardest_vs_easiest": "Eyes Closed (Foam) - Eyes Open (Solid Ground)",
}
X_LABEL = "Reference Δ COM path length (mm)"
Y_LABEL = "Pose Estimation Δ COM path length (mm)"
# --- NEW: cleare

def limits_zoom_positive(
    sub: pd.DataFrame,
    cols,
    *,
    neg_buffer_frac: float = 0.15,   # show a bit below 0
    margin_frac: float = 0.08,       # padding around data
    min_neg_buffer: float = 0.02,    # absolute minimum negative space
):
    """
    Data-driven limits that keep 0 visible but don't waste 50% of the panel.
    Uses combined x/y values (Qualisys + trackers) so the y=x line is meaningful.
    """
    vals = sub[cols].to_numpy().astype(float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1, 1)

    vmin = float(vals.min())
    vmax = float(vals.max())
    span = max(vmax - vmin, 1e-9)

    # positive-focused negative buffer based on overall positive magnitude
    pos_scale = max(abs(vmax), 1e-9)
    neg_buffer = max(min_neg_buffer, neg_buffer_frac * pos_scale)

    # upper padding
    upper = vmax + margin_frac * span

    # lower: include observed negatives if present, else small negative buffer
    lower = min(vmin - margin_frac * span, -neg_buffer)

    # ensure 0 is inside
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)

    # make x and y share same limits (square-ish interpretability)
    lim = max(abs(lower), abs(upper))
    # but *not* symmetric: we want more positive room; keep lower as computed
    # We'll return lower/upper, and you set both x and y to these.
    return (lower, upper)


contrast_wide = (
    manipulation_df
    .pivot_table(
        index=["participant_code", "trial_name", "manipulation"],
        columns="tracker",
        values="difference",
        aggfunc="first"
    )
    .reset_index()
)



tracker_styles = {
    "mediapipe": dict(color="#1f77b4", symbol="circle"),
    "rtmpose": dict(color="#d62728", symbol="diamond"),
}
fig = make_subplots(
    rows=1,
    cols=len(manip_order),
    subplot_titles=[MANIP_TITLES.get(m, m) for m in manip_order],
    shared_xaxes=False,
    shared_yaxes=False,
)

for col, manipulation in enumerate(manip_order, start=1):
    sub = contrast_wide.query("manipulation == @manipulation")

    # Identity line
    all_vals = sub[["qualisys"] + TRACKERS].to_numpy().flatten()
    all_vals = all_vals[pd.notna(all_vals)]
    
    cols_for_limits = ["qualisys"] + TRACKERS
    lower, upper = limits_zoom_positive(sub, cols_for_limits)
    
    xaxis_ref = f"x{col}" if col > 1 else "x"
    yaxis_ref = f"y{col}" if col > 1 else "y"

    fig.add_shape(
        type="line", x0=lower, x1=upper, y0=0, y1=0,
        line=dict(color="darkgrey", width=1.5, dash="dot"),
        xref=xaxis_ref, yref=yaxis_ref,
    )
    fig.add_shape(
        type="line", x0=0, x1=0, y0=lower, y1=upper,
        line=dict(color="darkgrey", width=1.5, dash="dot"),
        xref=xaxis_ref, yref=yaxis_ref,
    )

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Identity (tracker Δ = Qualisys Δ)",
            line=dict(color="black", dash="dash"),
            showlegend=(col == 1),  # only show once
        ),
        row=1, col=col
    )

    for tracker in TRACKERS:
        fig.add_trace(
            go.Scatter(
                x=sub["qualisys"],
                y=sub[tracker],
                mode="markers",
                name=tracker,                 # always the same name
                legendgroup=tracker,          # group across subplots
                showlegend=(col == 1),        # only show once
                marker=dict(
                    size=9,
                    opacity=0.7,
                    **tracker_styles[tracker],
                ),
                line=dict(color='white', width=0.5)
            ),
            row=1,
            col=col,
        )

    fig.update_xaxes(
        title_text=X_LABEL,
        range=[lower, upper],
        row=1,
        col=col,
        
    )


    fig.update_yaxes(
        title_text=Y_LABEL if col == 1 else None,   # only label left-most y-axis
        range=[lower, upper],
        row=1,
        col=col,
    )
fig.update_annotations(font_size=12)
fig.update_xaxes(
    title_font=dict(size=12)
)

fig.update_yaxes(
    title_font=dict(size=12)
)
fig.update_layout(
    height=400,
    width=1000,
    template="simple_white",
    legend=dict(
        title_text="",          # no legend title
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.25,                # push below x-axes (tune between -0.1 and -0.25)
        yanchor="top",
    ),
    margin=dict(t=80, b=120),   # extra bottom margin for legend
)
fig.show()



# import numpy as np
# import plotly.graph_objects as go


# TRACKER_ORDER = ["qualisys", "mediapipe", "rtmpose"]  # edit if you want
# tracker_symbols = {"qualisys": "circle", "mediapipe": "square", "rtmpose": "diamond"}

# fig = go.Figure()

# for trk in TRACKER_ORDER:
#     d = manipulation_df.query("tracker == @trk").copy()
#     if d.empty:
#         continue

#     # map manipulations to numeric x positions for jitter
#     x_map = {m: i for i, m in enumerate(manip_order)}
#     x = d["manipulation"].map(x_map).astype(float).to_numpy()

#     # jitter so trackers separate slightly
#     jitter = {"qualisys": -0.18, "mediapipe": 0.0, "rtmpose": 0.18}.get(trk, 0.0)
#     x = x + jitter + 0.03*np.random.randn(len(x))  # small random jitter

#     fig.add_trace(go.Scatter(
#         x=x,
#         y=d["difference"],
#         mode="markers",
#         name=trk,
#         marker=dict(size=9, symbol=tracker_symbols.get(trk, "circle")),
#         customdata=np.stack([d["participant_code"], d["trial_name"], d["manipulation"]], axis=1),
#         hovertemplate=(
#             "tracker=%{name}<br>"
#             "participant=%{customdata[0]}<br>"
#             "trial=%{customdata[1]}<br>"
#             "manip=%{customdata[2]}<br>"
#             "diff=%{y:.4f}<extra></extra>"
#         )
#     ))

# fig.update_xaxes(
#     tickmode="array",
#     tickvals=list(range(len(manip_order))),
#     ticktext=[m.replace("_", "<br>") for m in manip_order],
# )
# fig.update_layout(
#     title="Balance path-length contrasts by manipulation and tracker",
#     yaxis_title="Path-length difference (harder − easier)",
#     xaxis_title="Manipulation",
# )

# fig.show()


# import plotly.express as px

# d = manipulation_df.copy()
# d["manipulation"] = pd.Categorical(d["manipulation"], categories=manip_order, ordered=True)

# fig = px.violin(
#     d,
#     x="manipulation",
#     y="difference",
#     color="tracker",
#     points="all",          # show points on top
#     box=True,              # mini box inside violin
#     category_orders={"manipulation": manip_order},
# )

# fig.update_layout(
#     title="Contrast distributions by manipulation",
#     xaxis_title="Manipulation",
#     yaxis_title="Path-length difference (harder − easier)",
# )
# fig.show()
