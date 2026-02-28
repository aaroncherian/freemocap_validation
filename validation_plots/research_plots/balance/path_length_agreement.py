import pandas as pd
import pingouin as pg
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


@dataclass
class PlotConfig:
    reference_system: str = "qualisys"
    freemocap_trackers: tuple[str] = ("mediapipe",)

    plot_height: int = 450
    plot_width: int = 1200    

    subplot_title_font = dict(size=20)
    condition_order = (
        "Eyes Open/Solid Ground",
        "Eyes Closed/Solid Ground",
        "Eyes Open/Foam",
        "Eyes Closed/Foam",
    )

    agreement_x_title = "<b>FMC-MediaPipe path length (mm)</b>"
    agreement_y_title = "<b>Reference path length (mm)</b>"

    altman_x_title = "<b>Mean of systems (mm)</b>"
    altman_y_title = "<b>Difference between systems (mm)</b>"

    axis_title_font = dict(family="Arial", size=20)
    axis_tickfont = dict(size=16)


    def __post_init__(self):
        self.all_trackers = list(self.freemocap_trackers) + [self.reference_system]


def query_df(path_to_db: Path, trackers: list[str]):
    placeholders = ",".join(["?"] * len(trackers))

    query = f"""
        SELECT
            t.participant_code,
            t.trial_name,
            a.path,
            a.condition,
            a.tracker
        FROM artifacts a
        JOIN trials t ON a.trial_id = t.id
        WHERE
            t.trial_type = "balance"
            AND a.category = "com_analysis"
            AND a.tracker IN ({placeholders})
            AND a.file_exists = 1
            AND a.component_name LIKE '%path_length_com%'
        ORDER BY
            t.trial_name, a.path;
            """

    conn = sqlite3.connect(path_to_db)
    return pd.read_sql_query(query, conn, params=trackers)


def parse_database_df(df: pd.DataFrame, cfg: PlotConfig) -> pd.DataFrame:
    dfs = []
    for _, row in df.iterrows():
        sub = load_path_length_json(row["path"])
        sub["participant_code"] = row["participant_code"]
        sub["trial_name"] = row["trial_name"]
        sub["tracker"] = row["tracker"]
        dfs.append(sub)

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df["condition"] = pd.Categorical(
        combined_df["condition"], categories=cfg.condition_order, ordered=True
    )

    return combined_df


def load_path_length_json(json_path: str) -> pd.DataFrame:
    """
    Loads COM path length JSON artifact into a tidy dataframe with:
      condition, path_length, frame_interval (if present)
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {json_path}")

    raw = pd.read_json(p)

    col_map = {
        "Frame Intervals": "frame_interval",
        "Frame Interval": "frame_interval",
        "Path Lengths:": "path_length",
        "Path Lengths": "path_length",
        "Path Length": "path_length",
    }
    raw = raw.rename(columns=col_map)

    # If conditions are stored as the index, move them to a column.
    raw = raw.reset_index().rename(columns={"index": "condition"})

    # Keep only what we need
    keep = [c for c in ["condition", "path_length", "frame_interval"] if c in raw.columns]
    out = raw[keep].copy()

    # Ensure numeric
    out["path_length"] = pd.to_numeric(out["path_length"], errors="coerce")

    return out


def calculate_ICC(path_length_df: pd.DataFrame, cfg: PlotConfig):
    path_length_df["target"] = (
        path_length_df[["trial_name", "condition"]].astype(str).agg("|".join, axis=1)
    )

    icc_overall = pg.intraclass_corr(
        data=path_length_df,
        targets="target",
        raters="tracker",
        ratings="path_length",
    )

    icc_overall.set_index("Type")

    grouped = path_length_df.groupby("condition")

    icc_rows = []
    for condition, group in grouped:
        icc = pg.intraclass_corr(
            data=group,
            targets="target",
            raters="tracker",
            ratings="path_length",
        )
        row = icc.query("Type == 'ICC2'").iloc[0]
        icc_rows.append(
            {
                "condition": condition,
                "ICC": row["ICC"],
                "CI95%": row["CI95%"],
            }
        )

    overall_row = {
        "condition": "overall",
        "ICC": icc_overall.query("Type == 'ICC2'").iloc[0]["ICC"],
        "CI95%": icc_overall.query("Type == 'ICC2'").iloc[0]["CI95%"],
    }

    icc_rows.append(overall_row)
    return pd.DataFrame(icc_rows)


def get_bland_altman_stats(differences: pd.Series) -> dict[str, float]:
    mean = np.mean(differences)
    std = np.std(differences, ddof=1)
    loa_upper = mean + 1.96 * std
    loa_lower = mean - 1.96 * std

    return {"mean": mean, "std": std, "loa_upper": loa_upper, "loa_lower": loa_lower}


def calculate_bland_altman(path_length_df: pd.DataFrame, cfg: PlotConfig):
    path_length_wide = path_length_df.pivot(
        index=["condition", "participant_code", "trial_name"],
        columns="tracker",
        values="path_length",
    ).reset_index()

    overall_ba = path_length_wide.copy()
    overall_ba["ba_difference"] = overall_ba["mediapipe"] - overall_ba[cfg.reference_system]
    overall_ba["ba_mean"] = (overall_ba[cfg.reference_system] + overall_ba["mediapipe"]) / 2

    differences = overall_ba["ba_difference"]

    ba_rows = []
    overall_ba_stats = get_bland_altman_stats(differences)
    overall_ba_stats["condition"] = "overall"

    ba_rows.append(overall_ba_stats)

    grouped = path_length_wide.groupby("condition")
    for condition, group in grouped:
        group["ba_difference"] = group["mediapipe"] - group[cfg.reference_system]
        group["ba_mean"] = (group[cfg.reference_system] + group["mediapipe"]) / 2
        differences = group["ba_difference"]
        stats = get_bland_altman_stats(differences)
        stats["condition"] = condition
        ba_rows.append(stats)

    ba_stats = pd.DataFrame(ba_rows)
    return overall_ba, ba_stats

def calculate_regression_equation(path_length_df: pd.DataFrame, tracker: list[str], reference: str) -> dict:
    
    regression_df = path_length_df.pivot_table(
        index = ["condition", "participant_code", "trial_name", "target"],
        columns = "tracker",
        values = "path_length"
    ).reset_index()

    x = regression_df[reference].to_numpy()
    y = regression_df[tracker].to_numpy()

    m,b = np.polyfit(x,y,1)

    return m,b
    f = 2

def make_agreement_and_ba_subplot_figure(
    path_length_df: pd.DataFrame,
    overall_ba_df: pd.DataFrame,
    ba_stats: pd.DataFrame,
    reg_eqn: tuple[float, float],
    icc: float, 
    cfg: PlotConfig,
    tracker: str = "mediapipe",
    color_by_condition: bool = True,
    height: int = 450,
    width: int = 1050,
) -> go.Figure:
    """
    Left: paired agreement scatter (Qualisys vs tracker) + identity line
    Right: pooled Bland–Altman (mean vs diff) + bias/LoA lines
    Uses:
      - path_length_df (long)
      - overall_ba_df (your 'overall_ba' from calculate_bland_altman)
      - ba_stats (your 'ba_stats' from calculate_bland_altman)
    """

    # --- Agreement wide data (paired points) ---
    agree = (
        path_length_df[path_length_df["tracker"].isin([cfg.reference_system, tracker])]
        .pivot_table(
            index=["participant_code", "trial_name", "condition"],
            columns="tracker",
            values="path_length",
            aggfunc="first",
        )
        .dropna(subset=[cfg.reference_system, tracker])
        .reset_index()
    )

    # Global range for agreement plot
    all_vals = np.concatenate(
        [agree[cfg.reference_system].to_numpy(), agree[tracker].to_numpy()]
    )
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    pad = 0.05 * (vmax - vmin) if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 0.01
    agree_range = [vmin - pad, vmax + pad]

    # --- BA pooled lines from your ba_stats "overall" row ---
    overall_row = ba_stats.loc[ba_stats["condition"] == "overall"].iloc[0]
    bias = float(overall_row["mean"])
    loa_upper = float(overall_row["loa_upper"])
    loa_lower = float(overall_row["loa_lower"])

    # X span for BA lines
    x_min = float(overall_ba_df["ba_mean"].min())
    x_max = float(overall_ba_df["ba_mean"].max())
    x_pad = 0.08 * (x_max - x_min) if np.isfinite(x_max - x_min) and (x_max - x_min) > 0 else 0.01
    x0, x1 = x_min - x_pad, x_max + x_pad

    # Symmetric y-lims on BA
    m = max(abs(loa_upper), abs(loa_lower))
    y_range = [-m * 1.15, m * 1.15] if np.isfinite(m) and m > 0 else [-1, 1]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
    )

    # ---------- Left panel: agreement ----------
    fig.add_trace(
        go.Scatter(
            x=agree[cfg.reference_system],
            y=agree[tracker],
            mode="markers",
            marker=dict(size=8, opacity=0.75),
            showlegend=False,
            hovertemplate=(
                "Participant: %{customdata[0]}<br>"
                "Trial: %{customdata[1]}<br>"
                "Condition: %{customdata[2]}<br>"
                f"{cfg.reference_system}: %{{x:.4f}} mm<br>"
                f"{tracker}: %{{y:.4f}} mm<extra></extra>"
            ),
            customdata=agree[["participant_code", "trial_name", "condition"]].astype(str).values,
        ),
        row=1,
        col=1,
    )

    # Identity line
    fig.add_trace(
        go.Scatter(
            x=agree_range,
            y=agree_range,
            mode="lines",
            line=dict(dash="dash", color = "black"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    x_line = np.array(agree_range)
    y_line = reg_eqn[0]*x_line + reg_eqn[1]

    fig.add_trace(
        go.Scatter(
            x = x_line,
            y = y_line,
            mode = "lines",
            line = dict(color = "red", width = 1.5),
            showlegend=False,
            opacity = .7,
        ),
        row = 1, col = 1,
    )

    fig.add_annotation(
        x=0.95, y=0.05,
        xref = "x domain",
        yref = "y domain",
        text=f"ICC(2,1) = {icc:.3f}<br>slope = {reg_eqn[0]:.2f}",
        showarrow=False,
        font=dict(size=12),
        xanchor="right",
        yanchor="bottom",
    )
    

    fig.update_xaxes(title_text=cfg.agreement_x_title, 
                     range=agree_range, 
                     row=1, 
                     col=1,
                     title_font=cfg.axis_title_font,
                     tickfont=cfg.axis_tickfont)
    fig.update_yaxes(title_text=cfg.agreement_y_title, 
                     range=agree_range, 
                     row=1, 
                     col=1,
                     title_font=cfg.axis_title_font,
                     tickfont=cfg.axis_tickfont
                     )

    # ---------- Right panel: pooled BA ----------
    if color_by_condition:
        # simple deterministic palette; keep it lightweight
        conds = list(cfg.condition_order)
        # Plotly default palette is fine; but we’ll set explicit mapping for stability
        colors = sample_colorscale(
            "Viridis",
            [0.15 + 0.6*(i/(len(conds)-1)) for i in range(len(conds))]
        )
        color_map = dict(zip(conds, colors))

        for cond, sub in overall_ba_df.groupby("condition"):
            fig.add_trace(
                go.Scatter(
                    x=sub["ba_mean"],
                    y=sub["ba_difference"],
                    mode="markers",
                    name=str(cond),
                    marker=dict(size=9, opacity=0.8, color=color_map.get(cond)),
                    hovertemplate=(
                        "Participant: %{customdata[0]}<br>"
                        "Trial: %{customdata[1]}<br>"
                        "Condition: %{customdata[2]}<br>"
                        "Mean: %{x:.4f} mm<br>"
                        "Diff: %{y:.4f} mm<extra></extra>"
                    ),
                    customdata=sub[["participant_code", "trial_name", "condition"]].astype(str).values,
                ),
                row=1,
                col=2,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=overall_ba_df["ba_mean"],
                y=overall_ba_df["ba_difference"],
                mode="markers",
                marker=dict(size=9, opacity=0.8),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Bias + LoA lines (note xref/yref MUST be x2/y2 for col=2)
    fig.add_shape(
        type="line",
        x0=x0, x1=x1, y0=bias, y1=bias,
        xref="x2", yref="y2",
        line=dict(color="black", width=2, dash="dash"),
    )
    for y in (loa_upper, loa_lower):
        fig.add_shape(
            type="line",
            x0=x0, x1=x1, y0=y, y1=y,
            xref="x2", yref="y2",
            line=dict(color="black", width=1.5, dash="dot"),
        )

    fig.update_xaxes(title_text=cfg.altman_x_title, 
                     row=1, 
                     col=2,
                     title_font=cfg.axis_title_font,
                     tickfont=cfg.axis_tickfont
                     )
    
    fig.update_yaxes(
        title_text=cfg.altman_y_title,
        range=y_range,
        row=1,
        col=2,
        title_font=cfg.axis_title_font,
        tickfont=cfg.axis_tickfont
    )

    fig.update_layout(
        template="simple_white",
        height=height,
        width=width,
        margin=dict(t=90, b=70, l=80, r=40),
        legend_title_text="Condition" if color_by_condition else None,
    )

    fig.update_annotations(font_size=cfg.subplot_title_font["size"])

    return fig


if __name__ == "__main__":
    cfg = PlotConfig()
    root_path = Path(
        r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\balance"
    )
    root_path.mkdir(exist_ok=True, parents=True)

    path_to_db = Path(r"validation.db")

    db_df = query_df(path_to_db, cfg.all_trackers)

    path_length_df = parse_database_df(db_df, cfg)
    icc_df = calculate_ICC(path_length_df, cfg)

    ba_plot_df, ba_stats = calculate_bland_altman(path_length_df, cfg)

    slope, intercept = calculate_regression_equation(path_length_df=path_length_df, 
                                  tracker = "mediapipe",
                                   reference = cfg.reference_system)
    # NEW: combined figure
    fig = make_agreement_and_ba_subplot_figure(
        path_length_df=path_length_df,
        overall_ba_df=ba_plot_df,
        ba_stats=ba_stats,
        reg_eqn = (slope, intercept),
        icc = icc_df.query("condition == 'overall'")['ICC'].item(),
        cfg=cfg,
        tracker="mediapipe",
        color_by_condition=True,
        height=cfg.plot_height,
        width=cfg.plot_width,
    )
    fig.show()

    fig.write_image(root_path / "com_path_length_agreement_ba.svg", scale=3)

    f = 2