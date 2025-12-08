
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as qqual
import plotly.io as pio

import re

CONDITION_ORDER = ["neg_5_6","neg_2_8","neutral","pos_2_8","pos_5_6"]

# fixed, high-contrast palette (tweak if you like)
CONDITION_STYLE: dict[str, dict[str, str]] = {
    "neg_5_6": {"line": "#94342b", "fill": "rgba(180,76,67,0.22)"},    # red-brown
    "neg_2_8": {"line": "#d39182", "fill": "rgba(211,145,130,0.22)"},  # light clay
    "neutral": {"line": "#524F4F", "fill": "rgba(186,186,186,0.22)"},  # medium grey
    "pos_2_8": {"line": "#7bb6c6", "fill": "rgba(123,182,198,0.22)"},  # soft teal
    "pos_5_6": {"line": "#447c8e", "fill": "rgba(68,124,142,0.22)"},   # deep teal
    # "neg_6": {"line": "#2ca02c", "fill": "rgba(44,160,44,0.22)"},   # green
    # "pos_6" : {"line": "#1f77b4", "fill": "rgba(31,119,180,0.22)"}   # blue
}

def _rgb_from_hex(color: str) -> tuple[int,int,int] | None:
    """Return (r,g,b) from '#RGB' or '#RRGGBB'. None if not a hex color."""
    if not isinstance(color, str) or not color.startswith("#"):
        return None
    s = color.lstrip("#")
    if len(s) == 3:
        s = "".join(ch*2 for ch in s)  # expand #RGB -> #RRGGBB
    if len(s) != 6 or not re.fullmatch(r"[0-9A-Fa-f]{6}", s):
        return None
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)

def _hex_to_rgba(color: str, alpha: float) -> str:
    """Hex (#RGB or #RRGGBB) -> rgba(...,alpha). Fallback to neutral gray if unknown."""
    rgb = _rgb_from_hex(color)
    if rgb is None:
        return f"rgba(85,85,85,{alpha})"  # safe fallback
    r, g, b = rgb
    return f"rgba({r},{g},{b},{alpha})"





def load_condition_csvs(conditions: dict[str, Path], tracker: str = "mediapipe_dlc") -> pd.DataFrame:
    dfs = []
    for cond, root in conditions.items():
        csv_path = Path(root) / "validation" / tracker / f'{tracker}_joint_angle_by_stride.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for condition '{cond}': {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[df["angle"] == "ankle_dorsi_plantar_r"]
        df = df.rename(columns={"value": "ankle_angle"}) 
        df = df.drop(columns=["angle"])
        df["condition"] = cond
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    required = {"system","stride","percent_gait_cycle","ankle_angle","condition"}
    missing = required - set(all_df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return all_df

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summ = (
        df.groupby(["system","condition","percent_gait_cycle"])["ankle_angle"]
          .agg(["mean","std"])
          .reset_index()
    )
    summ["std"] = summ["std"].fillna(0.0)
    return summ

def global_yrange(summ: pd.DataFrame) -> tuple[float,float]:
    ymin = (summ["mean"] - summ["std"]).min()
    ymax = (summ["mean"] + summ["std"]).max()
    pad = 0.05 * (ymax - ymin + 1e-9)
    return float(ymin - pad), float(ymax + pad)

def add_band_and_mean(fig, row, col, x, mean, std, name,
                      line_color, fill_color, dash=None, showlegend=True):
    upper = mean + std
    lower = mean - std

    # upper boundary (invisible line, just anchor for fill)
    fig.add_trace(go.Scatter(
        x=x, y=upper, mode="lines",
        line=dict(width=0, color=line_color),
        hoverinfo="skip", showlegend=False
    ), row=row, col=col)

    # lower boundary (filled to previous)
    fig.add_trace(go.Scatter(
        x=x, y=lower, mode="lines",
        line=dict(width=0, color=line_color),
        fill="tonexty", fillcolor=fill_color,
        hoverinfo="skip", showlegend=False
    ), row=row, col=col)

    # mean line
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines", name=name,
        line=dict(color=line_color, dash=dash) if dash else dict(color=line_color),
        hovertemplate="%{x:.0f}% gait<br>angle=%{y:.3f}°<extra>"+name+"</extra>",
        showlegend=showlegend
    ), row=row, col=col)

def hex_to_rgb(hex_color: str):
    """
    Convert a hex color string like '#94342b' into an (r, g, b) tuple.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def family1_combined_row(
    summ: pd.DataFrame,
    out_path: Path,
    errorbar_step: int = 5,
    max_jitter: float = 1.0,
) -> Path:
    """
    Make ankle flex/ext plot with mean curves and jittered SD error bars.
    `summ` must have columns: system, condition, percent_gait_cycle, mean, std
    """
    systems = sorted(summ["system"].unique())

    # respect your canonical condition order, then append any extras
    conditions = [c for c in CONDITION_ORDER if c in set(summ["condition"].unique())]
    conditions += [c for c in sorted(summ["condition"].unique()) if c not in conditions]

    fig = make_subplots(
        rows=1,
        cols=len(systems),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=systems,
        horizontal_spacing=0.05,
    )

    ylo, yhi = global_yrange(summ)

    # ----- condition-wise jitter offsets -----
    if len(conditions) > 1:
        offsets = np.linspace(-max_jitter, max_jitter, len(conditions))
    else:
        offsets = np.array([0.0])
    cond_offset = dict(zip(conditions, offsets))

    for j, s in enumerate(systems, start=1):
        for c in conditions:
            sub = (
                summ[(summ["system"] == s) & (summ["condition"] == c)]
                .sort_values("percent_gait_cycle")
            )
            if sub.empty:
                continue

            style = CONDITION_STYLE.get(c, {"line": "#555"})

            x = sub["percent_gait_cycle"].to_numpy()
            m = sub["mean"].to_numpy()
            sd = sub["std"].to_numpy()

            # ===== mean line =====
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=m,
                    mode="lines",
                    name=c if j == 1 else None,
                    legendgroup=c,
                    line=dict(color=style["line"], width=3),
                    showlegend=(j == 1),
                    hovertemplate=(
                        f"<b>{s} – {c}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Angle: %{y:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=j,
            )

            # ===== SD error bars at subsampled points, jittered =====
            if len(x) == 0:
                continue

            idx = np.arange(0, len(x), errorbar_step)
            x_base = x[idx]
            m_err = m[idx]
            sd_err = sd[idx]

            x_err = x_base + cond_offset[c]
            # keep first/last bars anchored on the curve
            if x_err.size > 0:
                x_err[0] = x_base[0]
                x_err[-1] = x_base[-1]

            r, g, b = hex_to_rgb(style["line"])

            fig.add_trace(
                go.Scatter(
                    x=x_err,
                    y=m_err,
                    mode="markers",
                    name=None,
                    legendgroup=c,
                    showlegend=False,
                    marker=dict(
                        color=style["line"],
                        size=6,
                        symbol="circle-open",
                        opacity=0.5,
                    ),
                    error_y=dict(
                        type="data",
                        array=sd_err,
                        visible=True,
                        symmetric=True,
                        thickness=1.0,
                        width=2,
                        color=f"rgba({r},{g},{b},0.45)",
                    ),
                    hovertemplate=(
                        f"<b>{s} – {c}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Mean angle: %{y:.1f}°<br>"
                        "SD: %{error_y.array:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=j,
            )

        fig.update_yaxes(range=[ylo, yhi], row=1, col=j)

    # axis labels & layout
    for j in range(1, len(systems) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            row=1,
            col=j,
            title_font=dict(size=24),
        )
    fig.update_yaxes(
        title_text="<b>Angle (°)</b>",
        row=1,
        col=1,
        title_font=dict(size=24),
    )

    fig.update_layout(
        title="<b>Ankle flexion/extension per system: conditions overlaid (mean ± SD bars)</b>",
        template="plotly_white",
        height=840,
        width=1200 * len(systems),
        font=dict(size=24),
    )

    fig.write_html(out_path)
    return out_path

def family2_combined_row(summ: pd.DataFrame, out_path: Path,
                      qualisys_key: str = "qualisys",
                      freemocap_key: str = "mediapipe_dlc") -> Path:
    conditions = [c for c in CONDITION_ORDER if c in set(summ["condition"].unique())]
    conditions += [c for c in sorted(summ["condition"].unique()) if c not in conditions]

    fig = make_subplots(
        rows=2, cols=len(conditions),
        shared_xaxes=True, shared_yaxes=False,
        row_heights=[0.68, 0.32],
        subplot_titles=conditions
    )
    ylo, yhi = global_yrange(summ)

    # --- Row 1: your current overlays (same color per condition; QTM solid, FMC dashed)
    sys_band_alpha = {qualisys_key.lower(): 0.28, freemocap_key.lower(): 0.14}
    def dash_for(sys: str) -> str | None:
        return None if sys.lower()==qualisys_key.lower() else "dash"

    for j, c in enumerate(conditions, start=1):
        style = CONDITION_STYLE.get(c, {"line":"#555","fill":"rgba(85,85,85,0.18)"})
        # derive RGBA for system-specific band alpha
        r = int(style["line"][1:3],16); g = int(style["line"][3:5],16); b = int(style["line"][5:7],16)

        for s in [qualisys_key, freemocap_key]:
            sub = summ[(summ["system"]==s) & (summ["condition"]==c)].sort_values("percent_gait_cycle")
            if sub.empty:
                continue
            alpha = sys_band_alpha.get(s.lower(), 0.18)
            fill = _hex_to_rgba(style["line"], alpha)
            add_band_and_mean(
                fig, 1, j,
                sub["percent_gait_cycle"], sub["mean"], sub["std"],
                name=s, line_color=style["line"], fill_color=fill,
                dash=dash_for(s), showlegend=(j==1)
            )
        

        fig.update_yaxes(range=[ylo, yhi], row=1, col=j, title_text="Angle (°)" if j==1 else None)

    # --- Row 2: Δ = QTM − FMC (mean) with 95% CI using SEMs
    for j, c in enumerate(conditions, start=1):
        q = summ[(summ["system"]==qualisys_key) & (summ["condition"]==c)].sort_values("percent_gait_cycle")
        f = summ[(summ["system"]==freemocap_key) & (summ["condition"]==c)].sort_values("percent_gait_cycle")
        if q.empty or f.empty:
            continue
        # align on percent_gait_cycle just in case
        merged = pd.merge(q, f, on="percent_gait_cycle", suffixes=("_q","_f"))
        x = merged["percent_gait_cycle"]
        delta = merged["mean_q"] - merged["mean_f"]
        # 95% CI using SEM: sqrt(SEM_q^2 + SEM_f^2) * 1.96
        sem_q = merged.get("sem_q", pd.Series(0.0, index=merged.index))
        sem_f = merged.get("sem_f", pd.Series(0.0, index=merged.index))
        ci95 = 1.96 * np.sqrt(sem_q**2 + sem_f**2)

        style = CONDITION_STYLE.get(c, {"line":"#555","fill":"rgba(85,85,85,0.18)"})
        # thin, muted band for CI (same hue)
        r = int(style["line"][1:3],16); g = int(style["line"][3:5],16); b = int(style["line"][5:7],16)
        fill_ci = f"rgba({r},{g},{b},0.18)"

        # band
        fig.add_trace(go.Scatter(x=x, y=(delta + ci95), mode="lines",
                                 line=dict(width=0, color=style["line"]), showlegend=False, hoverinfo="skip"),
                      row=2, col=j)
        fig.add_trace(go.Scatter(x=x, y=(delta - ci95), mode="lines",
                                 line=dict(width=0, color=style["line"]), fill="tonexty",
                                 fillcolor=fill_ci, showlegend=False, hoverinfo="skip"),
                      row=2, col=j)
        # mean Δ
        fig.add_trace(go.Scatter(x=x, y=delta, mode="lines", name="Δ (Q − FMC)",
                                 line=dict(color=style["line"], width=2.2),
                                 hovertemplate="%{x:.0f}% gait<br>Δ=%{y:.3f}°<extra>Δ (Q − FMC)</extra>",
                                 showlegend=(j==1)),
                      row=2, col=j)

        # zero line for reference
        fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.25)", width=1), row=2, col=j)
        fig.update_yaxes(title_text="Δ (°)" if j==1 else None, row=2, col=j)

    for j in range(1, len(conditions)+1):
        fig.update_xaxes(title_text="Gait cycle (%)", row=2, col=j)

    fig.update_layout(title="Ankle plantarflexion/dorsiflexion angle — per system: systems overlaid (top) and Δ = Qualisys − FreeMoCap (bottom)",
                      template="plotly_white", height=580, width=380*len(conditions))
    fig.update_layout(width=1200, height=800)
    fig.write_html(out_path)
    return out_path

def family3_grid_with_neutral(summ: pd.DataFrame, out_path: Path, neutral_name: str = "neutral") -> Path:
    systems = sorted(summ["system"].unique())
    conditions = [c for c in CONDITION_ORDER if c in set(summ["condition"].unique())]
    conditions += [c for c in sorted(summ["condition"].unique()) if c not in conditions]
    if neutral_name not in conditions:
        raise ValueError(f"Neutral baseline '{neutral_name}' not in: {conditions}")

    rows, cols = len(systems), len(conditions)
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[f"<b>{s} · {c}</b>" for s in systems for c in conditions])
    ylo, yhi = global_yrange(summ)

    neutral = {
        s: summ[(summ["system"]==s) & (summ["condition"]==neutral_name)].sort_values("percent_gait_cycle")
        for s in systems
    }

    for i, s in enumerate(systems, start=1):
        base = neutral.get(s)
        for j, c in enumerate(conditions, start=1):
            sub = summ[(summ["system"]==s) & (summ["condition"]==c)].sort_values("percent_gait_cycle")

            if base is not None and not base.empty:
                fig.add_trace(go.Scatter(
                    x=base["percent_gait_cycle"], y=base["mean"],
                    mode="lines", name=f"{neutral_name} baseline",
                    line=dict(dash="dash", color="#000000", width=2),
                    showlegend=(i==1 and j==1),
                    hovertemplate="%{x:.0f}% gait<br>angle=%{y:.3f}°<extra>neutral baseline</extra>"
                ), row=i, col=j)

            if not sub.empty:
                style = CONDITION_STYLE.get(c, {"line":"#555", "fill":"rgba(85,85,85,0.18)"})
                add_band_and_mean(
                    fig, i, j,
                    sub["percent_gait_cycle"], sub["mean"], sub["std"],
                    name=c, line_color=style["line"], fill_color=style["fill"],
                    showlegend=False
                )
            fig.update_yaxes(range=[ylo, yhi], row=i, col=j)

    for j in range(1, cols+1):
        fig.update_xaxes(title_text="<b>Gait cycle (%)</b>", row=rows, col=j, title_font = dict(size=20))
    for i in range(1, rows+1):
        fig.update_yaxes(title_text="<b>Angle (°)</b>", row=i, col=1, title_font = dict(size=20))

    fig.update_layout(title="Ankle plantarflexion/dorsiflexion angle: Systems × Conditions: condition vs neutral (mean ± SD)",
                      template="plotly_white", height=320*rows, width=360*cols, font = dict(size=20))
    fig.update_annotations(font_size=18)
    fig.write_html(out_path)
    return out_path


def save_highres_png(fig, path, width_in=8, height_in=4.5, dpi=300):
    """
    Save figure as high-res PNG.
    width_in, height_in: desired figure size in inches
    dpi: dots per inch
    """
    pio.write_image(
        fig,
        path,
        format="png",
        width=int(width_in * dpi),
        height=int(height_in * dpi),
        scale=1,
        engine="kaleido",
    )

def build_all_figures(conditions: dict[str, str | Path], tracker: str = "mediapipe_dlc",
                      out_dir: str | Path = "ankle_summary_plots", neutral_name: str = "neutral") -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_condition_csvs({k: Path(v) for k,v in conditions.items()}, tracker=tracker)
    summ = summarize(df)
    outs = []
    outs.append(family1_combined_row(summ, out_dir / "family1_combined.html"))
    outs.append(family2_combined_row(summ, out_dir / "family2_combined.html"))
    outs.append(family3_grid_with_neutral(summ, out_dir / f"family3_combined.html", neutral_name=neutral_name))

    return outs

if __name__ == "__main__":
    # Example usage (edit these paths for your machine):
    conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }
    
    # conditions = {
    #     "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1",
    #     "neg_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1",
    #     "pos_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1",

    # }
    
    outs = build_all_figures(conditions, tracker="mediapipe_dlc", out_dir="ankle_summary_plots", neutral_name="neutral")
    for p in outs:
        print(p)


