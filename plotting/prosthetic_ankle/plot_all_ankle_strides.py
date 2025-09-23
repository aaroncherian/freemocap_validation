
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
    "neg_5_6": {"line": "#2ca02c", "fill": "rgba(44,160,44,0.22)"},   # green
    "neg_2_8": {"line": "#ff7f0e", "fill": "rgba(255,127,14,0.22)"},  # orange
    "neutral": {"line": "#111111", "fill": "rgba(17,17,17,0.18)"},    # near-black
    "pos_2_8": {"line": "#9467bd", "fill": "rgba(148,103,189,0.22)"}, # purple
    "pos_5_6": {"line": "#1f77b4", "fill": "rgba(31,119,180,0.22)"},   # blue
    "neg_6": {"line": "#2ca02c", "fill": "rgba(44,160,44,0.22)"},   # green
    "pos_6" : {"line": "#1f77b4", "fill": "rgba(31,119,180,0.22)"}   # blue
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

def family3_all_strides_grid_with_neutral(df: pd.DataFrame, neutral_name: str = "neutral") -> None:
    """
    Grid: rows = systems, cols = conditions.
    Each panel shows every stride (thin, translucent lines) for that system/condition,
    plus a dashed black neutral baseline for the same system.
    """
    systems = sorted(df["system"].unique())
    # preserve your preferred order then append extras
    conditions = [c for c in CONDITION_ORDER if c in set(df["condition"].unique())]
    conditions += [c for c in sorted(df["condition"].unique()) if c not in conditions]
    if neutral_name not in conditions:
        raise ValueError(f"Neutral baseline '{neutral_name}' not in: {conditions}")

    rows, cols = len(systems), len(conditions)
    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"<b>{s} · {c}</b>" for s in systems for c in conditions],
        horizontal_spacing=0.04, vertical_spacing=0.07
    )

    # y-range from all raw strides
    ymin = df["ankle_angle"].min()
    ymax = df["ankle_angle"].max()
    pad  = 0.05 * (ymax - ymin + 1e-9)
    ylo, yhi = float(ymin - pad), float(ymax + pad)


    # draw panels
    for i, s in enumerate(systems, start=1):
        for j, c in enumerate(conditions, start=1):
            panel = (
                df[(df["system"]==s) & (df["condition"]==c)]
                .sort_values(["stride","percent_gait_cycle"])
            )
            # one trace per stride
            for stride_id, g in panel.groupby("stride", sort=True):
                style = CONDITION_STYLE.get(c, {"line":"#555"})
                rgba = _hex_to_rgba(style["line"], 0.42)  # translucent per-stride line
                fig.add_trace(
                    go.Scatter(
                        x=g["percent_gait_cycle"], y=g["ankle_angle"],
                        mode="lines",
                        line=dict(width=1.3, color=rgba),
                        name=stride_id, # hidden from legend
                        hovertemplate="%{x:.0f}% gait<br>%{y:.3f}°<extra>stride "+str(stride_id)+"</extra>",
                        showlegend=False
                    ),
                    row=i, col=j
                )
            fig.update_yaxes(range=[ylo, yhi], row=i, col=j)

    # axes & layout
    for j in range(1, cols+1):
        fig.update_xaxes(title_text="<b>Gait cycle (%)</b>", row=rows, col=j, title_font=dict(size=20))
    for i in range(1, rows+1):
        fig.update_yaxes(title_text="<b>Angle (°)</b>", row=i, col=1, title_font=dict(size=20))

    fig.update_layout(
        title="<b>Ankle plantarflexion/dorsiflexion — all strides per panel (neutral baseline dashed)</b>",
        template="plotly_white",
        height=max(320*rows, 400),
        width=max(360*cols, 900),
        font=dict(size=20),
    )
    fig.update_annotations(font_size=18)

    fig.show()


# ----- example usage -----
if __name__ == "__main__":
    conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }
    df = load_condition_csvs({k: Path(v) for k, v in conditions.items()}, tracker="mediapipe_dlc")
    
    family3_all_strides_grid_with_neutral(df, neutral_name="neutral")
