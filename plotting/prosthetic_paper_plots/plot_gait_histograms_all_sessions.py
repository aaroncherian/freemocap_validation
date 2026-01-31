from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

QUALISYS_REL = Path("validation/qualisys/gait_parameters/qualisys_gait_metrics.csv")
FMC_REL      = Path("validation/mediapipe_dlc/gait_parameters/gait_metrics.csv")

def load_gait_dataframe(
        folder_list: list,
        qualisys_rel: Path = QUALISYS_REL,
        fmc_rel: Path = FMC_REL
) -> pd.DataFrame:

    rows = [] 

    for folder in folder_list:
        for fpath in [folder/qualisys_rel, folder/fmc_rel]:
            if not fpath.exists():
                raise FileNotFoundError(f"Missing CSV: {fpath}")
            
            tmp = pd.read_csv(fpath)

            required = {"system", "side", "metric", "event_index", "value"}
            missing = required - set(tmp.columns)
            if missing:
                raise ValueError(f"{fpath} missing columns: {missing}")
            
            tmp['trial_name'] = folder.stem
            rows.append(tmp)

    df  = pd.concat(rows, ignore_index=True)
    df["event_index"] = df["event_index"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def get_paired_stride_df(
        df: pd.DataFrame,
        metric: str,
        side: str = "right",
        reference_system = "qualisys",
        fmc_system: str|None = None 
) -> pd.DataFrame: 
    
    d = df[(df["metric"] == metric) & (df["side"] == side)].copy()
    
    systems = sorted(d["system"].dropna().unique().tolist())
    if reference_system not in systems:
        raise ValueError(f"Reference system '{reference_system}' not found in data systems: {systems}")
    
    if fmc_system is None:
        others = [s for s in systems if s != reference_system]
        if len(others) != 1:
            raise ValueError(f"Could not auto-detect fmc_system; found systems: {systems}")
        fmc_system = others[0]
    
    qual = d[d["system"]== reference_system].rename(columns = {"value": reference_system})[
        ["trial_name", "side", "metric", "event_index", reference_system]
    ]

    fmc = d[d["system"] == fmc_system].rename(columns = {"value": "freemocap"})[
        ["trial_name", "side", "metric", "event_index", "freemocap"]
    ]
    
    paired = qual.merge(
        fmc,
        on = ["trial_name", "side", "metric", "event_index"], how = "inner")
    
    paired["diff"]  = paired["freemocap"]  - paired[reference_system]
    return paired 

def plot_histogram(
    df: pd.DataFrame,
    plot_title: str,
    ax = None,
    max_frames: int = 7,
    show_ylabel: bool = True,
) -> plt.Axes:
    
    x = df["diff"].to_numpy()
    x = x[np.isfinite(x)]

    fps = 30
    frame_dt = 1/fps

    max_abs = max_frames * frame_dt

    ax.set_xlim(-max_abs, max_abs)
    
    # Ticks at 50ms intervals (scientific convention)
    max_ms = max_abs * 1000
    major_ticks_ms = np.arange(-150, 151, 50)  # -150, -100, -50, 0, 50, 100, 150
    major_ticks_ms = major_ticks_ms[np.abs(major_ticks_ms) <= max_ms]
    major_ticks = major_ticks_ms / 1000  # convert to seconds

    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1000:+.0f}" if v != 0 else "0"))
    ax.tick_params(axis="x", labelrotation=0) 
    
    bin_width = frame_dt
    edges = np.arange(-max_abs - bin_width/2, max_abs + bin_width, bin_width)

    ax.hist(x, bins = edges, edgecolor = None, color = "#1f77b4", alpha = .75)
    ax.hist(x, bins = edges, histtype = "step", linewidth = 1.0, color = "#0a3f64")

    ax.axvline(0, color="#0a3f64", linestyle="--", linewidth=1.2, alpha=0.4)
    ax.set_title(plot_title, fontweight="bold")
    
    if show_ylabel:
        ax.set_ylabel("Count", fontweight="bold")

    bias = np.mean(x) * 1000
    sd = np.std(x, ddof=1) * 1000
    ax.text(
        0.03,
        0.95,
        f"$\\mu$ = {bias:+.0f} $\\pm$ {sd:.0f} ms",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    ax.grid(axis = "y", alpha = 0.2)
    ax.set_axisbelow(True)
    return ax


path_to_recordings = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera')

list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]

list_of_valid_folders = []
for p in list_of_folders:
    if (p/'validation').is_dir():
        list_of_valid_folders.append(p)
    else:
        print(f"Skipping {p}")

gait_param_df = load_gait_dataframe(list_of_valid_folders)

paired_stance = get_paired_stride_df(
    df = gait_param_df,
    metric = "stance_duration",
    side = "right"
)

paired_swing = get_paired_stride_df(
    df = gait_param_df,
    metric = "swing_duration",
    side = "right"
)

fig_hist, axes_hist = plt.subplots(
    nrows = 1,
    ncols = 2,
    figsize = (10,5),
    sharey=True
)

plot_histogram(paired_stance, "Stance Phase", ax=axes_hist[0], show_ylabel=True)
plot_histogram(paired_swing, "Swing Phase", ax=axes_hist[1], show_ylabel=False)

# Single centered x-axis label
fig_hist.text(0.5, 0.02, "Error (ms)", ha="center", fontweight="bold")

plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave room for xlabel
plt.show()