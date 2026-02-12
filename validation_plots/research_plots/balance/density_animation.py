import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive (faster, good for rendering video)
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter, binary_dilation

import pandas as pd
import sqlite3
from pathlib import Path


# -------------------
# Settings
# -------------------
USE_LOG = True
N_BINS = 80
SIGMA = 1.2
DILATE_ITERS = 2
FPS = 60
DPI = 150

# How many *data samples* to add per animation frame (larger = faster video generation)
STEP = 1

# If you want smoothing of the raw ML/AP traces before binning:
SMOOTH_WINDOW = 15
SMOOTH_ORDER = 3

CMAP = cm.get_cmap("viridis")

root_path = Path(r"D:\validation\balance")
root_path.mkdir(exist_ok=True, parents=True)

# -------------------
# Load paths from DB
# -------------------
conn = sqlite3.connect("validation.db")
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
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE '%balance_positions'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub_df = pd.read_csv(row["path"])
    sub_df["participant_code"] = row["participant_code"]
    sub_df["trial_name"] = row["trial_name"]
    sub_df["condition"] = row.get("condition") or ""
    sub_df["tracker"] = row["tracker"]
    dfs.append(sub_df)

final_df = pd.concat(dfs, ignore_index=True)

# -------------------
# Filter to trial & tracker
# -------------------
trial = "2025-11-04_15-02-28_GMT-5_atc_nih_1"
tracker = "mediapipe"

df = final_df[
    (final_df["trial_name"] == trial) &
    (final_df["tracker"] == tracker)
].copy().sort_values("Frame").reset_index(drop=True)


# -------------------
# 4 conditions (EDIT THESE COLUMN NAMES IF NEEDED)
# -------------------
conditions = {
    "Eyes Open / Solid":       {"x": "Eyes Open/Solid Ground_x",      "y": "Eyes Open/Solid Ground_y"},
    "Eyes Closed / Solid":     {"x": "Eyes Closed/Solid Ground_x",    "y": "Eyes Closed/Solid Ground_y"},
    "Eyes Open / Foam":        {"x": "Eyes Open/Foam_x",              "y": "Eyes Open/Foam_y"},
    "Eyes Closed / Foam":      {"x": "Eyes Closed/Foam_x",            "y": "Eyes Closed/Foam_y"},
}

# -------------------
# Helpers
# -------------------
def safe_savgol(x, window=15, poly=3):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return x
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return x
    p = min(poly, w - 1)
    return savgol_filter(x, w, p, mode="interp")

def preprocess_xy(x, y):
    x = x.astype(float)
    y = y.astype(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size == 0:
        return x, y
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    x = safe_savgol(x, SMOOTH_WINDOW, SMOOTH_ORDER)
    y = safe_savgol(y, SMOOTH_WINDOW, SMOOTH_ORDER)
    return x, y


# -------------------
# Build processed dict + shared limits
# -------------------
processed = {}
all_ml, all_ap = [], []

for label, cols in conditions.items():
    ml, ap = preprocess_xy(df[cols["x"]].values, df[cols["y"]].values)
    processed[label] = {"ml": ml, "ap": ap}
    if ml.size:
        all_ml.append(ml)
        all_ap.append(ap)

if not all_ml:
    raise RuntimeError("No valid ML/AP data found for any condition.")

all_ml = np.concatenate(all_ml)
all_ap = np.concatenate(all_ap)

pad = 3
ml_lim = (np.min(all_ml) - pad, np.max(all_ml) + pad)
ap_lim = (np.min(all_ap) - pad, np.max(all_ap) + pad)

x_edges = np.linspace(ml_lim[0], ml_lim[1], N_BINS + 1)
y_edges = np.linspace(ap_lim[0], ap_lim[1], N_BINS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")

# Pre-bin indices once per condition
for label, data in processed.items():
    xi = np.searchsorted(x_edges, data["ml"], side="right") - 1
    yi = np.searchsorted(y_edges, data["ap"], side="right") - 1
    good = (xi >= 0) & (xi < N_BINS) & (yi >= 0) & (yi < N_BINS)

    data["xi"] = xi[good].astype(np.int32)
    data["yi"] = yi[good].astype(np.int32)
    data["n_valid"] = int(good.sum())

# Use the minimum length across conditions so animations stay aligned
n_frames_data = min(d["n_valid"] for d in processed.values())
for label in processed:
    processed[label]["xi"] = processed[label]["xi"][:n_frames_data]
    processed[label]["yi"] = processed[label]["yi"][:n_frames_data]
    processed[label]["n_valid"] = n_frames_data

for label in processed:
    processed[label]["ml"] = processed[label]["ml"][:n_frames_data]
    processed[label]["ap"] = processed[label]["ap"][:n_frames_data]

# Raw cumulative path length (mm) per condition
for label, data in processed.items():
    # Use the *smoothed continuous* ML/AP (more accurate than bin centers)
    ml = data["ml"][:n_frames_data]
    ap = data["ap"][:n_frames_data]

    dml = np.diff(ml)
    dap = np.diff(ap)
    step_dist = np.sqrt(dml**2 + dap**2)   # mm per step

    data["path_cum_mm"] = np.concatenate([[0.0], np.cumsum(step_dist)])

n_anim_frames = max(1, int(np.ceil(n_frames_data / STEP)))

# Accumulators (one per condition)
accum = {label: np.zeros((N_BINS, N_BINS), dtype=np.float32) for label in processed}

# -------------------
# Estimate global z max (consistent scaling)
# -------------------
z_max_est = 1.0
for label, data in processed.items():
    H = np.zeros((N_BINS, N_BINS), dtype=np.float32)
    np.add.at(H, (data["yi"], data["xi"]), 1.0)
    Z = gaussian_filter(H, sigma=SIGMA)
    if USE_LOG:
        Z = np.log1p(Z)

    support = binary_dilation(H > 0, iterations=DILATE_ITERS)
    vals = Z[support]
    if vals.size:
        z_max_est = max(z_max_est, np.percentile(vals, 99.5))

norm = Normalize(vmin=0, vmax=z_max_est * 0.9)

#Perfect — that will be faster too (half the axes, half the drawing).

# -------------------
# Figure layout: 1 row × 4 columns (surfaces only)
# -------------------
fig = plt.figure(figsize=(22, 6), facecolor="white")
fig.suptitle(f"COM Density Growth — {trial} ({tracker})", fontsize=16, fontweight="bold", y=0.98)

labels = list(processed.keys())
elev, azim = 35, -55

axes_surface = {}
surfaces = {}
tickers = {}
guides = {}

for i, label in enumerate(labels):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")

    ax.set_xlim(*ml_lim)
    ax.set_ylim(*ap_lim)
    ax.set_zlim(0, z_max_est * 1.1)

    ax.set_xlabel("ML (mm)", fontsize=9, labelpad=8)
    ax.set_ylabel("AP (mm)", fontsize=9, labelpad=8)
    ax.set_zlabel("Density", fontsize=9, labelpad=8)

    ax.set_title(label, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)

    # light panes / floor
    ax.set_facecolor("#f4f4f4")
    ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

    axes_surface[label] = ax
    surfaces[label] = None
    guides[label] = None
    tickers[label] = ax.text2D(
    0.02, 0.95,
    "Path: 0.0 mm",
    transform=ax.transAxes,
    fontsize=10,
    fontweight="bold"
    )


# Shared colorbar on far right
sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.93, 0.18, 0.012, 0.64])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Accumulated density", fontsize=11)

# Leave space for colorbar + title
fig.subplots_adjust(left=0.03, right=0.91, top=0.90, bottom=0.08, wspace=0.05)

# -------------------
# Animation (incremental accumulate)
# -------------------
def animate(frame_idx):
    start = frame_idx * STEP
    end = min((frame_idx + 1) * STEP, n_frames_data)
    path_idx = max(end - 1, 0)

    for label, data in processed.items():
        ax = axes_surface[label]

        if guides[label] is not None:
            guides[label].remove()
            guides[label] = None

        x0 = float(data["ml"][path_idx])
        y0 = float(data["ap"][path_idx])

        z0 = 0.0
        z1 = ax.get_zlim()[1]   # top of z axis

        guides[label], = ax.plot(
            [x0, x0],
            [y0, y0],
            [z0, z1],
            color="k",          # black/white guide
            linewidth=2.0,
            alpha=0.9
        )

        # Incremental accumulation
        if end > start:
            np.add.at(accum[label], (data["yi"][start:end], data["xi"][start:end]), 1.0)

        H = accum[label]
        Z = gaussian_filter(H, sigma=SIGMA)
        if USE_LOG:
            Z = np.log1p(Z)

        support = binary_dilation(H > 0, iterations=DILATE_ITERS)
        Z_plot = Z.copy()
        Z_plot[~support] = np.nan

        # Update surface
        if surfaces[label] is not None:
            surfaces[label].remove()
            surfaces[label] = None

        surfaces[label] = ax.plot_surface(
            Xc, Yc, Z_plot,
            cmap=CMAP,
            norm=norm,
            linewidth=0,
            antialiased=True
        )
        ax.view_init(elev=elev, azim=azim)

        # Update ticker (raw cumulative path length)
        path_mm = data["path_cum_mm"][path_idx]
        tickers[label].set_text(f"Path Length: {path_mm:,.1f} mm")

    return []

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=n_anim_frames,
    interval=1000 / FPS,
    blit=False,
    repeat=False
)

out_path = root_path / f"com_density_4cond_{trial}_{tracker}.mp4"
anim.save(str(out_path), writer="ffmpeg", fps=FPS, dpi=DPI)
print(f"Saved: {out_path}")
