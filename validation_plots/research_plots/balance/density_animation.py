import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter


import pandas as pd
import sqlite3
from pathlib import Path

from scipy.ndimage import binary_dilation

root_path = Path(r"D:\validation\balance")
root_path.mkdir(exist_ok=True, parents=True)

USE_LOG = True
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
    path = row["path"]
    tracker = row["tracker"]
    condition = row.get("condition") or ""
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_csv(path)

    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["condition"] = condition
    sub_df["tracker"] = tracker
    dfs.append(sub_df)

final_df = pd.concat(dfs, ignore_index=True)
# ── Filter to trial & tracker ──
trial = "2025-07-31_16-00-42_GMT-4_jsm_nih_trial_1"
tracker = "mediapipe"

df = final_df[
    (final_df["trial_name"] == trial) &
    (final_df["tracker"] == tracker)
].copy().sort_values("Frame").reset_index(drop=True)

conditions = {
    "Eyes Open / Solid Ground": {
        "x": "Eyes Open/Solid Ground_x",
        "y": "Eyes Open/Solid Ground_y",
    },
    "Eyes Closed / Foam": {
        "x": "Eyes Closed/Foam_x",
        "y": "Eyes Closed/Foam_y",
    },
}


# ── Preprocess ──
smooth_window = 15
smooth_order = 3

# Density grid params
n_bins = 80               # resolution of the XY grid
sigma = 1.2               # gaussian blur in bins
cmap_name = "viridis"
cmap = cm.get_cmap(cmap_name)

# Shared view
elev, azim = 30, -60

processed = {}
all_ml = []
all_ap = []

for label, cols in conditions.items():
    ml = df[cols["x"]].values.astype(float)
    ap = df[cols["y"]].values.astype(float)

    mask = ~(np.isnan(ml) | np.isnan(ap))
    ml, ap = ml[mask], ap[mask]

    # Center
    ml = ml - np.nanmean(ml)
    ap = ap - np.nanmean(ap)

    # Smooth (optional)
    ml = savgol_filter(ml, smooth_window, smooth_order)
    ap = savgol_filter(ap, smooth_window, smooth_order)

    # Velocity (optional; for weighting)
    dml = np.gradient(ml)
    dap = np.gradient(ap)
    vel = np.sqrt(dml**2 + dap**2)
    vel = np.convolve(vel, np.ones(5) / 5, mode="same")

    processed[label] = {"ml": ml, "ap": ap, "vel": vel}
    all_ml.append(ml)
    all_ap.append(ap)

all_ml = np.concatenate(all_ml)
all_ap = np.concatenate(all_ap)

# Shared XY limits
pad = 3
ml_lim = (np.min(all_ml) - pad, np.max(all_ml) + pad)
ap_lim = (np.min(all_ap) - pad, np.max(all_ap) + pad)

# Shared XY bin edges and grid centers
x_edges = np.linspace(ml_lim[0], ml_lim[1], n_bins + 1)
y_edges = np.linspace(ap_lim[0], ap_lim[1], n_bins + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")  # shape (n_bins, n_bins)

n_frames_data = min(len(p["ml"]) for p in processed.values())

# ── Animation timing ──
step = 2
n_anim_frames = max(1, n_frames_data // step)

# Choose weighting:
#   "counts" = pure visit count
#   "linger" = more height when moving slowly (1/(vel+eps))
weight_mode = "counts"

def weights_for(data, sl):
    if weight_mode == "counts":
        return None  # histogram2d defaults to counts
    elif weight_mode == "linger":
        eps = 1e-6
        w = 1.0 / (data["vel"][sl] + eps)
        # keep weights sane (avoid huge spikes)
        w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
        return w
    else:
        return None

# ── Figure ──
fig = plt.figure(figsize=(16, 7), facecolor="white")
fig.suptitle(
    f"COM Density Growth Surface — {trial} ({tracker})",
    fontsize=14, fontweight="bold", y=0.97
)

axes = []
surfaces = {}

for idx, label in enumerate(conditions.keys()):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    ax.set_xlim(*ml_lim)
    ax.set_ylim(*ap_lim)
    ax.set_xlabel("ML (mm)", fontsize=9, labelpad=8)
    ax.set_ylabel("AP (mm)", fontsize=9, labelpad=8)
    ax.set_zlabel("Density", fontsize=9, labelpad=8)
    ax.set_title(label, fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)
    axes.append(ax)
    surfaces[label] = None
    ax.set_facecolor("#f4f4f4")
    ax.xaxis.pane.set_facecolor((1,1,1,1))
    ax.yaxis.pane.set_facecolor((1,1,1,1))
    ax.zaxis.pane.set_facecolor((1,1,1,0))

# We’ll set z limits after a quick “max possible” estimate
# Estimate max density by using full data hist and blur
z_max_est = 1.0
for label, data in processed.items():
    H_raw, _, _ = np.histogram2d(
        data["ml"][:n_frames_data],
        data["ap"][:n_frames_data],
        bins=[x_edges, y_edges],
        weights=weights_for(data, slice(0, n_frames_data))
    )
    H_raw = H_raw.T
    Z = gaussian_filter(H_raw, sigma=sigma)
    if USE_LOG:
        Z = np.log1p(Z)
    support = H_raw > 0
    support = binary_dilation(support, iterations=2)

    vals = Z[support]
    if vals.size:
        z_max_est = max(z_max_est, np.percentile(vals, 99.5))

for ax in axes:
    ax.set_zlim(0, z_max_est * 1.1)

# Color normalization based on expected max
norm = Normalize(vmin=0, vmax=z_max_est * 0.9)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.08, aspect=25)
cbar.set_label("Accumulated density", fontsize=10)

plt.tight_layout()

def animate(frame_idx):
    end = min((frame_idx + 1) * step, n_frames_data)

    for ax_idx, (label, data) in enumerate(processed.items()):
        ax = axes[ax_idx]

        # Clear previous surface (Matplotlib doesn't update surfaces elegantly)
        if surfaces[label] is not None:
            surfaces[label].remove()
            surfaces[label] = None

        sl = slice(0, end)
        w = weights_for(data, sl)

        start = frame_idx * step
        end = min((frame_idx + 1) * step, data["n_valid"])

        if end > start:
            np.add.at(accum[label], (data["yi"][start:end], data["xi"][start:end]), 1.0)

        H_raw = accum[label]
        Z = gaussian_filter(H_raw, sigma=sigma)
        if USE_LOG:
            Z = np.log1p(Z)

        support = H_raw > 0
        support = binary_dilation(support, iterations=2)
        Z_plot = Z.copy()
        Z_plot[~support] = np.nan

        # Transpose to match Xc/Yc orientation
        H_raw = H_raw.T

        # Smooth for terrain look
        Z = gaussian_filter(H_raw, sigma=sigma)
        if USE_LOG:
            Z = np.log1p(Z)

        # If you're using log scaling:
        # Z = np.log1p(Z)

        # --- Mask: only show bins that were actually visited ---
        support = H_raw > 0

        # Optional: expand the visible region a bit so it doesn't look pixel-y
        support = binary_dilation(support, iterations=2)

        Z_plot = Z.copy()
        Z_plot[~support] = np.nan   # <- this prevents the purple floor

        surf = ax.plot_surface(
            Xc, Yc, Z_plot,
            cmap=cmap,
            norm=norm,
            linewidth=0,
            antialiased=True
        )
        
        surfaces[label] = surf

        # Optional: keep viewpoint constant
        ax.view_init(elev=elev, azim=azim)

    return []

for label, data in processed.items():
    # clip into bin range and convert to bin indices
    xi = np.searchsorted(x_edges, data["ml"], side="right") - 1
    yi = np.searchsorted(y_edges, data["ap"], side="right") - 1
    good = (xi >= 0) & (xi < n_bins) & (yi >= 0) & (yi < n_bins)

    data["xi"] = xi[good]
    data["yi"] = yi[good]
    data["n_valid"] = len(data["xi"])

accum = {label: np.zeros((n_bins, n_bins), dtype=np.float32) for label in processed}

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=n_anim_frames,
    interval=16,
    blit=False,
    repeat=False,
)

anim.save("sway_density_surface.mp4", writer="ffmpeg", fps=60, dpi=150)
plt.show()
