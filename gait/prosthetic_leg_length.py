from pathlib import Path
from skellymodels.managers.human import Human
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt



recordings = {
    "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
    "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
    "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
    "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
}

@dataclass
class LegResults:
    data: np.ndarray
    mean: float
    std:float

def leg_length_from_human(human:Human):
    leg_lengths = np.linalg.norm(human.body.xyz.as_dict['right_knee'] - human.body.xyz.as_dict['right_ankle'] , axis = 1)
    leg_length_mean = np.median(leg_lengths)
    leg_length_std =  np.median(np.abs(leg_lengths - leg_length_mean))
    
    return LegResults(
        data=leg_lengths,
        mean=leg_length_mean,
        std=leg_length_std
    )

freemocap_results: dict[str, LegResults] = {}
qualisys_results: dict[str, LegResults] = {}


for name,recording_path in recordings.items():

    path_to_freemocap_folder = recording_path/'validation'/'mediapipe_dlc'
    path_to_qualisys_folder = recording_path/'validation'/'qualisys'

    f_human:Human = Human.from_data(path_to_freemocap_folder)
    q_human:Human = Human.from_data(path_to_qualisys_folder)

    q_leg_length = leg_length_from_human(q_human)
    f_leg_length = leg_length_from_human(f_human)

    freemocap_results[name] = leg_length_from_human(f_human)
    qualisys_results[name] = leg_length_from_human(q_human)


fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle('FreeMoCap Leg Length Distributions by Condition')

for ax, (name, result) in zip(axes, freemocap_results.items()):
    ax.hist(result.data, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(result.data), color='red', linestyle='--', label='mean')
    ax.axvline(np.median(result.data), color='blue', linestyle='-', label='median')
    ax.set_title(name)
    ax.set_xlabel('Leg length (mm)')

axes[0].set_ylabel('Frame count')
axes[-1].legend()
plt.tight_layout()
plt.show()

INCH_TO_MM = 25.4

# Define the inch offsets
inch_offsets = {
    "neg_5": -0.5,
    "neg_25": -0.25,
    "neutral": 0.0,
    "pos_25": 0.25,
    "pos_5": 0.5,
}

condition_order = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]

# Baseline (neutral mean)
fmc_neutral = freemocap_results["neutral"].mean
q_neutral = qualisys_results["neutral"].mean

# Arrays to fill
fmc_deltas = []
q_deltas = []
expected_deltas = []
fmc_stds = []
q_stds = []

for cond in condition_order:
    f_res = freemocap_results[cond]
    q_res = qualisys_results[cond]

    # ∆ from neutral (mm)
    fmc_deltas.append(f_res.mean - fmc_neutral)
    q_deltas.append(q_res.mean - q_neutral)

    fmc_stds.append(f_res.std)
    q_stds.append(q_res.std)

    # Expected mechanical change (mm)
    expected_deltas.append(inch_offsets[cond] * INCH_TO_MM)

fmc_deltas = np.array(fmc_deltas)
q_deltas = np.array(q_deltas)
expected_deltas = np.array(expected_deltas)
fmc_stds = np.array(fmc_stds)
q_stds = np.array(q_stds)

x = np.arange(len(condition_order))

plt.figure(figsize=(8, 5))

# FreeMoCap
plt.errorbar(
    x - 0.1,
    fmc_deltas,
    yerr=fmc_stds,
    fmt='o',
    capsize=5,
    label='FreeMoCap Δ',
)

# Qualisys
plt.errorbar(
    x + 0.1,
    q_deltas,
    yerr=q_stds,
    fmt='o',
    capsize=5,
    label='Qualisys Δ',
)

# Expected mechanical offsets (mm)
plt.plot(
    x,
    expected_deltas,
    'k--o',
    label='Expected Δ (mm)',
)

plt.axhline(0, color='gray', linewidth=1)

plt.xticks(
    x,
    ["-0.5 in", "-0.25 in", "neutral", "+0.25 in", "+0.5 in"],
)
plt.ylabel("Δ Leg length from neutral [mm]")
plt.xlabel("Prosthetic alignment condition")
plt.title("Change in leg length relative to neutral")
plt.legend()
plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare X for sklearn
X = expected_deltas.reshape(-1, 1)

# Fit regression lines
fmc_model = LinearRegression().fit(X, fmc_deltas)
q_model  = LinearRegression().fit(X, q_deltas)

# Predict full line for plotting
x_line = np.linspace(expected_deltas.min(), expected_deltas.max(), 100).reshape(-1, 1)
fmc_line = fmc_model.predict(x_line)
q_line  = q_model.predict(x_line)

plt.figure(figsize=(8,6))

# Identity line (perfect agreement)
plt.plot(
    x_line, x_line,
    'k--', linewidth=1.2, label='Identity (1:1)'
)

# Regression lines
plt.plot(x_line, fmc_line, color='tab:orange', linewidth=2, label=f"FreeMoCap fit (slope={fmc_model.coef_[0]:.2f})")
plt.plot(x_line, q_line,  color='tab:blue',   linewidth=2, label=f"Qualisys fit (slope={q_model.coef_[0]:.2f})")

# Scatter points
plt.scatter(expected_deltas, fmc_deltas, color='tab:orange', s=80, marker='o', label='FreeMoCap Δ')
plt.scatter(expected_deltas, q_deltas,  color='tab:blue',   s=80, marker='o', label='Qualisys Δ')

plt.xlabel("Expected Δ leg length (mm)")
plt.ylabel("Measured Δ leg length (mm)")
plt.title("Regression of Measured vs Expected Leg-Length Change")
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)

plt.legend()
plt.tight_layout()
plt.show()


X = expected_deltas.reshape(-1, 1)

fmc_model = LinearRegression().fit(X, fmc_deltas)
q_model  = LinearRegression().fit(X, q_deltas)

x_line = np.linspace(expected_deltas.min(), expected_deltas.max(), 100).reshape(-1, 1)

fmc_line = fmc_model.predict(x_line)
q_line  = q_model.predict(x_line)

# ---------- Residuals ----------
fmc_residuals = fmc_deltas - expected_deltas
q_residuals   = q_deltas  - expected_deltas

# ---------- PLOTTING ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================================
# PANEL A — Regression (Measured vs Expected)
# =========================================
ax = axes[0]

# identity line
ax.plot(x_line, x_line, 'k--', linewidth=1.2, label='Identity (1:1)')

# regression lines
ax.plot(x_line, fmc_line, color='tab:orange', linewidth=2,
        label=f"FreeMoCap fit (slope={fmc_model.coef_[0]:.2f})")
ax.plot(x_line, q_line, color='tab:blue', linewidth=2,
        label=f"Qualisys fit (slope={q_model.coef_[0]:.2f})")

# scatter points
ax.scatter(expected_deltas, fmc_deltas, color='tab:orange', s=80, label='FreeMoCap Δ')
ax.scatter(expected_deltas, q_deltas,  color='tab:blue',   s=80, label='Qualisys Δ')

ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(0, color='gray', linewidth=0.8)

ax.set_xlabel("Expected Δ leg length (mm)")
ax.set_ylabel("Measured Δ leg length (mm)")
ax.set_title("A. Regression: Measured vs Expected Δ")
ax.legend()

# =========================================
# PANEL B — Residuals (Measured - Expected)
# =========================================
ax = axes[1]

ax.axhline(0, color='gray', linewidth=1)  # zero-error line

ax.scatter(expected_deltas, fmc_residuals,
           color='tab:orange', s=80, label='FreeMoCap residual')
ax.scatter(expected_deltas, q_residuals,
           color='tab:blue',   s=80, label='Qualisys residual')

ax.set_xlabel("Expected Δ leg length (mm)")
ax.set_ylabel("Residual (Measured − Expected) (mm)")
ax.set_title("B. Residuals vs Expected Δ")
ax.legend()

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Differences
fmc_diff = fmc_deltas - expected_deltas
q_diff   = q_deltas  - expected_deltas

# Means
fmc_mean = (fmc_deltas + expected_deltas) / 2
q_mean   = (q_deltas  + expected_deltas) / 2
fmc_std = np.std(fmc_diff, ddof=1)
q_std   = np.std(q_diff, ddof=1)
# Bias
fmc_bias = np.mean(fmc_diff)
q_bias   = np.mean(q_diff)

# Limits of agreement
fmc_bias = np.mean(fmc_diff)
fmc_upper = fmc_bias + 1.96 * fmc_std
fmc_lower = fmc_bias - 1.96 * fmc_std

# Qualisys
q_bias = np.mean(q_diff)
q_upper = q_bias + 1.96 * q_std
q_lower = q_bias - 1.96 * q_std

plt.figure(figsize=(7,6))

# Scatter
plt.scatter(fmc_mean, fmc_diff, color='tab:orange', s=80, label="FreeMoCap")
plt.scatter(q_mean,   q_diff,   color='tab:blue',   s=80, label="Qualisys")

# FMC lines
ax.axhline(fmc_bias,  color='tab:orange', linestyle='--')
ax.axhline(fmc_upper, color='tab:orange', linestyle=':')
ax.axhline(fmc_lower, color='tab:orange', linestyle=':')

# Qualisys lines
ax.axhline(q_bias,    color='tab:blue',   linestyle='--')
ax.axhline(q_upper,   color='tab:blue',   linestyle=':')
ax.axhline(q_lower,   color='tab:blue',   linestyle=':')


plt.xlabel("Mean of Expected and Measured Δ (mm)")
plt.ylabel("Difference (Measured − Expected) (mm)")
plt.title("Bland–Altman Plot for Leg-Length Δ")
plt.axhline(0, color='gray', linewidth=1)

plt.legend()
plt.tight_layout()
plt.show()