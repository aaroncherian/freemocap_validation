from __future__ import annotations

import copy
import csv
import numpy as np
from pathlib import Path

from validation.pipeline.builder import build_pipeline
from validation.pipeline.base import ValidationPipeline

from validation.steps.temporal_alignment.step import TemporalAlignmentStep
from validation.steps.spatial_alignment.step import SpatialAlignmentStep
from validation.steps.rmse.step import RMSEStep


def extract_total_rmse_mm(rmse_dir: Path) -> float:
    """
    Adjust filename if needed — this assumes RMSEStep writes a CSV
    containing a column like 'rmse_mm' or similar.
    """
    csvs = list(rmse_dir.glob("*.csv"))
    if not csvs:
        return float("nan")

    import pandas as pd
    df = pd.read_csv(csvs[0])

    # pick whatever scalar you trust most
    if "rmse_mm" in df.columns:
        return float(df["rmse_mm"].mean())

    return float("nan")


def sweep_lags(
    config_path: Path,
    *,
    lag_grid: np.ndarray,
    tracker: str | None = None,
) -> Path:

    base_ctx, _ = build_pipeline(config_path)

    if tracker is not None:
        base_ctx.project_config.freemocap_tracker = tracker

    sweeps_root = (
        base_ctx.recording_dir
        / "validation"
        / "_sweeps"
        / base_ctx.project_config.freemocap_tracker
    )
    sweeps_root.mkdir(parents=True, exist_ok=True)

    report_csv = sweeps_root / "lag_sweep.csv"

    with open(report_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lag_frames", "rmse_total_mm", "sweep_dir"],
        )
        writer.writeheader()

        for lag in lag_grid:
            lag = float(lag)
            sweep_dir = sweeps_root / f"lag_{lag:.3f}"
            sweep_dir.mkdir(parents=True, exist_ok=True)

            ctx = copy.deepcopy(base_ctx)
            ctx.output_root = sweep_dir

            # force lag + kill GUI
            cfg_key = "TemporalAlignmentStep.config"
            step_cfg = dict(ctx.get(cfg_key) or {})          # start with whatever came from YAML
            step_cfg["manual_lag_frames"] = lag
            step_cfg["interactive"] = False
            ctx.put(cfg_key, step_cfg)

            pipeline = ValidationPipeline(
                context=ctx,
                steps=[
                    TemporalAlignmentStep,
                    SpatialAlignmentStep,
                    RMSEStep,
                ],
            )

            pipeline.run()

            rmse_dir = sweep_dir / "validation" / "rmse"
            rmse_val = extract_total_rmse_mm(rmse_dir)

            writer.writerow(
                {
                    "lag_frames": f"{lag:.3f}",
                    "rmse_total_mm": rmse_val,
                    "sweep_dir": str(sweep_dir),
                }
            )

            print(f"lag={lag:.3f} → RMSE={rmse_val:.2f} mm")

    return report_csv


if __name__ == "__main__":
    cfg = Path(r"C:\Users\aaron\Documents\GitHub\freemocap_validation\config_yamls\validation\jsm\jsm_treadmill_1.yaml")

    # Coarse first — do NOT start with 0.01 steps
    lags = np.round(np.arange(2.0, 3.5 + 1e-9, 0.05), 3)

    out = sweep_lags(cfg, lag_grid=lags, tracker="mediapipe")
    print("Wrote sweep report:", out)
