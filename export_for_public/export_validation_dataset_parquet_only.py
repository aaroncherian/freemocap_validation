#!/usr/bin/env python
"""
Create a minimal de-identified public export of the FreeMoCap validation dataset.

The public release contains canonical aligned trajectory INPUT DATA ONLY:
    freemocap_data_by_frame.parquet

Derived gait/balance outputs, CSV convenience exports, and alignment transforms
are intentionally not included. They can be regenerated separately.

Private source layout
---------------------
Each trial is expected to contain:

    <trial>/validation/<system>/freemocap_data_by_frame.parquet

Public output layout
--------------------
    data/<participant_id>/<task-run>/<system>/aligned_3d_data/
        freemocap_data_by_frame.parquet

The metadata/ directory created alongside data/ is for export auditing. It
contains private provenance information and should not be included in the
public Zenodo release unless reviewed separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with `uv add pyyaml` or run with "
        "`uv run --with pyyaml python export_validation_dataset.py ...`"
    ) from exc


DATA_DIR = "data"
METADATA_DIR = "metadata"
ALIGNED_3D_DATA_DIR = "aligned_3d_data"
HUMAN_DATA_PARQUET = "freemocap_data_by_frame.parquet"

DEFAULT_SYSTEMS = (
    "qualisys",
    "mediapipe",
    "vitpose",
    "rtmpose",
)


@dataclass(frozen=True)
class Trial:
    participant_id: str
    trial_type: str
    trial_number: int
    trial_name: str
    data_root: Path

    @property
    def task(self) -> str:
        if self.trial_type.lower() in {"balance", "nih"}:
            return "balance"
        return self.trial_type.lower()

    @property
    def public_name(self) -> str:
        return f"task-{slug(self.task)}_trial-{self.trial_number:02d}"


@dataclass
class Config:
    output_root: Path
    registry_yamls: tuple[Path, ...]
    participant_ids: dict[str, str]
    systems: tuple[str, ...]


@dataclass
class ManifestRow:
    participant_id: str
    task: str
    trial: int
    system: str
    relative_path: str
    source_path: str
    size_bytes: int
    sha256: str


@dataclass
class WarningRow:
    level: str
    code: str
    source_path: str
    message: str


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def load_toml(path: Path) -> Config:
    with path.open("rb") as file:
        raw = tomllib.load(file)

    paths = raw["paths"]
    export = raw.get("export", {})
    identity = raw.get("identity", {})

    return Config(
        output_root=Path(paths["output_root"]).expanduser(),
        registry_yamls=tuple(Path(item).expanduser() for item in paths["registry_yamls"]),
        participant_ids={str(k): str(v) for k, v in identity.get("participant_ids", {}).items()},
        systems=tuple(export.get("systems", DEFAULT_SYSTEMS)),
    )


def load_trials(cfg: Config) -> list[Trial]:
    trials: list[Trial] = []

    for yaml_path in cfg.registry_yamls:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        private_code = str(raw["participant_code"])
        participant_id = cfg.participant_ids.get(private_code)

        if not participant_id:
            raise ValueError(f"No public participant ID configured for participant_code={private_code!r}")

        data_root = Path(str(raw["data_root"]))

        for item in raw.get("trials", []):
            trials.append(
                Trial(
                    participant_id=participant_id,
                    trial_type=str(item["trial_type"]).lower(),
                    trial_number=int(item["trial_number"]),
                    trial_name=str(item["trial_name"]),
                    data_root=data_root,
                )
            )

    return trials


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path, dry_run: bool) -> tuple[int, str]:
    if dry_run:
        return src.stat().st_size, sha256_file(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size, sha256_file(dst)


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def export(cfg: Config, overwrite: bool, dry_run: bool) -> tuple[list[ManifestRow], list[WarningRow]]:
    trials = load_trials(cfg)

    if cfg.output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {cfg.output_root}. Use --overwrite after reviewing the path.")
        if not dry_run:
            shutil.rmtree(cfg.output_root)

    if not dry_run:
        cfg.output_root.mkdir(parents=True)

    manifest: list[ManifestRow] = []
    warnings: list[WarningRow] = []
    seen_public_trials: set[tuple[str, str]] = set()

    for trial in trials:
        key = (trial.participant_id, trial.public_name)
        if key in seen_public_trials:
            raise ValueError(f"Duplicate public trial ID: {key}")
        seen_public_trials.add(key)

        validation_root = trial.data_root / trial.trial_name / "validation"
        if not validation_root.exists():
            warnings.append(
                WarningRow(
                    level="error",
                    code="missing_validation",
                    source_path=str(validation_root),
                    message="Trial validation folder does not exist.",
                )
            )
            continue

        public_trial = cfg.output_root / DATA_DIR / trial.participant_id / trial.public_name

        for system in cfg.systems:
            src = validation_root / system / HUMAN_DATA_PARQUET

            if not src.exists():
                warnings.append(
                    WarningRow(
                        level="error",
                        code="missing_parquet",
                        source_path=str(src),
                        message=f"Required {HUMAN_DATA_PARQUET!r} was not found for system {system!r}.",
                    )
                )
                continue

            dst = public_trial / system / ALIGNED_3D_DATA_DIR / HUMAN_DATA_PARQUET
            size, digest = copy_file(src, dst, dry_run)

            manifest.append(
                ManifestRow(
                    participant_id=trial.participant_id,
                    task=trial.task,
                    trial=trial.trial_number,
                    system=system,
                    relative_path=safe_rel(dst, cfg.output_root),
                    source_path=str(src),
                    size_bytes=size,
                    sha256=digest,
                )
            )

    return manifest, warnings


def write_metadata(
    cfg: Config,
    manifest: Sequence[ManifestRow],
    warnings: Sequence[WarningRow],
    dry_run: bool,
) -> None:
    summary = {
        "dry_run": dry_run,
        "participants_exported": len({row.participant_id for row in manifest}),
        "trials_exported": len({(row.participant_id, row.task, row.trial) for row in manifest}),
        "files_selected": len(manifest),
        "total_size_bytes": sum(row.size_bytes for row in manifest),
        "errors": sum(row.level == "error" for row in warnings),
        "warnings": sum(row.level == "warning" for row in warnings),
        "informational_messages": sum(row.level == "info" for row in warnings),
    }

    print(json.dumps(summary, indent=2))

    if dry_run:
        return

    metadata_root = cfg.output_root / METADATA_DIR
    metadata_root.mkdir(parents=True, exist_ok=True)

    public_fields = [
        "participant_id", "task", "trial", "system",
        "relative_path", "size_bytes", "sha256",
    ]

    with (metadata_root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=public_fields)
        writer.writeheader()
        for row in manifest:
            data = row.__dict__.copy()
            data.pop("source_path")
            writer.writerow(data)

    private_fields = public_fields + ["source_path"]

    with (metadata_root / "PRIVATE_provenance_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=private_fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow(row.__dict__)

    with (metadata_root / "export_warnings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["level", "code", "source_path", "message"])
        writer.writeheader()
        for row in warnings:
            writer.writerow(row.__dict__)

    (metadata_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Parquet-only public FreeMoCap validation dataset export."
    )
    parser.add_argument("config", type=Path, help="Path to the exporter TOML configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect/hash selected files without copying.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cfg = load_toml(args.config)
        manifest, warnings = export(cfg, args.overwrite, args.dry_run)
        write_metadata(cfg, manifest, warnings, args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    serious = [warning for warning in warnings if warning.level in {"error", "warning"}]
    if serious:
        print(f"Review required: {len(serious)} warning/error entries.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
