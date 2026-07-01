#!/usr/bin/env python
"""
Create a de-identified public export of the FreeMoCap validation dataset.

Private source layout
---------------------
Treadmill/gait trials are expected to contain:

    <trial>/validation/<system>/
        freemocap_data_by_frame.parquet
        freemocap_data_by_frame.csv
        transformation_3d.npy
        gait_events/
        gait_parameters/
        joint_angles/
        trajectories/
        rmse/

Balance/NIH trials are expected to contain:

    <trial>/validation/<system>/
        freemocap_data_by_frame.parquet
        freemocap_data_by_frame.csv
        transformation_3d.npy
        path_length_analysis/<analysis_folder>/
        rmse/

The selected balance analysis folder is read from the participant registry YAML:

- qualisys_analysis_folder for Qualisys
- trackers[].analysis_folder for markerless trackers

Older timestamped analysis folders are not exported.

Public output layout
--------------------
Each public trial is written as:

    data/<participant_id>/<task-run>/<system>/
        aligned_3d_data/
        analysis_outputs/

Only the canonical aligned files and selected downstream analysis outputs are
included in the public release.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with:\n"
        "  uv add pyyaml\n"
        "or run this script with:\n"
        "  uv run --with pyyaml python export_validation_dataset.py ..."
    ) from exc


# =============================================================================
# Public dataset layout
# =============================================================================

DATA_DIR = "data"
METADATA_DIR = "metadata"

ALIGNED_3D_DATA_DIR = "aligned_3d_data"
ANALYSIS_OUTPUTS_DIR = "analysis_outputs"

QUALISYS_SYSTEM = "qualisys"

HUMAN_DATA_PARQUET = "freemocap_data_by_frame.parquet"
HUMAN_DATA_CSV = "freemocap_data_by_frame.csv"
ALIGNMENT_TRANSFORM = "transformation_3d.npy"

PATH_LENGTH_ANALYSIS_DIR = "path_length_analysis"
RMSE_DIR = "rmse"

DEFAULT_SYSTEMS = (
    QUALISYS_SYSTEM,
    "mediapipe",
    "vitpose",
    "rtmpose",
)

DEFAULT_GAIT_ANALYSIS_FOLDERS = (
    "gait_events",
    "gait_parameters",
    "joint_angles",
    "trajectories",
    RMSE_DIR,
)

DEFAULT_CANONICAL_PATTERNS = (
    HUMAN_DATA_PARQUET,
    HUMAN_DATA_CSV,
    ALIGNMENT_TRANSFORM,
)

CANONICAL_ARTIFACT_TYPES = {
    HUMAN_DATA_PARQUET: "human_data_parquet",
    HUMAN_DATA_CSV: "human_data_csv",
    ALIGNMENT_TRANSFORM: "alignment_transform",
}

DEFAULT_ANALYSIS_EXTENSIONS = (
    ".csv",
    ".tsv",
    ".parquet",
    ".json",
    # ".npy",
)

DEFAULT_EXCLUDED_EXTENSIONS = (
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".ipynb",
    ".blend",
    ".blend1",
    ".mp4",
    ".avi",
    ".mov",
)

TEXT_EXTENSIONS = {
    ".txt",
    ".csv",
    # ".tsv",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".py",
}


@dataclass(frozen=True)
class Trial:
    participant_id: str
    private_participant_folder: str
    trial_type: str
    trial_number: int
    trial_name: str
    data_root: Path
    conditions: tuple[str, ...]
    analysis_folders: dict[str, str]

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
    gait_analysis_folders: tuple[str, ...]
    canonical_patterns: tuple[str, ...]
    analysis_extensions: tuple[str, ...]
    excluded_extensions: tuple[str, ...]
    sensitive_terms: tuple[str, ...]
    include_rmse: bool


@dataclass
class ManifestRow:
    participant_id: str
    task: str
    trial: int
    system: str
    data_stage: str
    artifact_type: str
    condition: str
    relative_path: str
    source_relative_path: str
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
    privacy = raw.get("privacy", {})
    identity = raw.get("identity", {})

    registry_yamls = tuple(
        Path(item).expanduser() for item in paths["registry_yamls"]
    )
    participant_ids = {
        str(key): str(value)
        for key, value in identity.get("participant_ids", {}).items()
    }

    gait_folders = list(
        export.get(
            "gait_analysis_folders",
            export.get(
                "gait_derivative_folders",
                DEFAULT_GAIT_ANALYSIS_FOLDERS,
            ),
        )
    )

    include_rmse = bool(export.get("include_rmse", True))
    if not include_rmse:
        gait_folders = [folder for folder in gait_folders if folder != RMSE_DIR]

    return Config(
        output_root=Path(paths["output_root"]).expanduser(),
        registry_yamls=registry_yamls,
        participant_ids=participant_ids,
        systems=tuple(export.get("systems", DEFAULT_SYSTEMS)),
        gait_analysis_folders=tuple(gait_folders),
        canonical_patterns=tuple(
            export.get("canonical_patterns", DEFAULT_CANONICAL_PATTERNS)
        ),
        analysis_extensions=tuple(
            extension.lower()
            for extension in export.get(
                "analysis_extensions",
                export.get(
                    "derivative_extensions",
                    DEFAULT_ANALYSIS_EXTENSIONS,
                ),
            )
        ),
        excluded_extensions=tuple(
            extension.lower()
            for extension in export.get(
                "excluded_extensions",
                DEFAULT_EXCLUDED_EXTENSIONS,
            )
        ),
        sensitive_terms=tuple(privacy.get("sensitive_terms", ())),
        include_rmse=include_rmse,
    )


def load_trials(cfg: Config) -> list[Trial]:
    trials: list[Trial] = []

    for yaml_path in cfg.registry_yamls:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        private_code = str(raw["participant_code"])
        private_folder = Path(str(raw["data_root"])).name
        participant_id = cfg.participant_ids.get(private_code)

        if not participant_id:
            raise ValueError(
                "No public participant ID configured for "
                f"participant_code={private_code!r}"
            )

        data_root = Path(str(raw["data_root"]))

        for item in raw.get("trials", []):
            trial_type = str(item["trial_type"]).lower()
            analysis_folders: dict[str, str] = {}

            if trial_type in {"balance", "nih"}:
                qualisys_folder = item.get("qualisys_analysis_folder")
                if qualisys_folder:
                    analysis_folders[QUALISYS_SYSTEM] = str(qualisys_folder)

                for tracker in item.get("trackers", []):
                    tracker_name = str(tracker["tracker"])
                    selected_folder = tracker.get("analysis_folder")
                    if selected_folder:
                        analysis_folders[tracker_name] = str(selected_folder)

            trials.append(
                Trial(
                    participant_id=participant_id,
                    private_participant_folder=private_folder,
                    trial_type=trial_type,
                    trial_number=int(item["trial_number"]),
                    trial_name=str(item["trial_name"]),
                    data_root=data_root,
                    conditions=tuple(
                        str(condition)
                        for condition in item.get("conditions", [])
                    ),
                    analysis_folders=analysis_folders,
                )
            )

    return trials


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatch(name.lower(), pattern.lower())
        for pattern in patterns
    )


def should_copy_canonical(path: Path, cfg: Config) -> bool:
    if path.suffix.lower() in cfg.excluded_extensions:
        return False

    return matches_any(path.name, cfg.canonical_patterns)


def iter_selected_files(root: Path, cfg: Config) -> Iterator[Path]:
    if not root.exists():
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in cfg.excluded_extensions:
            continue

        if suffix not in cfg.analysis_extensions:
            continue

        yield path


def copy_file(
    src: Path,
    dst: Path,
    dry_run: bool,
) -> tuple[int, str]:
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


def privacy_findings(
    path: Path,
    terms: Sequence[str],
) -> list[str]:
    findings: list[str] = []

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return findings

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"Could not scan file: {exc}"]

    lowered = text.lower()

    for term in terms:
        if term and term.lower() in lowered:
            findings.append(f"Sensitive term found: {term!r}")

    patterns = (
        r"[A-Za-z]:\\Users\\[^\\\r\n]+",
        r"[A-Za-z]:\\[^,\r\n\t\"]+",
        r"/home/[^/\r\n]+",
        r"/Users/[^/\r\n]+",
    )

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            findings.append(
                f"Possible local path matching {pattern!r}"
            )

    return findings


def add_privacy_warnings(
    warnings: list[WarningRow],
    source: Path,
    scan_terms: Sequence[str],
) -> None:
    for finding in privacy_findings(source, scan_terms):
        warnings.append(
            WarningRow(
                level="warning",
                code="privacy_scan",
                source_path=str(source),
                message=finding,
            )
        )


def condition_from_relative_path(relative_path: Path) -> str:
    if not relative_path.parts:
        return ""

    first_part = relative_path.parts[0]

    if first_part.startswith("speed_"):
        return first_part

    return ""


def export(
    cfg: Config,
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[ManifestRow], list[WarningRow]]:
    trials = load_trials(cfg)

    if cfg.output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {cfg.output_root}. "
                "Use --overwrite after reviewing the path."
            )

        if not dry_run:
            shutil.rmtree(cfg.output_root)

    if not dry_run:
        cfg.output_root.mkdir(parents=True)

    manifest: list[ManifestRow] = []
    warnings: list[WarningRow] = []

    automatic_terms = set(cfg.sensitive_terms)

    for trial in trials:
        automatic_terms.add(trial.private_participant_folder)

        tokens = re.split(
            r"[_\-\s]+",
            trial.private_participant_folder,
        )
        automatic_terms.update(
            token
            for token in tokens
            if len(token) >= 2 and not token.isdigit()
        )

    scan_terms = tuple(sorted(automatic_terms))
    seen_public_trials: set[tuple[str, str]] = set()

    for trial in trials:
        key = (trial.participant_id, trial.public_name)

        if key in seen_public_trials:
            raise ValueError(f"Duplicate public trial ID: {key}")

        seen_public_trials.add(key)

        trial_source = trial.data_root / trial.trial_name
        validation_root = trial_source / "validation"

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

        public_trial = (
            cfg.output_root
            / DATA_DIR
            / trial.participant_id
            / trial.public_name
        )

        for system in cfg.systems:
            system_source = validation_root / system

            if not system_source.exists():
                warnings.append(
                    WarningRow(
                        level="warning",
                        code="missing_system",
                        source_path=str(system_source),
                        message=(
                            f"Expected system folder {system!r} "
                            "was not found."
                        ),
                    )
                )
                continue

            system_public = public_trial / system

            copied_canonical = 0

            for src in sorted(
                path
                for path in system_source.iterdir()
                if path.is_file()
            ):
                if not should_copy_canonical(src, cfg):
                    warnings.append(
                        WarningRow(
                            level="info",
                            code="unrecognized_root_file",
                            source_path=str(src),
                            message=(
                                "Root file was not copied because it did "
                                "not match the canonical allowlist."
                            ),
                        )
                    )
                    continue

                dst = (
                    system_public
                    / ALIGNED_3D_DATA_DIR
                    / src.name
                )
                size, digest = copy_file(src, dst, dry_run)
                copied_canonical += 1

                manifest.append(
                    ManifestRow(
                        participant_id=trial.participant_id,
                        task=trial.task,
                        trial=trial.trial_number,
                        system=system,
                        data_stage=ALIGNED_3D_DATA_DIR,
                        artifact_type=CANONICAL_ARTIFACT_TYPES.get(
                            src.name,
                            "canonical_aligned_data",
                        ),
                        condition="",
                        relative_path=safe_rel(
                            dst,
                            cfg.output_root,
                        ),
                        source_relative_path=str(src),
                        size_bytes=size,
                        sha256=digest,
                    )
                )

                add_privacy_warnings(
                    warnings,
                    src,
                    scan_terms,
                )

            if copied_canonical == 0:
                warnings.append(
                    WarningRow(
                        level="warning",
                        code="no_canonical_files",
                        source_path=str(system_source),
                        message=(
                            "No canonical aligned 3D data files "
                            "were selected."
                        ),
                    )
                )

            if trial.task == "balance":
                selected = trial.analysis_folders.get(system)

                if not selected:
                    warnings.append(
                        WarningRow(
                            level="error",
                            code="missing_selected_analysis",
                            source_path=str(system_source),
                            message=(
                                "No selected path-length analysis folder "
                                "was provided by the registry YAML."
                            ),
                        )
                    )
                    continue

                analysis_root = (
                    system_source
                    / PATH_LENGTH_ANALYSIS_DIR
                )
                selected_source = analysis_root / selected

                if not selected_source.exists():
                    warnings.append(
                        WarningRow(
                            level="error",
                            code="selected_analysis_not_found",
                            source_path=str(selected_source),
                            message=(
                                "The YAML-selected path-length analysis "
                                "folder does not exist."
                            ),
                        )
                    )
                    continue

                for src in iter_selected_files(
                    selected_source,
                    cfg,
                ):
                    rel = src.relative_to(selected_source)
                    dst = (
                        system_public
                        / ANALYSIS_OUTPUTS_DIR
                        / PATH_LENGTH_ANALYSIS_DIR
                        / rel
                    )
                    size, digest = copy_file(
                        src,
                        dst,
                        dry_run,
                    )

                    manifest.append(
                        ManifestRow(
                            participant_id=trial.participant_id,
                            task=trial.task,
                            trial=trial.trial_number,
                            system=system,
                            data_stage=ANALYSIS_OUTPUTS_DIR,
                            artifact_type=PATH_LENGTH_ANALYSIS_DIR,
                            condition="",
                            relative_path=safe_rel(
                                dst,
                                cfg.output_root,
                            ),
                            source_relative_path=str(src),
                            size_bytes=size,
                            sha256=digest,
                        )
                    )

                    add_privacy_warnings(
                        warnings,
                        src,
                        scan_terms,
                    )

                all_versions = (
                    sorted(
                        path.name
                        for path in analysis_root.iterdir()
                        if path.is_dir()
                    )
                    if analysis_root.exists()
                    else []
                )

                ignored_versions = [
                    folder
                    for folder in all_versions
                    if folder != selected
                ]

                if ignored_versions:
                    warnings.append(
                        WarningRow(
                            level="info",
                            code="older_analysis_versions_excluded",
                            source_path=str(analysis_root),
                            message=(
                                f"Selected {selected!r}; excluded "
                                f"{len(ignored_versions)} other "
                                "analysis folders."
                            ),
                        )
                    )

                if cfg.include_rmse:
                    rmse_root = system_source / RMSE_DIR

                    for src in iter_selected_files(
                        rmse_root,
                        cfg,
                    ):
                        rel = src.relative_to(rmse_root)
                        dst = (
                            system_public
                            / ANALYSIS_OUTPUTS_DIR
                            / RMSE_DIR
                            / rel
                        )
                        size, digest = copy_file(
                            src,
                            dst,
                            dry_run,
                        )

                        manifest.append(
                            ManifestRow(
                                participant_id=trial.participant_id,
                                task=trial.task,
                                trial=trial.trial_number,
                                system=system,
                                data_stage=ANALYSIS_OUTPUTS_DIR,
                                artifact_type=RMSE_DIR,
                                condition="",
                                relative_path=safe_rel(
                                    dst,
                                    cfg.output_root,
                                ),
                                source_relative_path=str(src),
                                size_bytes=size,
                                sha256=digest,
                            )
                        )

                        add_privacy_warnings(
                            warnings,
                            src,
                            scan_terms,
                        )

            else:
                for analysis_folder in cfg.gait_analysis_folders:
                    source = system_source / analysis_folder

                    if not source.exists():
                        warnings.append(
                            WarningRow(
                                level="info",
                                code="missing_analysis_output",
                                source_path=str(source),
                                message=(
                                    f"Analysis output folder "
                                    f"{analysis_folder!r} was not found."
                                ),
                            )
                        )
                        continue

                    for src in iter_selected_files(source, cfg):
                        rel = src.relative_to(source)
                        condition = condition_from_relative_path(rel)
                        dst = (
                            system_public
                            / ANALYSIS_OUTPUTS_DIR
                            / analysis_folder
                            / rel
                        )
                        size, digest = copy_file(
                            src,
                            dst,
                            dry_run,
                        )

                        manifest.append(
                            ManifestRow(
                                participant_id=trial.participant_id,
                                task=trial.task,
                                trial=trial.trial_number,
                                system=system,
                                data_stage=ANALYSIS_OUTPUTS_DIR,
                                artifact_type=analysis_folder,
                                condition=condition,
                                relative_path=safe_rel(
                                    dst,
                                    cfg.output_root,
                                ),
                                source_relative_path=str(src),
                                size_bytes=size,
                                sha256=digest,
                            )
                        )

                        add_privacy_warnings(
                            warnings,
                            src,
                            scan_terms,
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
        "participants_exported": len(
            {row.participant_id for row in manifest}
        ),
        "trials_exported": len(
            {
                (row.participant_id, row.task, row.trial)
                for row in manifest
            }
        ),
        "files_selected": len(manifest),
        "total_size_bytes": sum(
            row.size_bytes for row in manifest
        ),
        "errors": sum(
            warning.level == "error"
            for warning in warnings
        ),
        "warnings": sum(
            warning.level == "warning"
            for warning in warnings
        ),
        "informational_messages": sum(
            warning.level == "info"
            for warning in warnings
        ),
    }

    print(json.dumps(summary, indent=2))

    if dry_run:
        return

    metadata_root = cfg.output_root / METADATA_DIR
    metadata_root.mkdir(parents=True, exist_ok=True)

    public_fields = [
        "participant_id",
        "task",
        "trial",
        "system",
        "data_stage",
        "artifact_type",
        "condition",
        "relative_path",
        "size_bytes",
        "sha256",
    ]

    with (metadata_root / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=public_fields,
        )
        writer.writeheader()

        for row in manifest:
            data = row.__dict__.copy()
            data.pop("source_relative_path")
            writer.writerow(data)

    private_fields = public_fields + [
        "source_relative_path"
    ]

    with (
        metadata_root
        / "PRIVATE_provenance_manifest.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=private_fields,
        )
        writer.writeheader()

        for row in manifest:
            writer.writerow(row.__dict__)

    with (
        metadata_root
        / "export_warnings.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "level",
                "code",
                "source_path",
                "message",
            ],
        )
        writer.writeheader()

        for row in warnings:
            writer.writerow(row.__dict__)

    (
        metadata_root
        / "export_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a de-identified public export of the "
            "FreeMoCap validation dataset."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the exporter TOML configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and hash selected files without copying them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output directory.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cfg = load_toml(args.config)
        manifest, warnings = export(
            cfg,
            args.overwrite,
            args.dry_run,
        )
        write_metadata(
            cfg,
            manifest,
            warnings,
            args.dry_run,
        )
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    serious = [
        warning
        for warning in warnings
        if warning.level in {"error", "warning"}
    ]

    if serious:
        print(
            f"Review required: {len(serious)} "
            "warning/error entries.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    #uv run python export_for_public/export_validation_dataset.py D:/validation/public_export_config.toml --overwrite