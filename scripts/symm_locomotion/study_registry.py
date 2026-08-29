# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protocol-fact registry for symmetric-locomotion study cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CLASSIFICATIONS = frozenset(
    {
        "eligible_main",
        "eligible_supplement",
        "historical_diagnostic",
        "ineligible_protocol",
        "failed",
        "incomplete",
    }
)
DESIGN_ROLES = frozenset({"main", "supplement", "historical_diagnostic"})
ARTIFACT_AVAILABILITY = frozenset({"tracked", "local_only"})
_HASH_PATTERN = re.compile(r"^[0-9a-f]+$")
_MISSING = object()
_PERFORMANCE_FACT_ROOTS = frozenset(
    {
        "auc",
        "favorable",
        "final_reward",
        "metric",
        "metrics",
        "outcome",
        "outcomes",
        "performance",
        "result",
        "results",
        "reward_auc",
        "success",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _dotted_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _protocol_constraint_errors(protocol: Mapping[str, Any]) -> list[str]:
    """Reject outcome/performance paths from an eligibility protocol."""
    errors: list[str] = []
    for section in ("required", "allowed"):
        constraints = protocol.get(section, {})
        if not isinstance(constraints, Mapping):
            continue
        for path in constraints:
            root = str(path).split(".", maxsplit=1)[0]
            if root in _PERFORMANCE_FACT_ROOTS:
                errors.append(f"protocol {section} constraint {path} is performance-derived and forbidden")
    required_present = protocol.get("required_present", [])
    if isinstance(required_present, Sequence) and not isinstance(required_present, str | bytes):
        for path in required_present:
            root = str(path).split(".", maxsplit=1)[0]
            if root in _PERFORMANCE_FACT_ROOTS:
                errors.append(f"protocol required_present constraint {path} is performance-derived and forbidden")
    return errors


def materialize_facts(manifest: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    """Merge cohort-wide protocol facts with one run's immutable overrides."""
    common = manifest.get("common_facts", {})
    facts = run.get("facts", {})
    if not isinstance(common, Mapping) or not isinstance(facts, Mapping):
        raise ValueError("common_facts and run facts must be mappings.")
    return _deep_merge(common, facts)


def protocol_mismatches(protocol: Mapping[str, Any], facts: Mapping[str, Any]) -> list[str]:
    """Return fact-only reasons why a run does not satisfy a cohort protocol."""
    reasons = _protocol_constraint_errors(protocol)
    required = protocol.get("required", {})
    allowed = protocol.get("allowed", {})
    required_present = protocol.get("required_present", [])
    if not isinstance(required, Mapping) or not isinstance(allowed, Mapping):
        return ["protocol required/allowed constraints must be mappings"]
    for path, expected in required.items():
        observed = _dotted_value(facts, str(path))
        if observed is _MISSING:
            reasons.append(f"missing required protocol fact {path}")
        elif observed != expected:
            reasons.append(f"protocol fact {path}={observed!r}, expected {expected!r}")
    for path, choices in allowed.items():
        observed = _dotted_value(facts, str(path))
        if observed is _MISSING:
            reasons.append(f"missing allowed-set protocol fact {path}")
        elif not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
            reasons.append(f"protocol allowed constraint {path} must be a sequence")
        elif observed not in choices:
            reasons.append(f"protocol fact {path}={observed!r} is outside the declared allowed set")
    if not isinstance(required_present, Sequence) or isinstance(required_present, str | bytes):
        reasons.append("protocol required_present must be a sequence")
    else:
        for path in required_present:
            observed = _dotted_value(facts, str(path))
            if observed is _MISSING or observed is None or observed == "":
                reasons.append(f"required audit fact {path} is absent")
    return reasons


def classify_run(
    run: Mapping[str, Any], protocol: Mapping[str, Any], *, common_facts: Mapping[str, Any] | None = None
) -> tuple[str, list[str]]:
    """Classify one run solely from design role and immutable protocol facts."""
    run_facts = run.get("facts", {})
    if not isinstance(run_facts, Mapping):
        return "ineligible_protocol", ["run facts must be a mapping"]
    facts = _deep_merge(common_facts or {}, run_facts)
    checkpoint = facts.get("checkpoint_status", {})
    checkpoint_state = checkpoint.get("state") if isinstance(checkpoint, Mapping) else None
    if checkpoint_state in {"failed", "error"}:
        return "failed", [f"checkpoint/training state is {checkpoint_state}"]
    if checkpoint_state not in {"complete", "failed", "error"}:
        return "incomplete", [f"checkpoint state is {checkpoint_state!r}, not 'complete'"]

    role = run.get("design_role")
    if role not in DESIGN_ROLES:
        return "ineligible_protocol", [f"unknown design role {role!r}"]
    mismatches = protocol_mismatches(protocol, facts)
    if mismatches:
        if role == "historical_diagnostic":
            return "historical_diagnostic", mismatches
        return "ineligible_protocol", mismatches
    if role == "historical_diagnostic":
        return "historical_diagnostic", ["run was prospectively assigned to the historical diagnostic stratum"]
    return ("eligible_main" if role == "main" else "eligible_supplement"), []


def classify_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return recomputed classifications for every registered run."""
    protocol = manifest.get("protocol", {})
    common_facts = manifest.get("common_facts", {})
    return [
        {
            "run_id": run.get("run_id"),
            "classification": classification,
            "reasons": reasons,
        }
        for run in manifest.get("runs", [])
        for classification, reasons in (classify_run(run, protocol, common_facts=common_facts),)
    ]


def _validate_hashes(facts: Mapping[str, Any], run_id: str) -> list[str]:
    errors: list[str] = []
    code_hashes = facts.get("code_hashes")
    if not isinstance(code_hashes, Mapping):
        return [f"run {run_id}: code_hashes must be present"]
    for name, length in (
        ("archived_git_commit", 40),
        ("source_snapshot_sha256", 64),
        ("agent_config_sha256", 64),
        ("env_config_sha256", 64),
    ):
        value = code_hashes.get(name)
        if not isinstance(value, str) or len(value) != length or _HASH_PATTERN.fullmatch(value) is None:
            errors.append(f"run {run_id}: code_hashes.{name} must be a lowercase {length}-hex digest")
    checkpoint = facts.get("checkpoint_status", {})
    if isinstance(checkpoint, Mapping) and checkpoint.get("state") == "complete":
        digest = checkpoint.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or _HASH_PATTERN.fullmatch(digest) is None:
            errors.append(f"run {run_id}: a complete checkpoint requires a lowercase SHA-256 digest")
    return errors


def _artifact_fact_binding_errors(run: Mapping[str, Any], facts: Mapping[str, Any], run_id: str) -> list[str]:
    """Require registered artifacts to carry the hashes used for classification."""
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return [f"run {run_id}: artifacts must be a mapping"]
    errors: list[str] = []
    code_hashes = facts.get("code_hashes", {})
    if isinstance(code_hashes, Mapping):
        initialization_artifact = artifacts.get("provenance/initialization.json")
        if initialization_artifact is None:
            bindings = {
                "git/symm_rl_isaaclab.diff": code_hashes.get("source_snapshot_sha256"),
                "params/agent.yaml": code_hashes.get("agent_config_sha256"),
                "params/env.yaml": code_hashes.get("env_config_sha256"),
            }
            for artifact_path, fact_digest in bindings.items():
                artifact_digest = artifacts.get(artifact_path)
                if artifact_digest is None:
                    errors.append(f"run {run_id}: artifacts must include {artifact_path} to bind protocol facts")
                elif artifact_digest != fact_digest:
                    errors.append(
                        f"run {run_id}: artifact hash for {artifact_path} does not match its classification fact"
                    )
        initialization_digest = facts.get("initialization_record_sha256")
        if initialization_digest is not None and initialization_artifact != initialization_digest:
            errors.append(f"run {run_id}: initialization artifact hash does not match its classification fact")

    checkpoint = facts.get("checkpoint_status", {})
    if isinstance(checkpoint, Mapping) and checkpoint.get("state") == "complete":
        final_iteration = checkpoint.get("final_iteration")
        checkpoint_path = f"model_{final_iteration}.pt"
        artifact_digest = artifacts.get(checkpoint_path)
        if artifact_digest is None:
            errors.append(f"run {run_id}: artifacts must include {checkpoint_path} for the completed checkpoint")
        elif artifact_digest != checkpoint.get("sha256"):
            errors.append(f"run {run_id}: checkpoint artifact hash does not match checkpoint_status.sha256")

    for artifact_path, digest in artifacts.items():
        if not isinstance(artifact_path, str) or not artifact_path:
            errors.append(f"run {run_id}: artifact paths must be non-empty strings")
        if not isinstance(digest, str) or len(digest) != 64 or _HASH_PATTERN.fullmatch(digest) is None:
            errors.append(f"run {run_id}: artifact {artifact_path!r} must have a lowercase SHA-256 digest")
    return errors


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate registry structure and declared fact-derived classifications."""
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    cohort_id = manifest.get("cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id:
        errors.append("cohort_id must be non-empty")
    if manifest.get("selection_basis") != "protocol_facts_only":
        errors.append("selection_basis must be 'protocol_facts_only'")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        errors.append("protocol must be a mapping")
        protocol = {}
    errors.extend(_protocol_constraint_errors(protocol))
    common_facts = manifest.get("common_facts", {})
    if not isinstance(common_facts, Mapping):
        errors.append("common_facts must be a mapping")
        common_facts = {}
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        return [*errors, "runs must be a list"]
    seen: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            errors.append("every run entry must be a mapping")
            continue
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            errors.append("every run requires a non-empty run_id")
            continue
        if run_id in seen:
            errors.append(f"duplicate run_id {run_id!r}")
        seen.add(run_id)
        path = run.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"run {run_id}: path must be non-empty")
        availability = run.get("artifact_availability", "tracked")
        if availability not in ARTIFACT_AVAILABILITY:
            errors.append(f"run {run_id}: artifact_availability must be one of {sorted(ARTIFACT_AVAILABILITY)!r}")
        if isinstance(path, str):
            path_parts = Path(path).parts
            if Path(path).is_absolute() or ".." in path_parts:
                errors.append(f"run {run_id}: path must be a repository-relative path without '..'")
            elif availability == "tracked" and tuple(path_parts[:3]) != ("logs", "rsl_rl", "good_runs"):
                errors.append(f"run {run_id}: tracked artifacts must live below logs/rsl_rl/good_runs")
        run_facts = run.get("facts", {})
        if not isinstance(run_facts, Mapping):
            errors.append(f"run {run_id}: facts must be a mapping")
            run_facts = {}
        facts = _deep_merge(common_facts, run_facts)
        errors.extend(_validate_hashes(facts, run_id))
        errors.extend(_artifact_fact_binding_errors(run, facts, run_id))
        classification, _ = classify_run(run, protocol, common_facts=common_facts)
        if classification not in CLASSIFICATIONS:
            errors.append(f"run {run_id}: internal classification {classification!r} is invalid")
        if run.get("classification") != classification:
            errors.append(
                f"run {run_id}: declared classification {run.get('classification')!r} "
                f"does not match fact-derived {classification!r}"
            )
    return errors


def verify_manifest_artifacts(
    manifest: Mapping[str, Any], repo_root: Path, skipped: list[str] | None = None
) -> list[str]:
    """Verify referenced folders and immutable artifact hashes without changing them."""
    errors: list[str] = []
    root = repo_root.resolve()
    tracked_root = (root / "logs" / "rsl_rl" / "good_runs").resolve()
    for run in manifest.get("runs", []):
        run_id = str(run.get("run_id"))
        run_path = Path(str(run.get("path", "")))
        run_path = run_path.resolve() if run_path.is_absolute() else (root / run_path).resolve()
        try:
            run_path.relative_to(root)
        except ValueError:
            errors.append(f"run {run_id}: referenced run folder escapes the repository: {run_path}")
            continue
        if run.get("artifact_availability", "tracked") == "tracked":
            try:
                run_path.relative_to(tracked_root)
            except ValueError:
                errors.append(f"run {run_id}: tracked run folder escapes {tracked_root}")
                continue
        if not run_path.is_dir():
            if run.get("artifact_availability", "tracked") == "local_only":
                if skipped is not None:
                    skipped.append(f"run {run_id}: local-only artifact folder is unavailable: {run_path}")
                continue
            errors.append(f"run {run_id}: referenced run folder does not exist: {run_path}")
            continue
        artifacts = run.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            errors.append(f"run {run_id}: artifacts must be a mapping")
            continue
        for relative_path, expected_sha256 in artifacts.items():
            artifact_path = (run_path / str(relative_path)).resolve()
            try:
                artifact_path.relative_to(run_path)
            except ValueError:
                errors.append(f"run {run_id}: artifact path escapes the run folder: {relative_path}")
                continue
            if not artifact_path.is_file():
                errors.append(f"run {run_id}: artifact is missing: {relative_path}")
            elif _sha256_file(artifact_path) != expected_sha256:
                errors.append(f"run {run_id}: artifact SHA-256 mismatch: {relative_path}")
        initialization_relative = "provenance/initialization.json"
        if initialization_relative in artifacts:
            initialization_path = (run_path / initialization_relative).resolve()
            if initialization_path.is_file():
                try:
                    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"run {run_id}: cannot read initialization provenance: {exc}")
                    continue
                facts = materialize_facts(manifest, run)
                code_hashes = facts.get("code_hashes", {})
                provenance_bindings = {
                    "repo_commit": code_hashes.get("archived_git_commit"),
                    "dirty_tree_diff_sha256": code_hashes.get("source_snapshot_sha256"),
                    "resolved_agent_config_sha256": code_hashes.get("agent_config_sha256"),
                    "resolved_env_config_sha256": code_hashes.get("env_config_sha256"),
                }
                for field_name, expected_value in provenance_bindings.items():
                    if initialization.get(field_name) != expected_value:
                        errors.append(
                            f"run {run_id}: initialization provenance {field_name} does not match protocol facts"
                        )
    return errors


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and structurally validate one study-registry manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load study registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Study registry must contain one JSON object: {path}")
    errors = validate_manifest(payload)
    if errors:
        raise ValueError(f"Invalid study registry {path}:\n- " + "\n- ".join(errors))
    return payload


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--verify_artifacts", "--verify-artifacts", action="store_true")
    parser.add_argument("--repo_root", "--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    failed = False
    for path in args.manifests:
        try:
            manifest = load_manifest(path)
        except ValueError as exc:
            print(exc)
            failed = True
            continue
        skipped: list[str] = []
        errors = verify_manifest_artifacts(manifest, args.repo_root, skipped) if args.verify_artifacts else []
        status = "valid" if not errors else "invalid"
        print(f"{path}: {status}; runs={len(manifest['runs'])}")
        for message in skipped:
            print(f"  - skipped: {message}")
        for error in errors:
            print(f"  - {error}")
        failed |= bool(errors)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
