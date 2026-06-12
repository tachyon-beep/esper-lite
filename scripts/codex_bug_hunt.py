#!/usr/bin/env python3
"""Run Codex bug audits per file across the esper-lite codebase.

Uses OpenAI Codex CLI in read-only sandbox mode to perform deep static
analysis on each Python file. Bug categories are tuned for esper-lite's
domain: RL training (PPO/Simic), morphogenetic lifecycle (Kasmina),
tensor operations (PyTorch), and telemetry contracts (Nissa/Karn).

Usage:
    python scripts/codex_bug_hunt.py
    python scripts/codex_bug_hunt.py --changed-since HEAD~5
    python scripts/codex_bug_hunt.py --branch main
    python scripts/codex_bug_hunt.py --dry-run
    python scripts/codex_bug_hunt.py --rate-limit 30

Prerequisites:
    pip install pyrate-limiter tenacity tqdm
    codex CLI must be on PATH (npm install -g @openai/codex)
"""
from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from codex_audit_common import (  # type: ignore[import-not-found]
    AsyncTqdm,
    chunked,
    ensure_log_file,
    extract_section,
    generate_summary,
    is_cache_path,
    load_context,
    print_summary,
    priority_from_report,
    resolve_path,
    run_codex_with_retry_and_logging,
)
from pyrate_limiter import Duration, Limiter, Rate


def _is_python_file(path: Path) -> bool:
    """Check if path is a Python source file (not test)."""
    return path.suffix == ".py" and not path.name.startswith("test_")


def _build_prompt(file_path: Path, template: str, context: str, extra_message: str | None = None) -> str:
    return (
        "You are a static analysis agent doing a deep bug audit of a morphogenetic\n"
        "neural network framework built on PyTorch. The system uses PPO-based RL to\n"
        "control the growth/pruning of seed modules inside a host neural network.\n"
        f"Target file: {file_path}\n\n"
        "Instructions:\n"
        "- Use the bug report template below verbatim.\n"
        "- Fill in every section. If unknown, write 'Unknown'.\n"
        "- You may read any repo file to confirm integration behavior. Prefer\n"
        "  verification over speculation.\n"
        "- Report bugs only if the primary fix belongs in the target file.\n"
        "  If the root cause is in another file, do not report it unless the\n"
        "  severity is P0. If you report a P0 outside the target file, explain\n"
        "  why and cite the true root-cause file.\n"
        "- Integration issues can reference other files for evidence, but the\n"
        "  actionable fix must be in the target file (unless P0 as above).\n"
        "- If you find multiple distinct bugs, output one full template per bug,\n"
        "  separated by a line with only '---'.\n"
        "- If you find no credible bug, output one template with Summary set to\n"
        f"  'No concrete bug found in {file_path}', Severity 'trivial', Priority 'P3',\n"
        "  and Root Cause Hypothesis 'No bug identified'.\n"
        "- Evidence should cite file paths and line numbers when possible.\n"
        + (f"\nIMPORTANT CONTEXT:\n{extra_message}\n" if extra_message else "")
        + "\n"
        "Bug Categories to Check:\n"
        "\n"
        "1. **RL Training Correctness (Simic/PPO)**:\n"
        "   - Advantage estimation errors (GAE lambda, discount factor application)\n"
        "   - PPO clipping applied incorrectly (ratio computation, surrogate loss)\n"
        "   - Value function target computation errors\n"
        "   - Reward normalization/scaling bugs (running stats, NaN propagation)\n"
        "   - Entropy bonus miscalculation (per-head differential entropy)\n"
        "   - Action masking not respected in loss computation\n"
        "   - Experience buffer indexing errors (off-by-one, stale data)\n"
        "   - LSTM hidden state not reset between episodes\n"
        "   - Explained variance computation errors\n"
        "   - KL divergence early stopping misconfigured\n"
        "\n"
        "2. **Morphogenetic Lifecycle (Kasmina)**:\n"
        "   - Seed state machine violations (invalid transitions, skipped stages)\n"
        "   - Alpha blending schedule errors (wrong ramp direction, division by zero)\n"
        "   - Gradient isolation violations (seed gradients leaking to host during TRAINING)\n"
        "   - SeedSlot extra_state serialization issues (checkpoint save/restore)\n"
        "   - Blueprint registration errors (wrong injection point dimensions)\n"
        "   - Fossilization not freezing parameters correctly\n"
        "   - Pruning not cleaning up resources (dangling references, memory leaks)\n"
        "   - Counterfactual evaluation computing wrong baseline\n"
        "   - Host forward pass with incorrect slot ordering\n"
        "   - MorphogeneticModel state_dict missing seed state\n"
        "\n"
        "3. **Tensor/Device Issues (PyTorch)**:\n"
        "   - CPU/GPU device mismatches (tensors on different devices in same op)\n"
        "   - dtype mismatches (float32 vs float64, int vs float in comparisons)\n"
        "   - In-place operations on tensors requiring grad (breaks autograd)\n"
        "   - torch.compile graph breaks (dynamic shapes, data-dependent control flow)\n"
        "   - Missing .detach() causing unintended gradient flow\n"
        "   - Tensor shape mismatches (broadcasting errors, wrong reshape)\n"
        "   - CUDA stream synchronization missing where needed\n"
        "   - Missing .no_grad() context in inference paths\n"
        "   - Float64 conversion causing unnecessary GPU sync\n"
        "   - .item() calls inside hot loops causing GPU sync stalls\n"
        "\n"
        "4. **Numerical Stability**:\n"
        "   - Log of zero or negative values (log-prob, entropy)\n"
        "   - Division by zero in normalization (batch norm with zero variance)\n"
        "   - Exploding gradients not caught by grad clipping\n"
        "   - NaN/Inf propagation through reward or loss computation\n"
        "   - Softmax overflow on large logits\n"
        "   - Catastrophic cancellation in running statistics\n"
        "   - Unstable loss computation (log-sum-exp without shift)\n"
        "\n"
        "5. **Observation/Action Space Contracts (Leyline)**:\n"
        "   - Observation vector dimension mismatch between builder and consumer\n"
        "   - Action enum values not matching policy head indices\n"
        "   - Feature normalization range violations (values outside expected bounds)\n"
        "   - Missing features in observation (sensor doesn't match capability)\n"
        "   - Slot-level vs global-level feature confusion\n"
        "   - Blueprint embedding dimension mismatch\n"
        "   - FactoredActions encoding/decoding errors\n"
        "\n"
        "6. **Governor & Safety (Tolaria)**:\n"
        "   - Rollback not restoring all necessary state\n"
        "   - Anomaly detection thresholds too loose or too tight\n"
        "   - Safety governor not injecting negative reward on catastrophe\n"
        "   - Validation evaluation using training data or wrong split\n"
        "   - Determinism violations (non-reproducible results with same seed)\n"
        "   - SharedBatchIterator drop_last inconsistency\n"
        "   - Execution engine blocking policy loop (inverted control flow violation)\n"
        "\n"
        "7. **Telemetry Contracts (Nissa/Karn)**:\n"
        "   - Typed payload schema violations (wrong field names or types)\n"
        "   - Missing telemetry emission for observable events\n"
        "   - Telemetry causing training performance regression\n"
        "   - Analytics backend not handling missing/None fields\n"
        "   - Event ordering assumptions violated by async emission\n"
        "   - Karn store/aggregation losing data\n"
        "   - TUI/Overwatch displaying stale or wrong metrics\n"
        "\n"
        "8. **State Management & Concurrency**:\n"
        "   - LSTM hidden state shape mismatch after architecture change\n"
        "   - Checkpoint missing critical state (optimizer, scheduler, normalizer)\n"
        "   - Mutable default arguments in dataclasses or function signatures\n"
        "   - Shared mutable state between vectorized environments\n"
        "   - Race conditions in multi-env parallel execution\n"
        "   - Resource leaks (unclosed files, unreleased GPU memory)\n"
        "   - Context manager __exit__ not cleaning up on exception\n"
        "\n"
        "9. **Performance/Resource Issues**:\n"
        "   - O(n^2) algorithms where O(n) or O(n log n) is possible\n"
        "   - GPU sync points in hot paths (.item(), .cpu(), .numpy())\n"
        "   - Memory leaks from accumulating tensor history (missing detach)\n"
        "   - Blocking I/O in training loop (file writes, logging)\n"
        "   - Redundant computation (re-computing features already available)\n"
        "   - Large tensor copies where in-place or views would suffice\n"
        "   - DataLoader worker configuration issues\n"
        "\n"
        "10. **Architectural / Policy Violations (CLAUDE.md)**:\n"
        "    - Defensive programming patterns hiding bugs (.get(), getattr(), hasattr())\n"
        "    - Backwards compatibility code (prohibited - see CLAUDE.md)\n"
        "    - Legacy shims or deprecated adapters (prohibited)\n"
        "    - Types/enums/constants not in leyline (shared contracts violation)\n"
        "    - Circular imports between domains\n"
        "    - Metaphor violations (body vs plant terminology misuse)\n"
        "    - Missing TODO comments for deferred functionality\n"
        "    - Silent exception swallowing (empty except, catch-all handlers)\n"
        "\n"
        "Analysis Depth Checklist:\n"
        "- [ ] Read CLAUDE.md sections relevant to this file's domain\n"
        "- [ ] Check leyline contract compliance (enum values, stage transitions)\n"
        "- [ ] Verify tensor shapes and device placement through call chain\n"
        "- [ ] Trace reward/loss computation for numerical stability\n"
        "- [ ] Check gradient flow paths for isolation violations\n"
        "- [ ] Validate observation builder matches policy input expectations\n"
        "- [ ] Look for GPU sync points in hot paths\n"
        "- [ ] Check for defensive programming patterns (CLAUDE.md prohibition)\n"
        "\n"
        "Repository context (read-only):\n"
        f"{context}\n\n"
        "Bug report template:\n"
        f"{template}\n"
    )


def _extract_file_references(text: str) -> set[str]:
    """Extract all file paths referenced in the text."""
    patterns = [
        r"\b([\w./-]+/[\w./-]+\.py):\d+",
        r"\b(src/[\w./-]+\.py)\b",
        r"`([\w./-]+\.py)`",
    ]
    files = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            files.add(match.group(1))
    return files


def _calculate_bug_similarity(report1: str, report2: str) -> float:
    """Calculate similarity score between two bug reports (0.0 to 1.0)."""
    summary1 = extract_section(report1, "Summary").lower()
    summary2 = extract_section(report2, "Summary").lower()
    evidence1 = extract_section(report1, "Evidence").lower()
    evidence2 = extract_section(report2, "Evidence").lower()

    files1 = _extract_file_references(report1)
    files2 = _extract_file_references(report2)

    if files1 and files2:
        file_overlap = len(files1 & files2) / len(files1 | files2)
    else:
        file_overlap = 0.0

    def word_similarity(text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    summary_sim = word_similarity(summary1, summary2)
    evidence_sim = word_similarity(evidence1, evidence2)

    return 0.5 * file_overlap + 0.3 * summary_sim + 0.2 * evidence_sim


def _find_similar_bug(report: str, bugs_dir: Path, threshold: float = 0.6) -> Path | None:
    """Search for similar bug in docs/bugs/fixed/. Returns path if found."""
    if not bugs_dir.exists():
        return None

    target_files = _extract_file_references(report)
    if not target_files:
        return None

    best_match: tuple[Path, float] | None = None

    for bug_file in bugs_dir.rglob("*.md"):
        if bug_file.name in ("README.md", "ticket-template.md"):
            continue

        existing_text = bug_file.read_text(encoding="utf-8")
        similarity = _calculate_bug_similarity(report, existing_text)

        if similarity >= threshold and (best_match is None or similarity > best_match[1]):
            best_match = (bug_file, similarity)

    return best_match[0] if best_match else None


def _merge_bug_reports(existing_path: Path, new_report: str, repo_root: Path) -> str:
    """Merge new analysis into existing bug report. Returns log message."""
    existing_text = existing_path.read_text(encoding="utf-8")

    new_evidence = extract_section(new_report, "Evidence")
    new_root_cause = extract_section(new_report, "Root Cause Hypothesis")

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    verification_section = (
        f"\n\n---\n\n## Re-verification ({timestamp})\n\n"
        f"**Status: RE-ANALYZED**\n\n"
        f"### New Analysis\n\n"
        f"Re-ran static analysis on {timestamp}. Key findings:\n\n"
        f"**Evidence:**\n{new_evidence}\n\n"
        f"**Root Cause:**\n{new_root_cause}\n"
    )

    updated_text = existing_text.rstrip() + verification_section
    existing_path.write_text(updated_text, encoding="utf-8")

    rel_path = existing_path.relative_to(repo_root) if repo_root else existing_path
    return f"Updated existing bug: {rel_path}"


def _deduplicate_and_merge(
    output_path: Path,
    bugs_dir: Path,
    repo_root: Path,
    similarity_threshold: float = 0.6,
) -> int:
    """Check generated reports against existing bugs and merge duplicates.

    Returns count of bugs merged into existing reports.
    """
    if not output_path.exists():
        return 0

    text = output_path.read_text(encoding="utf-8")
    reports: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            reports.append("\n".join(current).strip())
            current = []
            continue
        current.append(line)
    if current:
        reports.append("\n".join(current).strip())

    merged_count = 0
    kept_reports = []

    for report in reports:
        if not report.strip():
            continue

        summary = extract_section(report, "Summary")
        if "no concrete bug found" in summary.lower():
            continue

        similar_bug = _find_similar_bug(report, bugs_dir, similarity_threshold)

        if similar_bug:
            _merge_bug_reports(similar_bug, report, repo_root)
            merged_count += 1
        else:
            kept_reports.append(report)

    if kept_reports:
        new_text = "\n---\n".join(kept_reports).rstrip() + "\n"
        output_path.write_text(new_text, encoding="utf-8")
    elif output_path.exists():
        output_path.unlink()

    return merged_count


def _organize_by_priority(output_dir: Path) -> None:
    """Organize outputs into by-priority/ subdirectories."""
    by_priority_dir = output_dir / "by-priority"

    for md_file in output_dir.rglob("*.md"):
        if "by-priority" in md_file.parts:
            continue

        text = md_file.read_text(encoding="utf-8")
        priority = priority_from_report(text)

        dest_dir = by_priority_dir / priority
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / md_file.name
        shutil.copy2(md_file, dest_path)


async def _run_batches(
    *,
    files: list[Path],
    output_dir: Path,
    model: str | None,
    prompt_template: str,
    repo_root: Path,
    skip_existing: bool,
    batch_size: int,
    root_dir: Path,
    log_path: Path,
    context: str,
    rate_limit: int | None,
    organize_by_priority: bool,
    bugs_dir: Path | None,
    deduplicate: bool,
    extra_message: str | None = None,
) -> dict[str, int]:
    """Run analysis in batches. Returns statistics."""
    log_lock = asyncio.Lock()
    failed_files: list[tuple[Path, Exception]] = []
    total_merged = 0
    total_gated = 0

    rate_limiter = Limiter(Rate(rate_limit, Duration.MINUTE)) if rate_limit else None

    pbar = AsyncTqdm(total=len(files), desc="Analyzing files", unit="file")

    for batch in chunked(files, batch_size):
        tasks: list[asyncio.Task[dict[str, int]]] = []
        batch_files: list[Path] = []

        for file_path in batch:
            relative = file_path.relative_to(root_dir)
            output_path = output_dir / relative
            output_path = output_path.with_suffix(output_path.suffix + ".md")

            if skip_existing and output_path.exists():
                pbar.update(1)
                continue

            prompt = _build_prompt(file_path, prompt_template, context, extra_message)
            batch_files.append(file_path)

            task = asyncio.create_task(
                run_codex_with_retry_and_logging(
                    file_path=file_path,
                    output_path=output_path,
                    model=model,
                    prompt=prompt,
                    repo_root=repo_root,
                    log_path=log_path,
                    log_lock=log_lock,
                    file_display=str(file_path.relative_to(repo_root).as_posix()),
                    output_display=str(output_path.relative_to(repo_root).as_posix()),
                    rate_limiter=rate_limiter,
                    evidence_gate_summary_prefix="",
                )
            )
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for file_path, result in zip(batch_files, results, strict=False):
                if isinstance(result, Exception):
                    failed_files.append((file_path, result))
                elif isinstance(result, dict):
                    total_gated += result["gated"]

                    if deduplicate and bugs_dir and bugs_dir.exists():
                        relative = file_path.relative_to(root_dir)
                        output_path = output_dir / relative
                        output_path = output_path.with_suffix(output_path.suffix + ".md")
                        merged_count = _deduplicate_and_merge(output_path, bugs_dir, repo_root)
                        total_merged += merged_count

                pbar.update(1)

    pbar.close()

    print(f"\n{'─' * 60}", file=sys.stderr)

    if failed_files:
        print(f"\n{len(failed_files)} files failed:", file=sys.stderr)
        for path, exc in failed_files[:10]:
            print(f"  {path.relative_to(repo_root)}: {exc}", file=sys.stderr)
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more (see {log_path})", file=sys.stderr)

    if deduplicate and total_merged > 0:
        print(f"\n{total_merged} bugs merged into existing reports")

    if organize_by_priority:
        _organize_by_priority(output_dir)

    summary: dict[str, int] = generate_summary(output_dir, no_defect_marker="No concrete bug found")
    summary["merged"] = total_merged
    summary["gated"] = total_gated
    return summary


def _paths_from_file(path_file: Path, repo_root: Path, root_dir: Path) -> list[Path]:
    selected: list[Path] = []
    lines = path_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw_path = Path(stripped)
        path = raw_path if raw_path.is_absolute() else (repo_root / raw_path).resolve()
        if not path.exists():
            raise RuntimeError(f"paths-from entry does not exist: {raw_path}")
        if path.is_dir():
            selected.extend([p for p in path.rglob("*") if p.is_file() and not is_cache_path(p)])
        else:
            if not is_cache_path(path):
                selected.append(path)
    return [path for path in selected if _is_under_root(path, root_dir)]


def _is_under_root(path: Path, root_dir: Path) -> bool:
    try:
        path.relative_to(root_dir)
        return True
    except ValueError:
        return False


def _changed_files_since(repo_root: Path, root_dir: Path, git_ref: str) -> list[Path]:
    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", git_ref, "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _changed_files_on_branch(repo_root: Path, root_dir: Path, base_branch: str) -> list[Path]:
    """Get files changed on current branch vs base branch using merge-base."""
    merge_base_cmd = ["git", "merge-base", base_branch, "HEAD"]
    result = subprocess.run(merge_base_cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git merge-base failed for {base_branch}")
    merge_base = result.stdout.strip()

    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", f"{merge_base}..HEAD", "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")

    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _changed_files_in_range(repo_root: Path, root_dir: Path, commit_range: str) -> list[Path]:
    """Get files changed in commit range (e.g., 'abc123..def456')."""
    if ".." not in commit_range:
        raise ValueError(f"Invalid commit range format: {commit_range}. Expected format: 'START..END'")

    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", commit_range, "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git diff failed for range {commit_range}")

    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _list_files(
    *,
    root_dir: Path,
    repo_root: Path,
    changed_since: str | None,
    branch: str | None,
    commit_range: str | None,
    paths_from: Path | None,
    file_type: str,
) -> list[Path]:
    git_filters = [changed_since, branch, commit_range]
    active_filters = [f for f in git_filters if f is not None]
    if len(active_filters) > 1:
        raise ValueError("Only one of --changed-since, --branch, or --commit-range can be used at a time")

    selected: set[Path] | None = None

    if changed_since:
        changed = set(_changed_files_since(repo_root, root_dir, changed_since))
        selected = changed if selected is None else selected & changed

    if branch:
        changed = set(_changed_files_on_branch(repo_root, root_dir, branch))
        selected = changed if selected is None else selected & changed

    if commit_range:
        changed = set(_changed_files_in_range(repo_root, root_dir, commit_range))
        selected = changed if selected is None else selected & changed

    if paths_from:
        listed = set(_paths_from_file(paths_from, repo_root, root_dir))
        selected = listed if selected is None else selected & listed

    if selected is None:
        selected = {path for path in root_dir.rglob("*") if path.is_file() and not is_cache_path(path)}

    if file_type == "python":
        selected = {p for p in selected if _is_python_file(p)}

    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex bug audits per file on the esper-lite codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all Python files in src/esper
  %(prog)s

  # Scan only changed files since HEAD~5
  %(prog)s --changed-since HEAD~5

  # Scan files changed on current branch vs main
  %(prog)s --branch main

  # Scan files changed in a specific commit range
  %(prog)s --commit-range abc123..def456

  # Dry run to see what would be scanned
  %(prog)s --dry-run

  # Scan a specific domain only
  %(prog)s --root src/esper/simic

  # Use rate limiting for API quota management
  %(prog)s --rate-limit 30

  # Organize outputs by priority
  %(prog)s --organize-by-priority

  # Add extra context (e.g., recent refactor notes)
  %(prog)s --extra-message "Recent env-refactor branch changed observation builders"
        """,
    )
    parser.add_argument(
        "--root",
        default="src/esper",
        help="Root directory to scan for files (default: src/esper).",
    )
    parser.add_argument(
        "--template",
        default="docs/bugs/ticket-template.md",
        help="Bug report template path (default: docs/bugs/ticket-template.md).",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/bugs/generated",
        help="Directory to write bug reports (default: docs/bugs/generated).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum concurrent Codex runs per batch (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Codex model (passes --model to codex exec).",
    )
    parser.add_argument(
        "--changed-since",
        default=None,
        help="Only scan files changed since this git ref (e.g. HEAD~1).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Compare against base branch to find changed files (e.g. 'main').",
    )
    parser.add_argument(
        "--commit-range",
        default=None,
        help="Only scan files changed in commit range (e.g. 'abc123..def456').",
    )
    parser.add_argument(
        "--paths-from",
        default=None,
        help="Path to a file containing newline-separated paths to scan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have an output report.",
    )
    parser.add_argument(
        "--file-type",
        default="python",
        choices=["python", "all"],
        help="Filter by file type (default: python, excludes tests).",
    )
    parser.add_argument(
        "--context-files",
        nargs="+",
        default=None,
        help="Additional context files beyond CLAUDE.md/ROADMAP.md.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="Max requests per minute (e.g., 30 for API quota management).",
    )
    parser.add_argument(
        "--organize-by-priority",
        action="store_true",
        help="Organize outputs into by-priority/ subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be scanned without running analysis.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Check generated bugs against docs/bugs/fixed/ and merge duplicates.",
    )
    parser.add_argument(
        "--bugs-dir",
        default="docs/bugs/fixed",
        help="Directory to search for existing bugs (default: docs/bugs/fixed).",
    )
    parser.add_argument(
        "--extra-message",
        default=None,
        help="Additional context for the analysis prompt (e.g., refactor notes).",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.rate_limit is not None and args.rate_limit < 1:
        raise ValueError("--rate-limit must be >= 1")

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    root_dir = resolve_path(repo_root, args.root)
    template_path = resolve_path(repo_root, args.template)
    output_dir = resolve_path(repo_root, args.output_dir)
    log_path = resolve_path(repo_root, "docs/bugs/generated/CODEX_LOG.md")
    bugs_dir = resolve_path(repo_root, args.bugs_dir) if args.deduplicate else None

    paths_from = resolve_path(repo_root, args.paths_from) if args.paths_from else None
    files = _list_files(
        root_dir=root_dir,
        repo_root=repo_root,
        changed_since=args.changed_since,
        branch=args.branch,
        commit_range=args.commit_range,
        paths_from=paths_from,
        file_type=args.file_type,
    )

    if not files:
        print(f"No files found under {root_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Would analyze {len(files)} files:")
        for f in files[:20]:
            print(f"  {f.relative_to(repo_root)}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return 0

    template_text = template_path.read_text(encoding="utf-8")
    context_text = load_context(repo_root, extra_files=args.context_files)
    ensure_log_file(log_path, header_title="Codex Bug Hunt Log")

    stats = asyncio.run(
        _run_batches(
            files=files,
            output_dir=output_dir,
            model=args.model,
            prompt_template=template_text,
            repo_root=repo_root,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            root_dir=root_dir,
            log_path=log_path,
            context=context_text,
            rate_limit=args.rate_limit,
            organize_by_priority=args.organize_by_priority,
            bugs_dir=bugs_dir,
            deduplicate=args.deduplicate,
            extra_message=args.extra_message,
        )
    )

    print_summary(stats, icon="#", title="Esper Bug Hunt Summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
