"""Generate TrainingPair data from SWE-bench instances.

Two modes:

  training  — gold patch (positive) vs LLM unconstrained (negative)
              Output: data/cache/training_pairs.jsonl

  zeroshot  — LLM controlled patch (positive, verified) vs LLM unconstrained (negative)
              Output: data/cache/zeroshot_pairs.jsonl
              Used exclusively for evaluation, never for training.

Usage::

    python scripts/collect_pairs.py --mode training \\
        --input  data/raw/swebench_instances.jsonl \\
        --output data/cache/training_pairs.jsonl \\
        --workers 4 \\
        --llm-provider openai --llm-model gpt-4o

    python scripts/collect_pairs.py --mode zeroshot \\
        --output data/cache/zeroshot_pairs.jsonl

For local vLLM (no internet needed on compute node)::

    python scripts/collect_pairs.py --mode training \\
        --llm-provider openai \\
        --llm-api-base http://localhost:8000/v1 \\
        --llm-model Qwen/Qwen2.5-Coder-32B-Instruct \\
        --workers 8

Resume: re-run the same command; already-processed instance_ids are skipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# On-demand repo cloning into $TMPDIR (方案C)
# ---------------------------------------------------------------------------

def _ensure_mirror(mirror_path: Path, repo: str) -> None:
    """Ensure a --mirror clone of github.com/repo exists at mirror_path.

    Process-safe: uses an exclusive sentinel file so parallel workers don't
    race to clone the same repo.
    """
    if mirror_path.exists():
        sentinel = mirror_path.parent / f"{mirror_path.name}.cloning"
        while sentinel.exists():
            time.sleep(2)
        return

    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = mirror_path.parent / f"{mirror_path.name}.cloning"
    try:
        sentinel.open("x").close()  # atomic exclusive create
    except FileExistsError:
        while sentinel.exists():
            time.sleep(2)
        return

    try:
        url = f"https://github.com/{repo}.git"
        log.info("Mirror-cloning %s → %s", url, mirror_path)
        subprocess.run(
            ["git", "clone", "--mirror", url, str(mirror_path)],
            check=True, capture_output=True,
        )
    finally:
        sentinel.unlink(missing_ok=True)


def _ensure_repo(repos_dir: Path, repo: str, instance_id: str, base_commit: str) -> Path:
    """Return path to a checkout of *repo* at *base_commit* inside *repos_dir*.

    Uses a shared mirror clone (one network download per repo) then a fast
    local clone per instance.  Safe for parallel workers.
    """
    instance_path = repos_dir / instance_id
    if instance_path.exists():
        return instance_path

    mirror_name = repo.replace("/", "__") + ".git"
    mirror_path = repos_dir / "_mirrors" / mirror_name
    _ensure_mirror(mirror_path, repo)

    log.info("Local-cloning %s@%s → %s", repo, base_commit[:8], instance_path)
    subprocess.run(
        ["git", "clone", str(mirror_path), str(instance_path)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(instance_path), "checkout", base_commit],
        check=True, capture_output=True,
    )
    return instance_path

from data.swebench_loader import SWEBenchLoader
from data.sandbox import SandboxRunner
from data.llm_client import LLMClient
from data.patch_generator import PatchGenerator
from data.training_builder import TrainingSetBuilder
from data.zeroshot_builder import ZeroshotSetBuilder
from data.pair_types import TrainingPair


# ---------------------------------------------------------------------------
# Worker (runs in subprocess for isolation)
# ---------------------------------------------------------------------------

def _process_one(args_tuple) -> dict | None:
    """Process a single DataSample in a worker process."""
    sample_dict, mode, llm_kwargs, sandbox_kwargs, repos_dir_str = args_tuple

    from data.data_loader import DataSample
    from data.llm_client import LLMClient
    from data.patch_generator import PatchGenerator
    from data.sandbox import SandboxRunner
    from data.training_builder import TrainingSetBuilder
    from data.zeroshot_builder import ZeroshotSetBuilder

    sample = DataSample(**sample_dict)

    # Clone repo on-demand if repos_dir is specified.
    if repos_dir_str:
        repos_dir = Path(repos_dir_str)
        repo = sample.metadata.get("repo", "")
        base_commit = sample.metadata.get("base_commit", "")
        if repo and base_commit:
            try:
                repo_path = _ensure_repo(repos_dir, repo, sample.sample_id, base_commit)
                sample.metadata["repo_path"] = str(repo_path)
            except subprocess.CalledProcessError as exc:
                log.error("Failed to clone repo for %s: %s", sample.sample_id, exc.stderr)
                return None

    llm = LLMClient(**llm_kwargs)
    generator = PatchGenerator(llm)
    sandbox = SandboxRunner(**sandbox_kwargs)

    if mode == "training":
        builder = TrainingSetBuilder(generator, sandbox)
    else:
        builder = ZeroshotSetBuilder(generator, sandbox)

    pair = builder.build(sample)
    if pair is None:
        return None

    result = dataclasses.asdict(pair)
    result["sample_id"] = sample.sample_id  # for resume tracking
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TrainingPair data from SWE-bench")
    parser.add_argument(
        "--mode", choices=["training", "zeroshot"], default="training",
        help="training: gold(+) vs unconstrained(-); zeroshot: controlled_llm(+) vs unconstrained(-)",
    )
    parser.add_argument("--input", default="data/raw/swebench_instances.jsonl")
    parser.add_argument("--output", default=None,
                        help="Default: data/cache/training_pairs.jsonl or zeroshot_pairs.jsonl")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--repo-filter", default=None,
                        help="Only process instances from this repo, e.g. 'sympy/sympy'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--filter-prefix", default="")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    _default_repos_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "repos")
    parser.add_argument(
        "--repos-dir", default=_default_repos_dir,
        help="Directory for on-demand repo clones (default: $TMPDIR/repos). "
             "Pass empty string to use repo_path from the JSONL file.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/cache/{args.mode}_pairs.jsonl"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-processed instance IDs.
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    sid = row.get("sample_id")
                    if sid:
                        done_ids.add(sid)
                except json.JSONDecodeError:
                    pass
        log.info("Resuming: %d pairs already written", len(done_ids))

    loader = SWEBenchLoader(args.input)
    samples = list(loader)
    if args.repo_filter:
        samples = [s for s in samples if s.metadata.get("repo") == args.repo_filter]
        log.info("Filtered to %d instances from %s", len(samples), args.repo_filter)
    if args.max:
        samples = samples[:args.max]
    log.info("Loaded %d instances from %s", len(samples), args.input)

    pending = [s for s in samples if s.sample_id not in done_ids]
    if len(pending) < len(samples):
        log.info("Skipping %d already-done, %d pending", len(samples) - len(pending), len(pending))

    llm_kwargs = {
        "provider": args.llm_provider,
        "model": args.llm_model,
        "api_base": args.llm_api_base,
        "max_tokens": args.llm_max_tokens,
    }
    sandbox_kwargs = {
        "timeout": args.timeout,
        "filter_prefix": args.filter_prefix,
    }

    repos_dir_str = args.repos_dir or ""
    if repos_dir_str:
        Path(repos_dir_str).mkdir(parents=True, exist_ok=True)
        log.info("Repos dir: %s", repos_dir_str)

    work_items = [
        (dataclasses.asdict(s), args.mode, llm_kwargs, sandbox_kwargs, repos_dir_str)
        for s in pending
    ]

    ok = skip = fail = 0

    with out_path.open("a") as fh:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, item): item for item in work_items}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    log.error("Worker exception: %s", exc)
                    fail += 1
                    continue

                if result is None:
                    skip += 1
                else:
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()
                    ok += 1
                    log.info("[mode=%s] pair %d  (ok=%d skip=%d fail=%d)",
                             args.mode, ok, ok, skip, fail)

    log.info("Done. Written=%d  Filtered=%d  Errors=%d → %s", ok, skip, fail, out_path)


if __name__ == "__main__":
    main()
