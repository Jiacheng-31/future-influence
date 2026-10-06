#!/usr/bin/env python3
"""Rank by token NLL while matching selective's masked count per sequence."""

from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import orjson
except ImportError:  # pragma: no cover - production scoring environment has orjson
    orjson = None


METHOD = "sequence_equal_count_lowest_nll_mask_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 0/1 score cache that masks the lowest-NLL tokens while "
            "matching a selective score cache's non-positive count per sample."
        )
    )
    parser.add_argument("--nll", required=True, type=Path, help="Raw token-NLL JSONL.")
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Selective score JSONL whose score<=0 count defines each sample's mask budget.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Derived 0/1 mask JSONL.")
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_line(line: bytes) -> dict[str, Any]:
    value = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(value, dict):
        raise TypeError("score row is not a JSON object")
    return value


def dump_line(value: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(value) + b"\n"
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def score_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank score row at {path}:{line_number}")
            yield line_number, load_line(line)


def sequence_equal_nll_mask(
    nll_scores: list[float],
    reference_scores: list[float],
) -> list[float]:
    """Return 0 for the lowest NLLs using reference score<=0 as the exact budget."""
    if len(nll_scores) != len(reference_scores):
        raise ValueError("NLL/reference token-score lengths differ")
    if not nll_scores:
        raise ValueError("token-score sequence is empty")

    nll = np.asarray(nll_scores, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    if not np.isfinite(nll).all() or np.any(nll < 0.0):
        raise ValueError("NLL scores must be finite and non-negative")
    if not np.isfinite(reference).all():
        raise ValueError("reference scores must be finite")

    mask_count = int(np.count_nonzero(reference <= 0.0))
    result = np.ones(nll.size, dtype=np.float32)
    if mask_count == 0:
        return result.tolist()
    if mask_count == nll.size:
        result.fill(0.0)
        return result.tolist()

    # O(T) selection.  Resolve values tied at the kth threshold by response
    # position so the exact count is deterministic across NumPy versions.
    threshold = float(np.partition(nll, mask_count - 1)[mask_count - 1])
    lower = np.flatnonzero(nll < threshold)
    result[lower] = 0.0
    remaining = mask_count - int(lower.size)
    if remaining:
        tied = np.flatnonzero(nll == threshold)
        result[tied[:remaining]] = 0.0
    if int(np.count_nonzero(result == 0.0)) != mask_count:
        raise AssertionError("failed to materialize the exact per-sequence mask count")
    return result.tolist()


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.manifest.json")


def expected_sources(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": METHOD,
        "nll": fingerprint(args.nll),
        "reference": fingerprint(args.reference),
    }


def reusable_output(args: argparse.Namespace, expected: dict[str, Any]) -> bool:
    manifest = manifest_path(args.output)
    if not args.output.is_file() or not manifest.is_file():
        return False
    actual = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        all(actual.get(key) == value for key, value in expected.items())
        and actual.get("output_records") == 100_000
        and actual.get("mask_count_matches_reference") is True
    )


def main() -> None:
    args = parse_args()
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be positive")
    args.nll = args.nll.resolve()
    args.reference = args.reference.resolve()
    args.output = args.output.resolve()
    for path in (args.nll, args.reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = expected_sources(args)
    if reusable_output(args, expected) and not args.overwrite:
        print(f"reusing complete sequence-equal NLL mask: {args.output}", flush=True)
        return
    if (args.output.exists() or manifest_path(args.output).exists()) and not args.overwrite:
        raise ValueError(f"stale or incomplete output exists; use --overwrite: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    records = response_tokens = masked_tokens = 0
    try:
        with temporary.open("wb") as output_handle:
            pairs = zip_longest(score_rows(args.nll), score_rows(args.reference))
            for pair_index, pair in enumerate(pairs):
                nll_item, reference_item = pair
                if nll_item is None or reference_item is None:
                    raise ValueError("NLL/reference score files have different row counts")
                nll_line, nll_row = nll_item
                reference_line, reference_row = reference_item
                sample_id = str(pair_index)
                if str(nll_row.get("sample_id")) != sample_id:
                    raise ValueError(f"NLL sample_id is not dense at line {nll_line}")
                if str(reference_row.get("sample_id")) != sample_id:
                    raise ValueError(f"reference sample_id is not dense at line {reference_line}")
                for field in ("sequence_token_count", "response_token_count"):
                    if int(nll_row.get(field, -1)) != int(reference_row.get(field, -2)):
                        raise ValueError(f"sample_id={sample_id} {field} mismatch")

                nll_scores = nll_row.get("token_scores")
                reference_scores = reference_row.get("token_scores")
                if not isinstance(nll_scores, list) or not isinstance(reference_scores, list):
                    raise TypeError(f"sample_id={sample_id} token_scores is not a list")
                expected_count = int(nll_row["response_token_count"])
                if len(nll_scores) != expected_count or len(reference_scores) != expected_count:
                    raise ValueError(f"sample_id={sample_id} response token count mismatch")

                token_mask = sequence_equal_nll_mask(nll_scores, reference_scores)
                row_masked = sum(value == 0.0 for value in token_mask)
                reference_masked = sum(float(value) <= 0.0 for value in reference_scores)
                if row_masked != reference_masked:
                    raise AssertionError(f"sample_id={sample_id} mask budget mismatch")
                result = dict(nll_row)
                result["token_scores"] = token_mask
                output_handle.write(dump_line(result))
                records += 1
                response_tokens += expected_count
                masked_tokens += row_masked
                if records % args.progress_interval == 0:
                    print(
                        f"progress={records}/100000 masked={masked_tokens}/{response_tokens} "
                        f"({masked_tokens / response_tokens:.3%})",
                        flush=True,
                    )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if records != 100_000:
            raise ValueError(f"output records={records}, expected=100000")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        **expected,
        "score_semantics": "binary_train_mask; 0=ignore, 1=train",
        "mask_budget": "count(reference.token_scores <= 0) independently per sequence",
        "ranking": "mask lowest raw token NLL; ties by earliest response position",
        "output_records": records,
        "response_tokens": response_tokens,
        "masked_tokens": masked_tokens,
        "trained_tokens": response_tokens - masked_tokens,
        "mask_rate": masked_tokens / response_tokens,
        "mask_count_matches_reference": True,
    }
    manifest_file = manifest_path(args.output)
    manifest_temporary = manifest_file.with_name(f".{manifest_file.name}.{os.getpid()}.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temporary, manifest_file)
    print(
        f"wrote {records} masks to {args.output}; "
        f"masked={masked_tokens}/{response_tokens} ({masked_tokens / response_tokens:.3%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
