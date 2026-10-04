#!/usr/bin/env python3
"""Validate the exact private AIMO representation-extraction output."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_NAME = "qwen_prompt_representations_fp16.npy"
ROWS_NAME = "representation_rows.json"
MANIFEST_NAME = "extraction_manifest.json"
REQUIRED_FILES = frozenset({ARRAY_NAME, ROWS_NAME, MANIFEST_NAME})
EXPECTED_SHAPE = (137, 37, 4096)
EXPECTED_ROWS = 137
EXPECTED_POSITIVES = 36
EXPECTED_ARRAY_DATA_BYTES = int(np.prod(EXPECTED_SHAPE)) * 2
EXPECTED_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
EXPECTED_MODEL_REVISION = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
EXPECTED_BASELINE_REVISION = "7e8839966750059b6b1d247a12ab56552c79b342"
EXPECTED_CSV_SHA256 = "07774d862388ae4fcd40af472528c87b5c0a646b56bc378165372cb51c170679"
EXPECTED_PROMPT_FILE_SHA256 = (
    "2455c1a0cb69aef5bd3fc47b7665fdaa1cfcf50b3da6db02d24a2d6e5dc7c234"
)
EXPECTED_PROMPT_TEXT_SHA256 = (
    "72283c58f504d7c3d32d66066fcb06aecc20d14e93136cd6b1589f7f3ce7b002"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_TEXT = re.compile(
    r"(?i)(?:https?://|file://|/users/|/volumes/|/home/|[A-Z]:\\|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|api[_-]?key|password|secret|token)"
)


class ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bounded_bytes(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValidationError(f"unsafe output file: {path.name}") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_bytes:
            raise ValidationError(f"unsafe or oversized output file: {path.name}")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            block = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
        payload = b"".join(chunks)
        if len(payload) != file_stat.st_size:
            raise ValidationError(f"output file changed or exceeded its bound: {path.name}")
        return payload
    finally:
        os.close(descriptor)


def read_json(path: Path, max_bytes: int) -> tuple[Any, str]:
    payload = read_bounded_bytes(path, max_bytes)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON file: {path.name}") from exc
    return value, hashlib.sha256(payload).hexdigest()


def validate_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValidationError("output root is missing or unsafe")
    members: dict[str, Path] = {}
    for index, path in enumerate(root.iterdir()):
        if index >= len(REQUIRED_FILES):
            raise ValidationError("output root exceeds the exact allowlist")
        name = path.name
        if len(name.encode("utf-8")) > 128:
            raise ValidationError("output member name exceeds the fixed bound")
        members[name] = path
    if set(members) != REQUIRED_FILES:
        raise ValidationError("output root members do not match the exact allowlist")
    for path in members.values():
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"unsafe output member: {path.name}")


def validate_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows, rows_digest = read_json(path, 128 * 1024)
    if type(rows) is not list or len(rows) != EXPECTED_ROWS:
        raise ValidationError("row mapping count is invalid")
    hashes: list[str] = []
    positives = 0
    for row in rows:
        if type(row) is not dict or set(row) != {"problem_hash", "label"}:
            raise ValidationError("row mapping schema is invalid")
        problem_hash = row["problem_hash"]
        label = row["label"]
        if type(problem_hash) is not str or not HEX64.fullmatch(problem_hash):
            raise ValidationError("row mapping hash is invalid")
        if type(label) is not bool:
            raise ValidationError("row mapping label is invalid")
        hashes.append(problem_hash)
        positives += int(label)
    if len(set(hashes)) != EXPECTED_ROWS or hashes != sorted(hashes):
        raise ValidationError("row mapping hashes are duplicate or unsorted")
    if positives != EXPECTED_POSITIVES:
        raise ValidationError("row mapping class counts changed")
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    if FORBIDDEN_TEXT.search(serialized):
        raise ValidationError("row mapping contains a forbidden privacy surface")
    return rows, rows_digest


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest, _ = read_json(path, 64 * 1024)
    expected_keys = {
        "array_bytes",
        "array_dtype",
        "array_sha256",
        "array_shape",
        "baseline_revision",
        "csv_sha256",
        "gpu_count",
        "model_id",
        "model_revision",
        "prompt_file_sha256",
        "prompt_text_sha256",
        "rows_sha256",
        "schema_version",
        "torch_version",
    }
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise ValidationError("manifest schema is invalid")
    expected = {
        "array_dtype": "float16",
        "array_shape": list(EXPECTED_SHAPE),
        "baseline_revision": EXPECTED_BASELINE_REVISION,
        "csv_sha256": EXPECTED_CSV_SHA256,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "prompt_file_sha256": EXPECTED_PROMPT_FILE_SHA256,
        "prompt_text_sha256": EXPECTED_PROMPT_TEXT_SHA256,
        "schema_version": 1,
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValidationError(f"manifest identity mismatch: {key}")
    if type(manifest["gpu_count"]) is not int or not 1 <= manifest["gpu_count"] <= 8:
        raise ValidationError("manifest GPU count is invalid")
    if (
        type(manifest["torch_version"]) is not str
        or not 1 <= len(manifest["torch_version"]) <= 64
        or FORBIDDEN_TEXT.search(manifest["torch_version"])
    ):
        raise ValidationError("manifest torch version is invalid")
    for key in ("array_sha256", "rows_sha256"):
        if type(manifest[key]) is not str or not HEX64.fullmatch(manifest[key]):
            raise ValidationError(f"manifest digest is invalid: {key}")
    return manifest


def validate_array(path: Path, manifest: dict[str, Any]) -> None:
    payload = read_bounded_bytes(path, EXPECTED_ARRAY_DATA_BYTES + 4096)
    size = len(payload)
    if type(manifest["array_bytes"]) is not int or manifest["array_bytes"] != size:
        raise ValidationError("array byte count does not match manifest")
    if not EXPECTED_ARRAY_DATA_BYTES < size <= EXPECTED_ARRAY_DATA_BYTES + 4096:
        raise ValidationError("array file size is outside the NPY header bound")
    if hashlib.sha256(payload).hexdigest() != manifest["array_sha256"]:
        raise ValidationError("array digest does not match manifest")
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValidationError("array NPY version is not the pinned format")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        if (
            shape != EXPECTED_SHAPE
            or fortran_order
            or dtype != np.dtype(np.float16)
            or len(payload) - stream.tell() != EXPECTED_ARRAY_DATA_BYTES
        ):
            raise ValidationError("array header or data length is invalid")
        stream.seek(0)
        array = np.load(stream, allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ValidationError("array file is not a safe NPY") from exc
    if array.shape != EXPECTED_SHAPE or array.dtype != np.float16:
        raise ValidationError("array shape or dtype is invalid")
    for start in range(0, EXPECTED_ROWS, 8):
        if not np.isfinite(array[start : start + 8]).all():
            raise ValidationError("array contains non-finite values")


def validate(root: Path) -> dict[str, Any]:
    if root.is_symlink():
        raise ValidationError("output root symlink is forbidden")
    root = root.resolve(strict=True)
    validate_root(root)
    _, rows_digest = validate_rows(root / ROWS_NAME)
    manifest = validate_manifest(root / MANIFEST_NAME)
    if rows_digest != manifest["rows_sha256"]:
        raise ValidationError("row mapping digest does not match manifest")
    validate_array(root / ARRAY_NAME, manifest)
    return {
        "array_sha256": manifest["array_sha256"],
        "gpu_count": manifest["gpu_count"],
        "ok": True,
        "rows": EXPECTED_ROWS,
        "shape": list(EXPECTED_SHAPE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.output)
    except (OSError, ValidationError) as exc:
        raise SystemExit(json.dumps({"error": str(exc), "ok": False})) from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
