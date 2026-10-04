from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import validate_output as validator


class RepresentationOutputValidatorTests(unittest.TestCase):
    def make_output(self, root: Path) -> None:
        array = np.zeros(validator.EXPECTED_SHAPE, dtype=np.float16)
        array_path = root / validator.ARRAY_NAME
        np.save(array_path, array, allow_pickle=False)
        rows = [
            {
                "problem_hash": hashlib.sha256(f"problem-{index:03d}".encode()).hexdigest(),
                "label": index < validator.EXPECTED_POSITIVES,
            }
            for index in range(validator.EXPECTED_ROWS)
        ]
        rows.sort(key=lambda row: row["problem_hash"])
        rows_path = root / validator.ROWS_NAME
        rows_path.write_text(
            json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "array_bytes": array_path.stat().st_size,
            "array_dtype": "float16",
            "array_sha256": validator.sha256(array_path),
            "array_shape": list(validator.EXPECTED_SHAPE),
            "baseline_revision": validator.EXPECTED_BASELINE_REVISION,
            "csv_sha256": validator.EXPECTED_CSV_SHA256,
            "gpu_count": 2,
            "model_id": validator.EXPECTED_MODEL_ID,
            "model_revision": validator.EXPECTED_MODEL_REVISION,
            "prompt_file_sha256": validator.EXPECTED_PROMPT_FILE_SHA256,
            "prompt_text_sha256": validator.EXPECTED_PROMPT_TEXT_SHA256,
            "rows_sha256": validator.sha256(rows_path),
            "schema_version": 1,
            "torch_version": "2.8.0+cu126",
        }
        (root / validator.MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_valid_output_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            self.assertTrue(validator.validate(root)["ok"])

    def test_unexpected_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            (root / "kernel.log").write_text("not read", encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationError, "allowlist"):
                validator.validate(root)

    def test_directory_scan_stops_at_the_fourth_member(self) -> None:
        class FakeRoot:
            def is_symlink(self) -> bool:
                return False

            def is_dir(self) -> bool:
                return True

            def iterdir(self):
                yield Path(validator.ARRAY_NAME)
                yield Path(validator.ROWS_NAME)
                yield Path(validator.MANIFEST_NAME)
                yield Path("unexpected.log")
                raise AssertionError("validator enumerated beyond the fixed bound")

        with self.assertRaisesRegex(validator.ValidationError, "allowlist"):
            validator.validate_root(FakeRoot())  # type: ignore[arg-type]

    def test_symlink_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            rows = root / validator.ROWS_NAME
            target = root / "rows-target.json"
            rows.replace(target)
            rows.symlink_to(target.name)
            with self.assertRaisesRegex(validator.ValidationError, "allowlist|unsafe"):
                validator.validate(root)

    def test_array_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            manifest_path = root / validator.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["array_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationError, "digest"):
                validator.validate(root)

    def test_row_privacy_surface_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            rows_path = root / validator.ROWS_NAME
            rows = json.loads(rows_path.read_text(encoding="utf-8"))
            rows[0]["problem_hash"] = "user@example.com"
            rows_path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationError, "hash"):
                validator.validate(root)

    def test_nonfinite_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            array_path = root / validator.ARRAY_NAME
            array = np.load(array_path, allow_pickle=False)
            array[0, 0, 0] = np.nan
            np.save(array_path, array, allow_pickle=False)
            manifest_path = root / validator.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["array_sha256"] = validator.sha256(array_path)
            manifest["array_bytes"] = array_path.stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationError, "non-finite"):
                validator.validate(root)


if __name__ == "__main__":
    unittest.main()
