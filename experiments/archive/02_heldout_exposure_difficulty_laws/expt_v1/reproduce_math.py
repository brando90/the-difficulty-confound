"""Reproduce the published mathematics subset with the frozen analysis and config."""
import argparse
import csv
import json
import math
import platform
import tempfile
from pathlib import Path

import numpy as np
import scipy
from scipy import stats as scipy_stats

import analyze

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public_math"
OUTPUTS = ("actual_math_predictions.csv", "actual_math_comparisons.csv", "actual_math_fits.json")
TOLERANCE = 1e-8


def read(path):
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    return json.loads(path.read_text())


def compare(expected, actual, stats, location="root"):
    """Require identical structure/labels and tightly matching numerical leaves."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), location
        for name in expected:
            compare(expected[name], actual[name], stats, f"{location}.{name}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), location
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, stats, f"{location}[{index}]")
    else:
        try:
            left, right = float(expected), float(actual)
        except (ValueError, TypeError):
            assert expected == actual, (location, expected, actual)
            stats["exact_label_leaves"] += 1
            return
        assert math.isfinite(left) and math.isfinite(right), location
        # Counts and row identities are exact, regardless of numeric tolerance.
        if location.rsplit(".", 1)[-1] in ("n_train", "n_targets", "n_size_blocks"):
            assert expected == actual, location
        difference = abs(left - right)
        assert math.isclose(left, right, rel_tol=TOLERANCE, abs_tol=TOLERANCE), (location, left, right)
        stats["numeric_leaves"] += 1
        stats["max_absolute_difference"] = max(stats["max_absolute_difference"], difference)


def verify_subset():
    public_rows = read(PUBLIC / "source_data/math_grid.csv")
    metadata = read(PUBLIC / "source_data/E5_decontam_verification.json")
    assert len(public_rows) == 24
    assert len({(row["size"], row["cond"]) for row in public_rows}) == 24
    assert {float(row["T"]) for row in public_rows} == {21e6}
    assert len({row["size"] for row in public_rows}) == 4
    conditions = {row["cond"] for row in public_rows}
    assert len(conditions) == 6 and set(metadata) == {"subsets"}
    assert set(metadata["subsets"]) == conditions
    assert all(set(value) == {"n_examples"} for value in metadata["subsets"].values())
    source_names = ("math_grid.csv", "E5_decontam_verification.json")
    subset_hashes = {name: analyze.sha(PUBLIC / "source_data" / name) for name in source_names}
    original_receipt = read(PUBLIC / "results/actual_math_metadata.json")
    provenance_path = PUBLIC / "provenance.json"
    existing = read(provenance_path) if provenance_path.exists() else None
    if existing:
        assert existing["public_subset_sha256"] == subset_hashes
        assert existing["original_metadata_sha256"] == analyze.sha(PUBLIC / "results/actual_math_metadata.json")
        assert existing["original_full_source_sha256"] == original_receipt["sources"]
    private_available = all((ROOT / "source_data" / name).exists() for name in source_names)
    if private_available:
        original_rows = read(ROOT / "source_data/math_grid.csv")
        selected = [row for row in original_rows if float(row["T"]) == 21e6]
        assert public_rows == selected, "Public subset must preserve every selected field and row order"
        private_metadata = read(ROOT / "source_data/E5_decontam_verification.json")["subsets"]
        assert metadata == {"subsets": {condition: {"n_examples": private_metadata[condition]["n_examples"]} for condition in metadata["subsets"]}}
        assert {name: analyze.sha(ROOT / "source_data" / name) for name in source_names} == original_receipt["sources"]
    else:
        assert existing, "Missing original inputs and initial subset-provenance receipt"
    return subset_hashes, original_receipt, private_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipts", action="store_true", help="Save publication provenance and this verification receipt")
    args = parser.parse_args()
    subset_hashes, original_receipt, private_available = verify_subset()
    config = read(ROOT / "config.json")
    unique_conditions = {}
    subset_metadata = read(PUBLIC / "source_data/E5_decontam_verification.json")["subsets"]
    for row in read(PUBLIC / "source_data/math_grid.csv"):
        value = (float(row["alpha"]), math.log(subset_metadata[row["cond"]]["n_examples"]))
        assert row["cond"] not in unique_conditions or unique_conditions[row["cond"]] == value
        unique_conditions[row["cond"]] = value
    alignment, log_breadth = np.array(list(unique_conditions.values())).T
    correlations = {
        "n_unique_conditions": len(unique_conditions),
        "unit": "six designed training conditions, each counted once",
        "pearson": float(scipy_stats.pearsonr(alignment, log_breadth).statistic),
        "spearman": float(scipy_stats.spearmanr(alignment, log_breadth).statistic),
        "kendall_tau_b": float(scipy_stats.kendalltau(alignment, log_breadth).statistic),
        "p_value": "n/a", "interpretation": "descriptive correlation on the fixed six-condition design; no population inference",
    }
    stats = {"numeric_leaves": 0, "exact_label_leaves": 0, "max_absolute_difference": 0.0}
    files = {}
    original_root = analyze.ROOT
    try:
        analyze.ROOT = PUBLIC
        with tempfile.TemporaryDirectory(prefix="reproduce_math_") as temporary:
            output = Path(temporary)
            analyze.actual_data(output, config)
            for name in OUTPUTS:
                expected = read(PUBLIC / "results" / name)
                actual = read(output / name)
                compare(expected, actual, stats, name)
                if name == "actual_math_predictions.csv":
                    identity = lambda row: (row["axis"], row["size"], row["condition"], row["law"], row["n_train"])
                    assert [identity(row) for row in expected] == [identity(row) for row in actual]
                    assert len(expected) == 144 and len(set(map(identity, expected))) == 144
                files[name] = {"rows": len(expected), "published_sha256": analyze.sha(PUBLIC / "results" / name), "reproduced_sha256": analyze.sha(output / name)}
            regenerated_metadata = read(output / "actual_math_metadata.json")
            assert regenerated_metadata["sources"] == subset_hashes
            for field in ("n_observations", "alpha_log_breadth_correlation", "limitation"):
                compare(original_receipt[field], regenerated_metadata[field], stats, f"metadata.{field}")
    finally:
        analyze.ROOT = original_root
    provenance = {
        "publication_scope": "24 used iso-token mathematics observations, required subset counts, summary, predictions and fitted parameters; unused source observations excluded",
        "selection": "math_grid.csv rows with T == 21000000, preserving original order and every selected CSV field; verification metadata reduced to six condition n_examples values",
        "n_observations": 24,
        "public_subset_sha256": subset_hashes,
        "original_full_source_sha256": original_receipt["sources"],
        "original_metadata_sha256": analyze.sha(PUBLIC / "results/actual_math_metadata.json"),
        "original_receipt_note": "results/actual_math_metadata.json is preserved byte-for-byte; its source hashes describe the original full private inputs, not these deliberately reduced public inputs",
        "subset_matches_original_selected_rows": True,
        "subset_counts_match_original": True,
    }
    verification = {
        "status": "PASS", "verified_on": "10-16-2026",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "relative_tolerance": TOLERANCE, "absolute_tolerance": TOLERANCE,
        "analysis_sha256": analyze.sha(ROOT / "analyze.py"), "config_sha256": analyze.sha(ROOT / "config.json"),
        "wrapper_sha256": analyze.sha(Path(__file__)),
        "private_originals_checked_this_run": private_available,
        "row_order_keys_labels_and_counts": "exact",
        "alignment_vs_log_breadth": correlations,
        "comparisons": files, **stats,
    }
    if args.write_receipts:
        if not (PUBLIC / "provenance.json").exists():
            assert private_available, "Initial publication provenance needs original-input comparison"
            analyze.save(PUBLIC / "provenance.json", provenance)
        analyze.save(PUBLIC / "verification.json", verification)
    print(json.dumps(verification, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
