import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_vs_psi0.py"
SPEC = importlib.util.spec_from_file_location("run_baseline_vs_psi0", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_output_path_is_backend_specific():
    assert MODULE.default_output_path_for_backend("bow") == Path(
        "evaluation_results_baseline_vs_psi0_bow.json"
    )
    assert MODULE.default_output_path_for_backend("dense") == Path(
        "evaluation_results_baseline_vs_psi0_dense.json"
    )


def test_resolve_json_output_path_preserves_explicit_override():
    explicit = Path("custom-results.json")

    assert MODULE.resolve_json_output_path(explicit, "dense") == explicit
