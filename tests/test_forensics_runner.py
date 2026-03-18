import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_task_forensics.py"
SPEC = importlib.util.spec_from_file_location("inspect_task_forensics", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_backend_is_bow():
    parser = MODULE.build_parser()
    args = parser.parse_args([])

    assert args.backend == "bow"
