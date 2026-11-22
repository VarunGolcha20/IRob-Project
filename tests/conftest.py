"""Skip test modules whose heavy dependencies (torch, MuJoCo/dm_control) are not installed,
so `pytest -q tests/` also runs the torch-free tests (test_npdyn.py) on a bare numpy machine."""
import importlib.util

_missing = [m for m in ("torch", "mujoco", "dm_control") if importlib.util.find_spec(m) is None]

collect_ignore = []
if _missing:
    collect_ignore += ["test_analytical.py", "test_data.py", "test_models.py"]
    print(f"[conftest] {', '.join(_missing)} not installed: running the torch-free tests only")
