import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_run_if_et_window_only_records_successful_runs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_if_et_window.sh"
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("")

    last_run_dir = tmp_path / "cache"
    timeout_bin = tmp_path / "timeout"
    python_bin = tmp_path / "fake-python"

    _write_executable(
        timeout_bin,
        "#!/bin/bash\n"
        "shift 2\n"
        "exec \"$@\"\n",
    )
    _write_executable(
        python_bin,
        "#!/bin/bash\n"
        "exit 1\n",
    )

    env = os.environ | {
        "PROJECT_ROOT_OVERRIDE": str(project_root),
        "PYTHON_OVERRIDE": str(python_bin),
        "TIMEOUT_OVERRIDE": str(timeout_bin),
        "LAST_RUN_DIR_OVERRIDE": str(last_run_dir),
        "ET_DOW_OVERRIDE": "1",
        "ET_HOUR_OVERRIDE": "08",
        "ET_MIN_OVERRIDE": "30",
        "ET_DATE_OVERRIDE": "2026-04-17",
        "NOW_UNIX_OVERRIDE": "1234567890",
    }

    failed = subprocess.run(
        ["bash", str(script), "earnings_preprocess"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    assert not (last_run_dir / "last-earnings_preprocess").exists()

    _write_executable(
        python_bin,
        "#!/bin/bash\n"
        "exit 0\n",
    )
    succeeded = subprocess.run(
        ["bash", str(script), "earnings_preprocess"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert succeeded.returncode == 0
    assert (last_run_dir / "last-earnings_preprocess").read_text().strip() == "1234567890"
