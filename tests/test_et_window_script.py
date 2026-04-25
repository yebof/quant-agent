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
    assert (
        last_run_dir / "last-earnings_preprocess"
    ).read_text().strip() == "2026-04-17 1234567890"


def test_run_if_et_window_fires_once_per_et_session_date(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_if_et_window.sh"
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("")

    last_run_dir = tmp_path / "cache"
    timeout_bin = tmp_path / "timeout"
    python_bin = tmp_path / "fake-python"
    counter_file = tmp_path / "invocations.txt"

    _write_executable(
        timeout_bin,
        "#!/bin/bash\n"
        "shift 2\n"
        "exec \"$@\"\n",
    )
    _write_executable(
        python_bin,
        "#!/bin/bash\n"
        f"echo run >> \"{counter_file}\"\n"
        "exit 0\n",
    )

    base_env = os.environ | {
        "PROJECT_ROOT_OVERRIDE": str(project_root),
        "PYTHON_OVERRIDE": str(python_bin),
        "TIMEOUT_OVERRIDE": str(timeout_bin),
        "LAST_RUN_DIR_OVERRIDE": str(last_run_dir),
        "ET_DOW_OVERRIDE": "1",
        "ET_DATE_OVERRIDE": "2026-04-17",
    }

    first = subprocess.run(
        ["bash", str(script), "morning"],
        env=base_env | {
            "ET_HOUR_OVERRIDE": "09",
            "ET_MIN_OVERRIDE": "30",
            "NOW_UNIX_OVERRIDE": "1234567890",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0

    second = subprocess.run(
        ["bash", str(script), "morning"],
        env=base_env | {
            "ET_HOUR_OVERRIDE": "11",
            "ET_MIN_OVERRIDE": "00",
            "NOW_UNIX_OVERRIDE": "1234573290",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0
    assert counter_file.read_text().splitlines() == ["run"]


def test_run_if_et_window_intra_check_fires_every_tick(tmp_path):
    """intra_check is a stateless circuit breaker — must fire on every 30-min
    launchd tick inside market hours, ignoring the once-per-day guard used by
    morning / midday / close / evening / earnings_preprocess."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_if_et_window.sh"
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("")

    last_run_dir = tmp_path / "cache"
    timeout_bin = tmp_path / "timeout"
    python_bin = tmp_path / "fake-python"
    counter_file = tmp_path / "intra-invocations.txt"

    _write_executable(
        timeout_bin,
        "#!/bin/bash\n"
        "shift 2\n"
        "exec \"$@\"\n",
    )
    _write_executable(
        python_bin,
        "#!/bin/bash\n"
        f"echo tick >> \"{counter_file}\"\n"
        "exit 0\n",
    )

    base_env = os.environ | {
        "PROJECT_ROOT_OVERRIDE": str(project_root),
        "PYTHON_OVERRIDE": str(python_bin),
        "TIMEOUT_OVERRIDE": str(timeout_bin),
        "LAST_RUN_DIR_OVERRIDE": str(last_run_dir),
        "ET_DOW_OVERRIDE": "1",  # Monday
        "ET_DATE_OVERRIDE": "2026-04-20",
    }

    # Fire four times across the market day. Each must succeed + increment counter.
    for hh, mm, ts in (("09", "30", "1"), ("11", "30", "2"), ("14", "00", "3"), ("15", "45", "4")):
        result = subprocess.run(
            ["bash", str(script), "intra_check"],
            env=base_env | {
                "ET_HOUR_OVERRIDE": hh,
                "ET_MIN_OVERRIDE": mm,
                "NOW_UNIX_OVERRIDE": ts,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{hh}:{mm} should have fired: {result.stderr}"

    assert counter_file.read_text().splitlines() == ["tick", "tick", "tick", "tick"]
    # Critical: intra_check must NOT write a last-run marker (would re-introduce
    # the once-per-day cap it's explicitly exempt from).
    assert not (last_run_dir / "last-intra_check").exists()


def test_run_if_et_window_skips_when_another_session_active(tmp_path):
    """Different launchd plists can overlap; the shared session lock must
    prevent an intra_check from running on top of a long morning session."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_if_et_window.sh"
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("")

    last_run_dir = tmp_path / "cache"
    lock_dir = last_run_dir / "active-session.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "owner").write_text("morning 2026-04-20 1000 12345")

    timeout_bin = tmp_path / "timeout"
    python_bin = tmp_path / "fake-python"
    counter_file = tmp_path / "blocked-intra.txt"

    _write_executable(timeout_bin, "#!/bin/bash\nshift 2\nexec \"$@\"\n")
    _write_executable(
        python_bin,
        "#!/bin/bash\n"
        f"echo fired >> \"{counter_file}\"\n"
        "exit 0\n",
    )

    result = subprocess.run(
        ["bash", str(script), "intra_check"],
        env=os.environ | {
            "PROJECT_ROOT_OVERRIDE": str(project_root),
            "PYTHON_OVERRIDE": str(python_bin),
            "TIMEOUT_OVERRIDE": str(timeout_bin),
            "LAST_RUN_DIR_OVERRIDE": str(last_run_dir),
            "ET_DOW_OVERRIDE": "1",
            "ET_HOUR_OVERRIDE": "09",
            "ET_MIN_OVERRIDE": "30",
            "ET_DATE_OVERRIDE": "2026-04-20",
            "NOW_UNIX_OVERRIDE": "1010",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert not counter_file.exists()
    assert "another quant-agent session is active" in result.stderr


def test_run_if_et_window_intra_check_skips_outside_window(tmp_path):
    """intra_check window is 09:30-16:00 ET. Outside, it must not fire."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_if_et_window.sh"
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("")

    last_run_dir = tmp_path / "cache"
    timeout_bin = tmp_path / "timeout"
    python_bin = tmp_path / "fake-python"
    counter_file = tmp_path / "out-of-window.txt"

    _write_executable(timeout_bin, "#!/bin/bash\nshift 2\nexec \"$@\"\n")
    _write_executable(
        python_bin,
        "#!/bin/bash\n"
        f"echo fired >> \"{counter_file}\"\n"
        "exit 0\n",
    )

    base_env = os.environ | {
        "PROJECT_ROOT_OVERRIDE": str(project_root),
        "PYTHON_OVERRIDE": str(python_bin),
        "TIMEOUT_OVERRIDE": str(timeout_bin),
        "LAST_RUN_DIR_OVERRIDE": str(last_run_dir),
        "ET_DOW_OVERRIDE": "1",
        "ET_DATE_OVERRIDE": "2026-04-20",
        "NOW_UNIX_OVERRIDE": "1",
    }

    # Pre-market 08:00: outside window → skip
    r1 = subprocess.run(
        ["bash", str(script), "intra_check"],
        env=base_env | {"ET_HOUR_OVERRIDE": "08", "ET_MIN_OVERRIDE": "00"},
        capture_output=True, text=True, check=False,
    )
    assert r1.returncode == 0

    # Post-close 16:30: outside window → skip
    r2 = subprocess.run(
        ["bash", str(script), "intra_check"],
        env=base_env | {"ET_HOUR_OVERRIDE": "16", "ET_MIN_OVERRIDE": "30"},
        capture_output=True, text=True, check=False,
    )
    assert r2.returncode == 0

    # Neither tick should have exec'd the fake-python
    assert not counter_file.exists()
