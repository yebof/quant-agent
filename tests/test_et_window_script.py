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


def test_run_if_et_window_serializes_non_intra_modes(tmp_path):
    """The cross-mode session lock must serialize the heavyweight LLM
    sessions (morning/midday/close/evening/earnings_preprocess) so they
    never run concurrently. Use midday vs a held morning lock to verify."""
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
    counter_file = tmp_path / "blocked-midday.txt"

    _write_executable(timeout_bin, "#!/bin/bash\nshift 2\nexec \"$@\"\n")
    _write_executable(
        python_bin,
        "#!/bin/bash\n"
        f"echo fired >> \"{counter_file}\"\n"
        "exit 0\n",
    )

    result = subprocess.run(
        ["bash", str(script), "midday"],
        env=os.environ | {
            "PROJECT_ROOT_OVERRIDE": str(project_root),
            "PYTHON_OVERRIDE": str(python_bin),
            "TIMEOUT_OVERRIDE": str(timeout_bin),
            "LAST_RUN_DIR_OVERRIDE": str(last_run_dir),
            "ET_DOW_OVERRIDE": "1",
            "ET_HOUR_OVERRIDE": "13",
            "ET_MIN_OVERRIDE": "00",
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


def test_run_if_et_window_intra_check_bypasses_session_lock(tmp_path):
    """intra_check is the stateless circuit breaker — it MUST fire on every
    30-min tick during 09:30-16:00 ET regardless of what else is running.
    A held lock from a long morning/midday must not block it; otherwise the
    flash-crash protection goes silent during exactly the windows when an
    adverse move would be most damaging. Mirrors its exemption from the
    last-run guard."""
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
    counter_file = tmp_path / "intra-fired.txt"

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

    assert result.returncode == 0, result.stderr
    assert counter_file.exists(), "intra_check must fire even when lock is held"
    assert counter_file.read_text().strip() == "fired"
    # The held lock must remain — intra_check never touches it.
    assert (lock_dir / "owner").read_text() == "morning 2026-04-20 1000 12345"


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


# ============================================================================
# RC5 (2026-07-16): bash-side kill notification + external dead-man ping.
# When `timeout` SIGTERM/SIGKILLs python, the in-process finally-block
# notifier never runs — 13 straight days of morning kills produced zero
# failure pushes. The wrapper must report violent deaths itself, and ping
# HEALTHCHECKS_URL so an EXTERNAL monitor sees liveness.
# ============================================================================

def _base_env(tmp_path, python_body: str) -> dict:
    project_root = tmp_path / "project"
    project_root.mkdir(exist_ok=True)
    (project_root / ".env").write_text("")
    timeout_bin = tmp_path / "timeout"
    python_bin = tmp_path / "fake-python"
    curl_log = tmp_path / "curl.log"
    curl_bin = tmp_path / "curl"
    _write_executable(timeout_bin, "#!/bin/bash\nshift 2\nexec \"$@\"\n")
    _write_executable(python_bin, f"#!/bin/bash\n{python_body}\n")
    _write_executable(curl_bin, f"#!/bin/bash\necho \"$@\" >> {curl_log}\nexit 0\n")
    env = os.environ | {
        "PROJECT_ROOT_OVERRIDE": str(project_root),
        "PYTHON_OVERRIDE": str(python_bin),
        "TIMEOUT_OVERRIDE": str(timeout_bin),
        "LAST_RUN_DIR_OVERRIDE": str(tmp_path / "cache"),
        "ET_DOW_OVERRIDE": "1",
        "ET_HOUR_OVERRIDE": "08",
        "ET_MIN_OVERRIDE": "30",
        "ET_DATE_OVERRIDE": "2026-07-16",
        "NOW_UNIX_OVERRIDE": "1234567890",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",  # fake curl first
    }
    return env


def _run(script_env, mode="earnings_preprocess"):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_if_et_window.sh"
    return subprocess.run(["bash", str(script), mode], env=script_env,
                          capture_output=True, text=True, check=False)


def test_wrapper_notifies_telegram_on_timeout_kill(tmp_path):
    env = _base_env(tmp_path, "exit 124")
    env |= {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "42"}
    result = _run(env)
    assert result.returncode == 124
    log = (tmp_path / "curl.log").read_text()
    assert "sendMessage" in log
    assert "KILLED" in log


def test_wrapper_does_not_duplicate_python_notifier_on_plain_failure(tmp_path):
    """Ordinary non-zero exits: python's own finally-block already pushed —
    a second bash push would train the operator to ignore duplicates."""
    env = _base_env(tmp_path, "exit 1")
    env |= {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "42"}
    result = _run(env)
    assert result.returncode == 1
    log_file = tmp_path / "curl.log"
    assert not log_file.exists() or "sendMessage" not in log_file.read_text()


def test_wrapper_respects_telegram_kill_switch(tmp_path):
    # audit round 2 (#44): the wrapper must honour every spelling python's
    # notifier accepts ("1"/"true"/"yes", case-insensitive) — previously
    # only the literal "1" muted the bash-side KILLED push.
    for i, disabled in enumerate(("1", "true", "YES", " on ")):
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        env = _base_env(sub, "exit 137")
        env |= {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "42",
                "TELEGRAM_DISABLED": disabled}
        _run(env)
        log_file = sub / "curl.log"
        assert not log_file.exists() or "sendMessage" not in log_file.read_text(), (
            f"TELEGRAM_DISABLED={disabled!r} must mute the bash-side push"
        )


def test_wrapper_pings_healthchecks_on_success_and_fail(tmp_path):
    env = _base_env(tmp_path, "exit 0")
    env |= {"HEALTHCHECKS_URL": "https://hc-ping.example/uuid-1"}
    assert _run(env).returncode == 0
    assert "https://hc-ping.example/uuid-1" in (tmp_path / "curl.log").read_text()

    second = tmp_path / "second"
    second.mkdir()
    env2 = _base_env(second, "exit 124")
    env2 |= {"HEALTHCHECKS_URL": "https://hc-ping.example/uuid-1"}
    _run(env2)
    assert "https://hc-ping.example/uuid-1/fail" in (second / "curl.log").read_text()
