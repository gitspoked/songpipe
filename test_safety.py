import os
import sys
import time
from pathlib import Path

import pytest
from songpipe import safety
from songpipe.safety import (
    parse_vm_stat, estimate_peak_gb, safe_duration, preflight,
    should_kill, big_hogs, SafetyError, PreflightResult,
)

# A trimmed vm_stat fixture, page size 16384 bytes.
# free=10000, inactive=1000000, speculative=5000, purgeable=40000
#   -> reclaimable pages = 1,055,000 ; *16384 /1e9 = 17.28 GB
# active and wired are large but MUST be excluded.
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    10000.
Pages active:                                1800000.
Pages inactive:                              1000000.
Pages speculative:                              5000.
Pages throttled:                                   0.
Pages wired down:                             236997.
Pages purgeable:                               40000.
"""

def test_parse_vm_stat_sums_reclaimable_only():
    gb = parse_vm_stat(VM_STAT)
    assert abs(gb - 17.28) < 0.05

def test_parse_vm_stat_excludes_active_and_wired():
    # If active(1.8M) or wired were counted, we'd be way over 30 GB.
    assert parse_vm_stat(VM_STAT) < 20.0

def test_estimate_peak_is_base_at_zero():
    assert estimate_peak_gb(0) == safety.ACE_BASE_GB

def test_estimate_peak_monotonic():
    assert estimate_peak_gb(180) > estimate_peak_gb(25) > estimate_peak_gb(0)

def test_safe_duration_fits_budget():
    avail = 40.0
    d = safe_duration(avail)
    assert estimate_peak_gb(d) <= safety.SAFETY_FRACTION * avail + 1e-6

def test_preflight_grants_when_it_fits():
    r = preflight(25, available_gb=60.0)
    assert isinstance(r, PreflightResult)
    assert r.granted == 25 and not r.clamped and not r.forced

def test_preflight_clamps_when_over_budget():
    r = preflight(180, available_gb=20.0)
    assert r.clamped and r.granted < 180
    assert estimate_peak_gb(r.granted) <= r.budget_gb + 1e-6

def test_preflight_force_overrides_without_clamping():
    r = preflight(180, available_gb=20.0, force=True)
    assert r.granted == 180 and r.forced and not r.clamped

def test_preflight_raises_when_nothing_safe_fits():
    with pytest.raises(SafetyError):
        preflight(180, available_gb=5.0)  # budget < base load, even MIN won't fit

def test_should_kill_below_floor():
    assert should_kill(5.0, 6.0) is True

def test_should_kill_at_or_above_floor():
    assert should_kill(6.0, 6.0) is False
    assert should_kill(12.0, 6.0) is False

class NoopWatchdog:
    """Hermetic stand-in for MemoryWatchdog: never reads live RAM, never kills.

    Tests that are NOT about watchdog behavior must use this, or they depend on
    the machine's memory at test time (the real watchdog's immediate first check
    SIGKILLs the child when available RAM is below the 6 GB floor).
    """
    def __init__(self, pid, **kw):
        self.pid = pid
        self.triggered = False
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_run_guarded_returns_output_on_success():
    res = safety.run_guarded([sys.executable, "-c", "print('ok')"],
                             watchdog_factory=NoopWatchdog)
    assert res.returncode == 0 and res.stdout.strip() == "ok"
    assert not res.killed_by_watchdog

def test_run_guarded_flags_watchdog_kill():
    # Inject a fake watchdog that kills the child the moment it starts, so we can
    # prove run_guarded reports a memory kill without actually exhausting RAM.
    import os, signal
    class InstantKiller:
        def __init__(self, pid, **kw):
            self.pid = pid; self.triggered = False
        def __enter__(self):
            self.triggered = True
            os.kill(self.pid, signal.SIGKILL)
            return self
        def __exit__(self, *a):
            return False
    res = safety.run_guarded([sys.executable, "-c", "import time; time.sleep(10)"],
                             watchdog_factory=InstantKiller)
    assert res.killed_by_watchdog

# ── #1  Watchdog must not fail open ──────────────────────────────────────────

def test_watchdog_survives_transient_read_error(monkeypatch):
    """available_ram_gb raises once then returns safe; thread must stay alive and not trigger."""
    call_count = [0]

    def flaky_ram():
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("vm_stat transient error")
        return 100.0  # way above any floor — must never trigger

    monkeypatch.setattr(safety, "available_ram_gb", flaky_ram)
    wd = safety.MemoryWatchdog(os.getpid(), floor_gb=0.001, interval_s=0.05)
    with wd:
        time.sleep(0.25)  # let the watchdog poll a few times

    assert not wd.triggered, "watchdog must not trigger on a transient read error"
    # At least one poll after the first error must have succeeded
    assert call_count[0] >= 2, "watchdog must have continued polling after the error"


# ── #3  preflight must not mutate module globals ──────────────────────────────

def test_preflight_does_not_mutate_global_safety_fraction():
    original = safety.SAFETY_FRACTION
    # force=True bypasses the budget-vs-estimate check so we get a result even
    # at 0.3 fraction; the key assertions are about global state and budget_gb.
    r = preflight(30, available_gb=20.0, safety_fraction=0.3, force=True)
    assert safety.SAFETY_FRACTION == original, (
        "preflight must not rebind the module-level SAFETY_FRACTION"
    )
    # budget_gb must reflect 0.3, not the original fraction
    assert abs(r.budget_gb - 6.0) < 1e-9, (
        f"expected budget_gb=6.0 (0.3*20), got {r.budget_gb}"
    )


# ── #4  run_guarded false-positive kill ──────────────────────────────────────

def test_run_guarded_no_false_positive_when_triggered_after_success():
    """Watchdog sets triggered=True AFTER the process exits cleanly; must report False."""
    import signal

    class LateTriggerer:
        """Marks triggered=True but never actually kills; the child exits cleanly first."""
        def __init__(self, pid, **kw):
            self.pid = pid
            self.triggered = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            # Simulate watchdog waking after the child already finished
            self.triggered = True
            return False

    res = safety.run_guarded(
        [sys.executable, "-c", "print('done')"],
        watchdog_factory=LateTriggerer,
    )
    assert res.returncode == 0
    assert not res.killed_by_watchdog, (
        "triggered=True after a clean exit must not report killed_by_watchdog"
    )


def test_big_hogs_parses_ps_above_threshold():
    ps = ("  RSS COMM\n"
          "12079104 /Applications/OrbStack.app/Contents/Frameworks/OrbStack\n"
          " 838860 claude\n"
          " 100000 Terminal\n")
    hogs = big_hogs(threshold_gb=4.0, ps_output=ps)
    names = [h[0] for h in hogs]
    assert any("OrbStack" in n for n in names)
    assert all(gb >= 4.0 for _, gb in hogs)
    assert not any("claude" in n for n in names)


# ── run_guarded: timeout + cwd (watchguards-over-tools spec) ─────────────────

def test_run_guarded_timeout_kills_child():
    t0 = time.time()
    res = safety.run_guarded([sys.executable, "-c", "import time; time.sleep(30)"],
                             timeout_s=0.5, watchdog_factory=NoopWatchdog)
    assert time.time() - t0 < 10          # did not wait out the sleep
    assert res.timed_out is True
    assert res.returncode != 0
    assert res.killed_by_watchdog is False

def test_run_guarded_no_timeout_flag_on_success():
    res = safety.run_guarded([sys.executable, "-c", "print('ok')"],
                             timeout_s=30, watchdog_factory=NoopWatchdog)
    assert res.timed_out is False and res.returncode == 0 and "ok" in res.stdout

def test_run_guarded_runs_in_cwd(tmp_path):
    res = safety.run_guarded([sys.executable, "-c", "import os; print(os.getcwd())"],
                             cwd=str(tmp_path), watchdog_factory=NoopWatchdog)
    assert Path(res.stdout.strip()).resolve() == tmp_path.resolve()

def test_run_guarded_watchdog_kill_with_timeout_armed():
    # Combined-guards attribution contract: a watchdog kill while timeout_s is
    # armed must report killed_by_watchdog=True and timed_out=False (a -9 exit
    # with a pending timeout must not be misread as a timeout).
    import signal
    class InstantKiller:
        def __init__(self, pid, **kw):
            self.pid = pid; self.triggered = False
        def __enter__(self):
            self.triggered = True
            os.kill(self.pid, signal.SIGKILL)
            return self
        def __exit__(self, *a):
            return False
    res = safety.run_guarded([sys.executable, "-c", "import time; time.sleep(10)"],
                             watchdog_factory=InstantKiller, timeout_s=30)
    assert res.killed_by_watchdog is True
    assert res.timed_out is False
