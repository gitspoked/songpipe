"""Memory-safety guard for songpipe.

ACE-Step runs on MLX/Metal, which allocates *wired* (non-pageable) buffers in unified
memory. With swap=0 a too-large render exhausts RAM and the kernel panics/reboots — a
normal OOM-kill never gets its chance. So we PREVENT dangerous jobs (preflight) and KILL
the child if memory still runs low mid-run (watchdog).

The pure decision functions here are unit-tested with fixtures; the side-effecting shell
(reading vm_stat, spawning threads, SIGKILL) is kept thin around them.
"""
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

# --- tunables (conservative; tighten from real run logs) -----------------------
ACE_BASE_GB = 8.0          # fixed cost: DiT turbo + 1.7B LM resident on MLX
ACE_PER_SEC_GB = 0.18      # marginal cost per second of audio (latent length)
SAFETY_FRACTION = 0.55     # only spend this share of available RAM on the estimate
MIN_DURATION_S = 8         # below this it isn't worth running
HARD_MAX_DURATION_S = 240  # dumb absolute ceiling; --force required above
WATCHDOG_FLOOR_GB = 6.0    # kill the ACE child if available RAM drops below this
WATCHDOG_INTERVAL_S = 1.0

_RECLAIMABLE = ("free", "inactive", "speculative", "purgeable")


class SafetyError(RuntimeError):
    """Raised when no duration is safe to run on the current machine."""


@dataclass
class PreflightResult:
    requested: int
    granted: int
    available_gb: float
    est_peak_gb: float
    budget_gb: float
    clamped: bool
    forced: bool


def parse_vm_stat(text: str) -> float:
    """Available RAM in GB = (free+inactive+speculative+purgeable) * page_size.

    'Pages free' alone is ~0 on a healthy Mac; the reclaimable buckets are the real
    available memory. Active and wired pages are deliberately excluded.
    """
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 16384
    pages = 0
    for bucket in _RECLAIMABLE:
        pm = re.search(rf"Pages {bucket}[^:]*:\s+(\d+)", text)
        if pm:
            pages += int(pm.group(1))
    return pages * page / 1e9


def available_ram_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    return parse_vm_stat(out)


def estimate_peak_gb(duration_s: float) -> float:
    return ACE_BASE_GB + ACE_PER_SEC_GB * duration_s


def safe_duration(available_gb: float, fraction: float = None) -> int:
    """Largest whole-second duration whose estimated peak fits the budget."""
    frac = fraction if fraction is not None else SAFETY_FRACTION
    budget = frac * available_gb
    d = (budget - ACE_BASE_GB) / ACE_PER_SEC_GB
    return int(d) if d > 0 else 0


def preflight(duration: int, *, available_gb: float = None, force: bool = False,
              safety_fraction: float = None) -> PreflightResult:
    # #3: use a local fraction — never mutate the module global
    frac = safety_fraction if safety_fraction is not None else SAFETY_FRACTION
    if available_gb is None:
        available_gb = available_ram_gb()
    budget = frac * available_gb
    est = estimate_peak_gb(duration)

    if force:
        return PreflightResult(duration, duration, available_gb, est, budget, False, True)
    if duration <= HARD_MAX_DURATION_S and est <= budget:
        return PreflightResult(duration, duration, available_gb, est, budget, False, False)

    granted = min(safe_duration(available_gb, fraction=frac), HARD_MAX_DURATION_S)
    if granted < MIN_DURATION_S:
        raise SafetyError(
            f"only ~{available_gb:.1f} GB available; even a {MIN_DURATION_S}s render "
            f"(~{estimate_peak_gb(MIN_DURATION_S):.1f} GB) exceeds the "
            f"{frac:.0%} budget ({budget:.1f} GB). Free memory and retry "
            f"(quit big apps), or pass force=True to override at your own risk.")
    return PreflightResult(duration, granted, available_gb,
                           estimate_peak_gb(granted), budget, True, False)


def should_kill(available_gb: float, floor_gb: float) -> bool:
    return available_gb < floor_gb


def big_hogs(threshold_gb: float = 4.0, ps_output: str = None):
    """List (name, gb) of processes holding >= threshold_gb RSS, biggest first."""
    if ps_output is None:
        ps_output = subprocess.run(["ps", "-axo", "rss,comm", "-r"],
                                   capture_output=True, text=True, check=True).stdout
    hogs = []
    for line in ps_output.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        gb = int(parts[0]) / 1048576  # ps RSS is in KiB
        if gb >= threshold_gb:
            hogs.append((parts[1], gb))
    return sorted(hogs, key=lambda h: -h[1])


@dataclass
class GuardedResult:
    returncode: int
    stdout: str
    stderr: str
    killed_by_watchdog: bool
    timed_out: bool = False


def run_guarded(cmd, *, floor_gb: float = WATCHDOG_FLOOR_GB, watchdog_factory=None,
                timeout_s: float = None, cwd: str = None):
    """Run `cmd` as a subprocess under a memory watchdog, with an optional hard timeout.

    The watchdog SIGKILLs the child if available RAM drops below floor_gb; the timeout
    kills it if it runs longer than timeout_s. Returns a GuardedResult with
    `killed_by_watchdog` / `timed_out` telling which guard (if any) fired.
    """
    if watchdog_factory is None:
        watchdog_factory = MemoryWatchdog
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=cwd)
    timed_out = False
    with watchdog_factory(proc.pid, floor_gb=floor_gb) as wd:
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            out, err = proc.communicate()
    # #4: a real SIGKILL yields returncode -9; a clean exit yields 0.
    # Gate on returncode to avoid false-positive when the watchdog wakes after a clean finish.
    killed_by_watchdog = bool(getattr(wd, "triggered", False)) and proc.returncode != 0
    return GuardedResult(proc.returncode, out or "", err or "", killed_by_watchdog, timed_out)


class MemoryWatchdog:
    """Background thread that SIGKILLs `pid` if available RAM drops below floor_gb.

    Last line of defense before a kernel panic: kill the ACE subprocess a moment early.
    Use as a context manager around the subprocess run.
    """
    def __init__(self, pid: int, floor_gb: float = WATCHDOG_FLOOR_GB,
                 interval_s: float = WATCHDOG_INTERVAL_S):
        self.pid = pid
        self.floor_gb = floor_gb
        self.interval_s = interval_s
        self.triggered = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        # #6: immediate check before first interval wait; #1: guard against transient errors
        try:
            if should_kill(available_ram_gb(), self.floor_gb):
                self.triggered = True
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
        except Exception:
            pass

        while not self._stop.wait(self.interval_s):
            try:  # #1: transient read errors skip this tick; they do NOT kill the thread
                if should_kill(available_ram_gb(), self.floor_gb):
                    self.triggered = True
                    try:
                        os.kill(self.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
            except Exception:
                continue

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2 * self.interval_s)
        return False
