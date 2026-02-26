import subprocess
import os
import signal
import tempfile
import time

# Candidate floating-point hardware events and their FLOP multipliers.
_ALL_FP_EVENTS = [
    ('fp_arith_inst_retired.scalar_double',      1),
    ('fp_arith_inst_retired.128b_packed_double',  2),
    ('fp_arith_inst_retired.256b_packed_double',  4),
    ('fp_arith_inst_retired.512b_packed_double',  8),
]


# Small workload used to test whether each perf event is supported.
_FP_PROBE_CMD = ['python3', '-c', 'import numpy as np; a=np.random.rand(500,500); b=np.dot(a,a)']


def _probe_supported_events() -> list[tuple[str, int]]:
    """Return the list of perf FP events available on this machine."""
    supported = []
    for event, multiplier in _ALL_FP_EVENTS:
        try:
            result = subprocess.run(
                ['perf', 'stat', '-e', event, '--'] + _FP_PROBE_CMD,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=15,
            )
            stderr = result.stderr.decode(errors='replace')
            # perf can succeed with return code 0 but still report unsupported/not counted events.
            if result.returncode == 0 and '<not supported>' not in stderr and '<not counted>' not in stderr:
                supported.append((event, multiplier))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Stop probing when perf is unavailable or probing takes too long.
            break
    return supported


_FP_EVENTS: list[tuple[str, int]] = _probe_supported_events()

if _FP_EVENTS:
    _EVENT_NAMES = ','.join(e for e, _ in _FP_EVENTS)
    print(f"[perf] Using FP events: {[e for e, _ in _FP_EVENTS]}")
else:
    _EVENT_NAMES = 'instructions'
    print("[perf] No FP hardware events available (VMware/WSL2?), falling back to 'instructions'")


def _parse_count(content: str, event: str) -> int:
    """Extract the numeric counter value for a given event from perf output."""
    for line in content.splitlines():
        if event in line and '<not' not in line:
            parts = line.strip().split()
            if parts:
                try:
                    return int(parts[0].replace(',', ''))
                except (ValueError, IndexError):
                    pass
    return 0


def _parse_flops(content: str) -> int:
    """Compute FLOPs from available FP counters; otherwise return instruction count."""
    if _FP_EVENTS:
        return sum(_parse_count(content, e) * m for e, m in _FP_EVENTS)
    return _parse_count(content, 'instructions')


class PerfCounter:
    """Wrapper around Linux perf to estimate FLOPs for the current process."""

    def __init__(self):
        self.flop_count = 0
        self.pid = os.getpid()
        self.process = None
        self.output_file = None

    def start(self):
        """Start perf stat in background mode for this process."""
        self.flop_count = 0
        # Temporary output file where perf writes stats on termination.
        self.output_file = tempfile.mktemp(prefix='perf_stat_', suffix='.txt')
        try:
            self.process = subprocess.Popen(
                ['perf', 'stat', '-p', str(self.pid), '-e', _EVENT_NAMES, '-o', self.output_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Wait briefly for perf to initialize and create its output file.
            deadline = time.monotonic() + 2.0
            while not os.path.exists(self.output_file):
                if time.monotonic() > deadline or self.process.poll() is not None:
                    break
                time.sleep(0.01)
        except FileNotFoundError:
            print("Warning: 'perf' not found. Install with: sudo apt-get install linux-perf")

    def stop(self) -> int:
        """Stop perf, parse collected counters, and return estimated FLOPs."""
        if self.process is None:
            return 0

        try:
            # SIGINT makes perf print final statistics before exiting.
            self.process.send_signal(signal.SIGINT)
            self.process.communicate(timeout=5)

            if self.output_file and os.path.exists(self.output_file):
                with open(self.output_file, 'r') as f:
                    content = f.read()
                os.remove(self.output_file)
                self.flop_count = _parse_flops(content)
            else:
                print(f"Warning: perf output not found ({self.output_file})")
        except Exception as e:
            print(f"Error reading perf stats: {e}")

        return self.flop_count

    def get_flops(self) -> int:
        """Return the most recent FLOP estimate."""
        return self.flop_count


class PAPICounter:
    """Compatibility wrapper exposing a PAPI-like counter interface."""

    def __init__(self):
        self.counter = PerfCounter()

    def start(self):
        self.counter.start()

    def stop(self) -> int:
        return self.counter.stop()

    def get_flops(self) -> int:
        return self.counter.get_flops()
