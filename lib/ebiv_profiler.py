"""
EBIV Pipeline Profiler — Release 3.1

Lightweight timing instrumentation for every stage of the EBIV pipeline.
Usage:
    prof = PipelineProfiler(enabled=True)
    with prof.measure("fft_forward"):
        F = rfft2(W)
    ...
    prof.report()          # print summary table
    prof.save_csv(path)    # export raw data
"""

import time
import logging
import numpy as np
from collections import defaultdict


class PipelineProfiler:
    """Accumulates per-stage wall-clock times and reports statistics."""

    def __init__(self, enabled=True, report_every_n=50):
        """
        Parameters
        ----------
        enabled : bool
            If False, all timing is skipped (zero overhead for production runs).
        report_every_n : int
            Auto-print a summary table every N frames processed in the
            streaming loop.  Set to 0 to disable auto-reporting.
        """
        self.enabled = enabled
        self.report_every_n = report_every_n

        # stage_name -> list of durations (seconds)
        self._timings = defaultdict(list)

        # Frame counter (caller increments via .tick())
        self._frame_count = 0

        # For the context manager
        self._current_stage = None
        self._t0 = 0.0

    # ------------------------------------------------------------------
    #  Core API
    # ------------------------------------------------------------------

    class _Timer:
        """Context manager returned by measure()."""
        __slots__ = ('_profiler', '_stage')

        def __init__(self, profiler, stage):
            self._profiler = profiler
            self._stage = stage

        def __enter__(self):
            if self._profiler.enabled:
                self._profiler._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            if self._profiler.enabled:
                dt = time.perf_counter() - self._profiler._t0
                self._profiler._timings[self._stage].append(dt)

    def measure(self, stage_name):
        """Return a context manager that times the enclosed block.

        Example:
            with prof.measure("fft_forward"):
                F = rfft2(W)
        """
        return self._Timer(self, stage_name)

    def record(self, stage_name, dt):
        """Manually record a duration (seconds) for a stage."""
        if self.enabled:
            self._timings[stage_name].append(dt)

    def tick(self):
        """Increment the frame counter.  Call once per main-loop iteration.
        Returns True when it's time to auto-report."""
        if not self.enabled:
            return False
        self._frame_count += 1
        if self.report_every_n > 0 and self._frame_count % self.report_every_n == 0:
            return True
        return False

    # ------------------------------------------------------------------
    #  Reporting
    # ------------------------------------------------------------------

    def _build_table(self):
        """Return a list of (stage, count, mean_ms, std_ms, min_ms, max_ms, pct)."""
        rows = []
        total_mean = 0.0
        for stage, times in self._timings.items():
            arr = np.array(times) * 1000.0  # convert to ms
            rows.append((stage, len(arr), arr.mean(), arr.std(), arr.min(), arr.max()))
            total_mean += arr.mean()

        # Sort by mean time descending (most expensive first)
        rows.sort(key=lambda r: r[2], reverse=True)

        # Add percentage column
        result = []
        for stage, n, mean, std, mn, mx in rows:
            pct = (mean / total_mean * 100) if total_mean > 0 else 0.0
            result.append((stage, n, mean, std, mn, mx, pct))

        return result, total_mean

    def report(self, title="Pipeline Profile"):
        """Print a formatted summary table to the logger."""
        if not self.enabled or not self._timings:
            return

        rows, total_mean = self._build_table()

        lines = []
        lines.append("")
        lines.append(f"={'=' * 80}")
        lines.append(f"  {title}  (frames: {self._frame_count})")
        lines.append(f"{'=' * 80}")
        header = f"  {'Stage':<30s} {'Calls':>7s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'%':>6s}"
        lines.append(header)
        lines.append(f"  {'-' * 78}")

        for stage, n, mean, std, mn, mx, pct in rows:
            lines.append(
                f"  {stage:<30s} {n:>7d} {mean:>7.2f}ms {std:>7.2f}ms "
                f"{mn:>7.2f}ms {mx:>7.2f}ms {pct:>5.1f}%"
            )

        lines.append(f"  {'-' * 78}")
        lines.append(f"  {'TOTAL (sum of means)':<30s} {'':>7s} {total_mean:>7.2f}ms")
        lines.append(f"{'=' * 80}")

        logging.info("\n".join(lines))

    def save_csv(self, filepath):
        """Export raw per-call timings as CSV for offline analysis."""
        if not self.enabled or not self._timings:
            return

        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "call_index", "duration_ms"])
            for stage, times in self._timings.items():
                for i, t in enumerate(times):
                    writer.writerow([stage, i, t * 1000.0])

        logging.info(f"Profiler CSV saved to: {filepath}")

    def reset(self):
        """Clear all accumulated data."""
        self._timings.clear()
        self._frame_count = 0
