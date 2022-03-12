# -*- coding: utf-8 -*-

"""
neuronperf._timing
~~~~~~~~~~~~~~~~~~~~~~~
Provides utility functions for timing and time unit conversions.
"""

from typing import Any, Callable

import sys
import time
import typing

import numpy as np


time_unit_ratios = {
    'ns': { 'ns': 1, 'us': 1e-3, 'ms': 1e-6, 's': 1e-9 },
    'us': { 'ns': 1e3, 'us': 1, 'ms': 1e-3, 's': 1e-6 },
    'ms': { 'ns': 1e6, 'us': 1e3, 'ms': 1, 's': 1e-3 },
    's': { 'ns': 1e9, 'us': 1e6, 'ms': 1e3, 's': 1 }
}


supported_time_units = time_unit_ratios.keys()


def timestamp_convert(timestamps,
                      input_time_unit: str,
                      output_time_unit: str):
    """Convert timestamp(s) from one time unit to another.

    :param ts: A timestamp or iterable of timestamps.
    :param input_time_unit: A string specifying the input time unit.
    :param output_time_unit: A string specifying the output time unit.
    :returns: A single timestamp or container of timestamps in the output time unit.
    """
    try:
        ratio = time_unit_ratios[input_time_unit][output_time_unit]
    except:
        raise ValueError(f"Can't convert {input_time_unit} to {output_time_unit}")

    return timestamps * ratio


class Timer():
    def __init__(self,
                 timer_fn: Callable[[], Any] = time.perf_counter,
                 timer_unit: str = 's'):
        self.timer_fn = timer_fn
        self.timer_unit = timer_unit
        self._start = []
        self._end = []

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def __delitem__(self, index):
        del self._start[index]
        del self._end[index]

    def __getitem__(self, index):
        # it's possible that start and end won't match if negative indices are used,
        # b/c timer may have started and not stopped yet
        if index < 0: index = index % len(self._end)
        return self._start[index], self._end[index]

    def __iter__(self):
        return zip(self._start, self._end)

    def __len__(self):
        return len(self._end)

    def __str__(self):
        return str(self.timestamps())

    def start(self):
        # If we've already started, consider this a request to restart.
        # This also handles partial timestamps due to a Timer-unrelated error.
        if len(self._start) > len(self._end): self._start.pop()
        self._start.append(self.timer_fn())

    def stop(self):
        # if we haven't started, ignore this
        if 0 == len(self._start): return
        self._end.append(self.timer_fn())

    def next(self):
        """Manually advance the timer to the next timestamp measurement."""
        self.stop()
        self.start()

    def reset(self):
        self._start.clear()
        self._end.clear()

    def insert(self, timestamps: tuple, time_unit: str):
        """Manually insert a timestamp pair. Does not affect ongoing timing.

        :param timestamps: Timestamp pair to insert.
        :param time_unit: The time unit of the incoming timestamps.
        """
        if len(timestamps) != 2 or not time_unit: raise ValueError()
        timestamps = timestamp_convert(np.array(timestamps), time_unit, self.timer_unit)
        self._start.insert(0, timestamps[0])
        self._end.insert(0, timestamps[1])

    def start_timestamps(self, time_unit: str = None):
        if not time_unit: return np.array(self._start)
        return timestamp_convert(np.array(self._start), self.timer_unit, time_unit)

    def end_timestamps(self, time_unit: str = None):
        if not time_unit: return np.array(self._end)
        return timestamp_convert(np.array(self._end), self.timer_unit, time_unit)

    def timestamps(self, time_unit: str = None):
        """Returns a list of pairs of timestamps (start, end).

        :param time_unit: The time unit of the output timestamp(s). `None` will use the timer's native unit.
        """
        starts, ends = self.start_timestamps(time_unit), self.end_timestamps(time_unit)
        return np.stack((starts[:len(ends)], ends), axis=-1)

    def durations(self, time_unit: str = None):
        """Returns an `ndarray` of timestamp deltas, optionally converted into a provided time unit.

        :param time_unit: The time unit of the output timestamp(s). `None` will use the timer's native unit.
        :returns: An `ndarray` of timestamp deltas.
        """
        starts, ends = self.start_timestamps(), self.end_timestamps()
        return timestamp_convert(ends - starts[:len(ends)], self.timer_unit, time_unit)

    def total_duration(self, time_unit: str = None):
        """Returns total duration of all time measurements, optionally converted into a provided time unit.

        :param time_unit: The time unit of the output timestamp(s). `None` will use the timer's native unit.
        :
        """
        starts, ends = self.start_timestamps(), self.end_timestamps()
        total = np.sum(ends - starts[:len(ends)])
        return total if not time_unit else timestamp_convert(total, self.timer_unit, time_unit)

    def avg(self, time_unit: str = None):
        """Returns average duration, optionally converted into a provided time unit.

        :param time_unit: The time unit of the output timestamp(s). `None` will use the timer's native unit.
        :returns: The average duration.
        """
        return self.durations(time_unit).mean() if len(self._end) > 0 else 0
