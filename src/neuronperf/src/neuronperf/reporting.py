# -*- coding: utf-8 -*-

"""
neuronperf.reporting
~~~~~~~~~~~~~~~~~~~~
Provides utilities for producing reports from benchmarking results.
"""

from typing import List

import csv
import itertools
import json
import logging
import time

import numpy as np

from . import __version__


log = logging.getLogger(__name__)

CSV_COLS = [
    "model_name",
    "n_models",
    "workers_per_model",
    "pipeline_size",
    "batch_size",
    "throughput_avg",
    "throughput_peak",
    "latency_ms_p0",
    "latency_ms_p50",
    "latency_ms_p90",
    "latency_ms_p95",
    "latency_ms_p99",
    "latency_ms_p100",
    "cpu_avg_percent",
    "cpu_percent_p50",
    "mem_avg_percent",
    "mem_percent_p50",
    "e2e_avg_ms",
    "infer_avg_ms",
    "total_infs",
    "total_s",
    "performance_level",
    "model_filename",
    "device_type",
    "instance_type",
    "cost_per_1m_inf",
]

PRINT_COLS = [
    "throughput_avg",
    "latency_ms_p50",
    "latency_ms_p99",
    "n_models",
    "pipeline_size",
    "workers_per_model",
    "batch_size",
    "model_filename",
]

REQUIRED_CONFIG_KEYS = [
    "multiprocess",
    "multiinterpreter",
    "device_type",
    "batch_size",
    "model_filename",
    "model_name",
    "n_models",
    "pipeline_size",
]

REQUIRED_RESULTS_KEYS = [
    "workers_per_model",
    "status",
    "timers",
    "n_infs",
    "total_s",
]


def _validate_config(config):
    for required_key in REQUIRED_CONFIG_KEYS:
        if required_key not in config:
            raise ValueError(
                (
                    f"Model config is missing required key '{required_key}'. "
                    "Something probably went wrong during benchmarking. Provided:\n{config}"
                )
            )


def _validate_results(results):
    for required_key in REQUIRED_RESULTS_KEYS:
        if required_key not in results:
            raise ValueError(
                (
                    f"Benchmarking results are missing required key '{required_key}'. "
                    "Something probably went wrong during benchmarking. Provided:\n{results}"
                )
            )


def _get_report_name(model_name: str) -> str:
    return "{}.results-{}".format(model_name, time.strftime("%Y%m%d-%H%M%S"))


def get_report(
    benchmark_results, cost_per_hour: float = None, window_size: int = 1, verbosity: int = 0
) -> dict:
    r"""Get a performance report from benchmarker results.

    :param benchmark_results: Results from a :class:`benchmarking:Benchmarker` object.
    :param float cost_per_hour: The cost / hour for this device.
    :param int window_size: Window size in seconds used to measure throughput.
    :param int verbosity: Controls logging during report generation. Use 0 (default), 1, or 2.
    :returns: A dictionary containing performance information.
    """
    report = {}
    config, results = benchmark_results
    _validate_config(config)
    _validate_results(results)
    try:
        report["NeuronPerf_version"] = __version__

        # copy benchmarker info from config into report
        for k, v in config.items():
            report[k] = v

        # number of intervals is the same across all stats, so we can use this as a proxy
        report["n_stats_intervals"] = len(results["cpu_percents"])

        report["workers_per_model"] = results["workers_per_model"]
        report["status"] = results["status"]

        # timing stats
        report["load_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["load"]), float
        ).mean()
        report["input_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["input"]), float
        ).mean()
        report["warmup_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["warmup"]), float
        ).mean()
        report["env_setup_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["env_setup"]), float
        ).mean()
        report["setup_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["setup"]), float
        ).mean()
        report["preprocess_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["preprocess"]), float
        ).mean()
        report["infer_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["infer"]), float
        ).mean()
        report["postprocess_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["postprocess"]), float
        ).mean()
        report["e2e_avg_ms"] = np.fromiter(
            (t.avg("ms") for t in results["timers"]["e2e"]), float
        ).mean()
        report["worker_avg_s"] = round(
            np.fromiter((t.avg("s") for t in results["timers"]["worker"]), float).mean(), 2
        )
        report["total_infs"] = results["n_infs"] * config["batch_size"]
        report["total_s"] = round(results["total_s"], 2)

        percentiles = [0, 50, 90, 95, 99, 100]

        cpu_percents = np.fromiter(results["cpu_percents"], float)
        if cpu_percents.size > 2:
            cpu_percentiles = np.percentile(cpu_percents[1:-1], percentiles)
            report["cpu_avg_percent"] = cpu_percentiles.mean()
            for i, p in enumerate(percentiles):
                report[f"cpu_percent_p{p}"] = cpu_percentiles[i]

        mem_percents = np.fromiter(results["mem_percents"], float)
        if mem_percents.size > 2:
            mem_percentiles = np.percentile(mem_percents[1:-1], percentiles)
            report["mem_avg_percent"] = mem_percentiles.mean()
            for i, p in enumerate(percentiles):
                report[f"mem_percent_p{p}"] = mem_percentiles[i]

        # latency
        latencies = np.fromiter(
            itertools.chain.from_iterable(t.durations("ms") for t in results["timers"]["e2e"]),
            float,
        )
        latency_percentiles = np.percentile(latencies, percentiles)
        for i, p in enumerate(percentiles):
            report["latency_ms_p{}".format(p)] = latency_percentiles[i]

        # bucketize ending timestamps
        end_timestamps = np.fromiter(
            itertools.chain.from_iterable(t.end_timestamps("s") for t in results["timers"]["e2e"]),
            float,
        )
        bucket_ends = np.floor(end_timestamps / window_size)
        # group timestamps by window and correct for batch size
        _, bucket_counts = np.unique(bucket_ends, return_counts=True)
        bucket_counts *= config["batch_size"]
        # find max and normalize by window size
        report["throughput_peak"] = bucket_counts.max() / window_size
        report["throughput_avg"] = bucket_counts[1:-1].mean() / window_size

        if verbosity > 0:
            report["throughput_hist"] = bucket_counts
        if verbosity > 1:
            report["e2e_durations_ms"] = np.fromiter(
                (t.durations("ms") for t in results["timers"]["e2e"]), float
            )

        # Try to estimte cost / inference
        if cost_per_hour:
            try:
                infs_per_hour = 3600 * report["throughput_avg"]
                report["cost_per_1m_inf"] = cost_per_hour * (1_000_000 / infs_per_hour)
            except:
                # We'll ignore this, as it's caused by a missing field that would have
                # already generated an earlier error log. We should continue producing
                # a report nonetheless.
                pass

        # Truncate floats to 3 places for readability.
        for key, value in report.items():
            if isinstance(value, float):
                report[key] = round(value, 3)

    except:
        log.exception(
            (
                "Failed to produce a report from benchmarking results. "
                "Something probably went wrong during benchmarking."
            )
        )
    return report


def get_reports(results, cost_per_hour: float = None) -> List[dict]:
    r"""
    Summarizes and combines the detailed results from
    ``neuronperf.benchmark``, when run with ``return_timers=True``.
    One report dictionary is produced per model configuration benchmarked.
    The list of reports can be fed directly to other reporting utilities,
    such as ``neuronperf.write_csv``.

    :param results: Benchmarker results.
    :param float cost_per_hour: The cost / hour for this device.
    """
    reports = []
    for idx, (config, result) in enumerate(results):
        try:
            _validate_config(config)
            _validate_results(result)
        except ValueError:
            log.exception(f"Result {idx} is missing required information, skipping.")
            continue
        report = get_report((config, result), cost_per_hour)
        reports.append(report)
    return reports


def print_reports(reports: List[dict], cols=PRINT_COLS, sort_by="throughput_peak", reverse=False):
    r"""Print a subset of report cols to the terminal.

    :param reports: Results from `get_reports`.
    :param cols: The columns in the report to be displayed.
    :param sort_by: Sort the cols by the specified key.
    :param reverse: Sort order.
    """
    if not reports:
        print("No reports were found. Did benchmarking succeed?")
        return
    # Print headers.
    col_width = max(map(lambda col: len(col), cols)) + 1
    row_format = "{{:<{}}}".format(col_width) * len(cols)
    print(row_format.format(*cols))
    # Extract all rows.
    rows = []
    for report in reports:
        row = []
        for col in cols:
            row.append(report[col] if col in report else "N/A")
        rows.append(row)
    # Sort rows by the specified key, if the key exists.
    if sort_by in cols:
        sort_index = cols.index(sort_by)
        rows = sorted(rows, key=lambda row: row[sort_index], reverse=reverse)
    # Print all rows.
    for row in rows:
        print(row_format.format(*row))


def write_csv(reports: List[dict], filename: str = None, cols=CSV_COLS):
    r"""Write a benchmarking report to CSV file.

    :param reports: Results from `get_reports`.
    :param filename: File name to write out. If not provided, generated from model_name in report and current timestamp.
    :param cols: The columns in the report to be kept.
    """
    if not filename:
        filename = "{}.csv".format(_get_report_name(reports[0]["model_name"]))
    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(cols)
            for idx, report in enumerate(reports):
                row = []
                for col in cols:
                    if col in report:
                        row.append(report[col] if report[col] is not None else "N/A")
                    else:
                        log.debug(f"Report {idx} is missing field '{col}'.")
                        row.append("N/A")
                writer.writerow(row)
        return filename
    except OSError:
        log.exception(f"Failed to write '{filename}'. Check that you have write permissions.")


def write_json(reports: List[dict], filename: str = None):
    if not filename:
        filename = "{}.json".format(_get_report_name(reports[0]["model_name"]))
    try:
        with open(filename, "w", encoding="utf-8") as jsonfile:
            json.dump(reports, jsonfile)
        return filename
    except OSError:
        log.exception(
            (
                f"Failed to write '{filename}'. Check that the report "
                "contains data and that you have write permissions."
            )
        )
