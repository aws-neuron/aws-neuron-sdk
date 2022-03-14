# -*- coding: utf-8 -*-

"""
neuronperf.benchmarking
~~~~~~~~~~~~~~~~~~~~~~~
Provides utility functions and classes that underlie the framework benchmarkers.
"""

from typing import Any, Callable, Dict, List, Union

import collections
import concurrent
import concurrent.futures
import copy
import functools
import logging
import multiprocessing
import os
import psutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import dill


from . import model_index
from .compile_constants import NEURONCORE_PIPELINE_CORES, FAST_MATH, FAST_MATH_OPTIONS
from .reporting import get_reports
from .scripts import run_benchmark_file
from .timing import Timer


log = logging.getLogger(__name__)

# Wrapper for sending back subprocess failure info. Needs to be at top level for pickle.
BenchmarkerErrorWrapper = collections.namedtuple("BenchmarkerErrorWrapper", "trace")

ERROR = "error"
SUPPORTED_DEVICE_TYPES = ["neuron", "cpu", "cuda", "gpu"]  # TODO: "tpu"]
BENCHMARK_SECS = 120


class Benchmarker(threading.Thread):
    r"""
    :class:`benchmarking:Benchmarker` benchmarks a single model.

    This class is a `threading.Thread`. Call `start` to launch a non-blocking
    benchmarking thread. Calling `stop` will end the benchmarking and block
    until all subroutines complete.

    An object of this class may be serialized and sent to multiple subprocesses
    for parallel use. After benchmarking, results can be obtained with
    `results`.
    """

    def __init__(
        self,
        id: int,
        device_id: int,
        load_fn: Callable[[str], Any],
        model_filename: str,
        inputs,
        workers_per_model: int,
        env_setup_fn: Callable[[int, Dict, Any], None] = None,
        setup_fn: Callable[[int, Dict, Any], None] = None,
        preprocess_fn: Callable[[Any], Any] = None,
        postprocess_fn: Callable[[Any], Any] = None,
        dataset_loader_fn: Callable[[Any, int], Any] = None,
        model_class_name: str = None,
        model_class_file: str = None,
    ):
        super().__init__()

        self.id = id
        self.device_id = device_id
        self.load_fn = load_fn
        self.model_filename = model_filename
        self.inputs = inputs
        self.input_iter = None  # Prepared in setup()
        self.input_lock = threading.Lock()
        self.workers_per_model = workers_per_model
        self.env_setup_fn = env_setup_fn
        self.setup_fn = setup_fn
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.dataset_loader_fn = dataset_loader_fn
        self.model_class_name = model_class_name
        self.model_class_file = model_class_file

        # Mutable internal state.
        self.model = None
        self.benchmark_timer = Timer()
        self.env_setup_timer = Timer()
        self.setup_timer = Timer()
        self.load_timer = Timer()
        self.warmup_timer = Timer()
        self.input_timer = Timer()
        self.preprocess_timers = [Timer() for _ in range(workers_per_model)]
        self.infer_timers = [Timer() for _ in range(workers_per_model)]
        self.postprocess_timers = [Timer() for _ in range(workers_per_model)]
        self.e2e_timers = [Timer() for _ in range(workers_per_model)]
        self.worker_timers = [Timer() for _ in range(workers_per_model)]
        self.n_infs = [0] * workers_per_model
        self.process_id = 0  # set at launch time
        self.benchmarking = False
        self.benchmarking_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.status = "ready"
        self.error = None

    def _status(self, status, error=None):
        """Update internal status, unless a previous error has occurred."""
        with self.status_lock:
            if self.status == ERROR:
                return
            self.status = status
            if error:
                self.error = error

    def next_input(self):
        self.input_lock.acquire()
        self.input_timer.start()
        try:
            return next(self.input_iter)
        finally:
            self.input_timer.stop()
            self.input_lock.release()

    def prepare_inputs(self):
        """Prepares input iterator; runs an optional custom setup function."""
        if self.dataset_loader_fn:

            def input_iter():
                dataset_loader = self.dataset_loader_fn(self.inputs, self.workers_per_model)
                while True:
                    inputs = next(dataset_loader)
                    yield inputs if isinstance(inputs, tuple) else (inputs,)

            self.input_iter = input_iter()
        else:

            def input_iter():
                inputs = self.inputs if isinstance(self.inputs, tuple) else (self.inputs,)
                while True:
                    yield inputs

            self.input_iter = input_iter()

    def load(self):
        """Loads the model that will be used for benchmarking."""
        with self.load_timer:
            self.model = self.load_fn(self.model_filename, device_id=self.device_id)

    def warmup(self):
        """Warmup the model with a single e2e inference."""
        with self.warmup_timer:
            inputs = self.next_input()
            if self.preprocess_fn:
                inputs = self.preprocess_fn(*inputs)
            outputs = self.model(*inputs if isinstance(inputs, tuple) else inputs)
            if self.postprocess_fn:
                self.postprocess_fn(outputs)
        self.n_infs[0] += 1  # track warmup infs in worker 0

    def setup(self):
        """Perform all setup work prior to benchmarking."""
        self.prepare_inputs()

        if self.env_setup_fn:
            with self.env_setup_timer:
                self.env_setup_fn()

        self.load()

        if self.setup_fn:
            with self.setup_timer:
                self.setup_fn(self.model)

        self.warmup()

    def infer(self, worker_id) -> tuple:
        """Execute a single inference."""
        with self.e2e_timers[worker_id]:
            inputs = self.next_input()
            if self.preprocess_fn:
                with self.preprocess_timers[worker_id]:
                    inputs = self.preprocess_fn(*inputs)
            with self.infer_timers[worker_id]:
                outputs = self.model(*inputs if isinstance(inputs, tuple) else inputs)
            if self.postprocess_fn:
                with self.postprocess_timers[worker_id]:
                    outputs = self.postprocess_fn(outputs)
        return outputs

    def worker_thread(self, worker_id):
        """A single worker thread that runs inference until signalled to stop."""
        n_infs = 0
        try:
            log.debug(f"Benchmarker {self.id}, Worker {worker_id} started.")
            with self.worker_timers[worker_id]:
                while self.benchmarking and self.status != ERROR:
                    self.infer(worker_id)
                    n_infs += 1
            if self.status == ERROR:
                log.debug(
                    f"Benchmarker {self.id}, Worker {worker_id} stopped early due to an error after {n_infs} inferences."
                )
        except StopIteration:
            pass
        except:
            trace = "".join(traceback.format_exception(*sys.exc_info()))
            log.error(
                f"Benchmarker {self.id}, Worker {worker_id} encountered an error during benchmarking:\n{trace}"
            )
            self._status(ERROR, BenchmarkerErrorWrapper(trace))
        finally:
            self.n_infs[worker_id] += n_infs
            log.debug(
                f"Benchmarker {self.id}, Worker {worker_id} finished after {self.n_infs[worker_id]} inferences."
            )

    def run(self):
        with self.benchmarking_lock:
            if self.benchmarking:
                raise RuntimeError(
                    f"Benchmarker {self.id} can't start because it is already running."
                )
            self.benchmarking = True
            self._status("running")

        # Set our process id, now that we are launched.
        self.process_id = os.getpid()

        # Launch all workers and begin benchmarking.
        # If any individual worker reports an error, self.status will reflect
        # that after this method.
        with self.benchmark_timer:
            try:
                self.setup()
            except:
                trace = "".join(traceback.format_exception(*sys.exc_info()))
                log.error(f"Benchmarker {self.id} encountered an error during prep:\n{trace}")
                self._status(ERROR, BenchmarkerErrorWrapper(trace))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_per_model) as exe:
                    for worker_id in range(self.workers_per_model):
                        exe.submit(self.worker_thread, worker_id)

        # There are three ways to reach the next section:
        # 1. We ran out of benchmarking examples in a provided dataset (graceful quit on StopIteration).
        # 2. We were asked to stop().
        # 3. We encountered an error.

        # In cases 1 and 3, we can acquire the lock, update our state if necessary, and quit.
        # In case 2, we already hold the lock, so we can skip this section and let stop() handle cleanup.
        if self.benchmarking_lock.acquire(blocking=False):
            try:
                self.benchmarking = False
                self._status("finished")
            finally:
                self.benchmarking_lock.release()

    def stop(self):
        # Setting self.benchmarking = False triggers workers to terminate gracefully.
        # We must hold the benchmarking_lock until the thread has joined to ensure
        # consistent use of the self.benchmarking flag.
        with self.benchmarking_lock:
            if not self.benchmarking:
                return
            self._status("stopping")
            self.benchmarking = False
            self.join()
            self._status("finished")

    def results(self) -> dict:
        with self.benchmarking_lock:
            if self.benchmarking:
                raise RuntimeError("Cannot produce results until benchmarking has completed.")
            return {
                "id": self.id,
                "device_id": self.device_id,
                "workers_per_model": self.workers_per_model,
                "n_infs": sum(self.n_infs),
                "status": self.status,
                "process_id": self.process_id,
                "total_s": self.benchmark_timer.total_duration("s"),
                "timers": {
                    "env_setup": [self.env_setup_timer],
                    "setup": [self.setup_timer],
                    "load": [self.load_timer],
                    "input": [self.input_timer],
                    "warmup": [self.warmup_timer],
                    "preprocess": self.preprocess_timers,
                    "infer": self.infer_timers,
                    "postprocess": self.postprocess_timers,
                    "e2e": self.e2e_timers,
                    "worker": self.worker_timers,
                },
            }


class StatsThread(threading.Thread):
    """A thread to collect some system metrics duirng benchmarking."""

    def __init__(self, interval: float):
        super().__init__()
        self.interval = interval  # interval (in seconds) to collect metrics
        self.cpu_percents = []
        self.mem_percents = []
        self.running = True

    def run(self):
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=self.interval, percpu=False)
            mem_percent = psutil.virtual_memory()[2]
            self.cpu_percents.append(cpu_percent)
            self.mem_percents.append(mem_percent)

    def join(self, **kwargs):
        self.running = False
        super().join(**kwargs)


def _combine_results(results: List[dict]) -> dict:
    """Combines the results of multiple benchmarkers into a single results structure."""
    combined_results = {}
    for result in results:
        # workers_per_model should be the same across all benchmarkers, so we only need it once.
        combined_results.setdefault("workers_per_model", result["workers_per_model"])
        # If an error occurred anywhere, preserve it.
        combined_results["status"] = (
            result["status"] if combined_results.get("status", "") != ERROR else ERROR
        )
        combined_results["n_infs"] = combined_results.get("n_infs", 0) + result["n_infs"]
        # Keep the longest subprocess duration.
        combined_results["total_s"] = max(combined_results.get("total_s", 0), result["total_s"])
        # Concatenate all timing info.
        timers = combined_results.get("timers", {})
        for k, v in result["timers"].items():
            timer_list = timers.get(k, [])
            timer_list.extend(v)
            timers[k] = timer_list
        combined_results["timers"] = timers
    return combined_results


def _get_num_workers(pipeline_size: int) -> int:
    """Returns a best-guess number of worker threads for a single benchmarking process."""
    return 2 if pipeline_size == 1 else pipeline_size - 1


def get_instance_type() -> str:
    """Try to obtain the maximum number of NeuronCores available on this instance."""
    try:
        import urllib.request

        with urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-type"
        ) as response:
            instance_type = response.read().decode("utf-8")
        log.debug("Automatically determined instance type: {}".format(instance_type))
        return instance_type
    except:
        return None


def _get_cost_per_hour(instance_type: str) -> float:
    # Hourly rates
    instancetype_to_cost = {
        "inf1.xlarge": 0.228,
        "inf1.2xlarge": 0.362,
        "inf1.6xlarge": 1.18,
        "inf1.24xlarge": 4.721,
    }
    try:
        return instancetype_to_cost[instance_type]
    except:
        # Just ignore unknown instance types for now
        return None


def _get_max_neuroncores(instance_type: str = None) -> int:
    """Try to obtain the maximum number of NeuronCores available on this instance."""
    instancetype_to_neuroncores = {
        "inf1.xlarge": 4,
        "inf1.2xlarge": 4,
        "inf1.6xlarge": 16,
        "inf1.24xlarge": 64,
    }
    try:
        if not instance_type:
            instance_type = get_instance_type()
        return instancetype_to_neuroncores[instance_type]
    except:
        num_cores = 2
        log.warning(f"Unknown Neuron device size. Assuming {num_cores} NeuronCores is the maximum.")
        return num_cores


def _get_num_gpus(instance_type: str = None) -> int:
    """Try to obtain the maximum number of NeuronCores available on this instance."""
    instancetype_to_gpus = {
        "g4dn.xlarge": 1,
        "g4dn.2xlarge": 1,
        "g4dn.4xlarge": 1,
        "g4dn.8xlarge": 1,
        "g4dn.16xlarge": 1,
        "g4dn.12xlarge": 4,
        "g4dn.metal": 8,
        "g4ad.xlarge": 1,
        "g4ad.2xlarge": 1,
        "g4ad.4xlarge": 1,
        "g4ad.8xlarge": 2,
        "g4ad.16xlarge": 4,
        "p4d.24xlarge": 8,
    }
    try:
        if not instance_type:
            instance_type = get_instance_type()
        return instancetype_to_gpus[instance_type]
    except:
        log.warning("Unknown GPU device size. Assuming 1 GPU is available.")
        return 1


def _get_num_devices(device_type: str, instance_type: str = None) -> int:
    """This is a stub, to be populated later for other instance types."""
    if device_type == "neuron":
        return _get_max_neuroncores(instance_type)
    elif device_type == "cpu":
        return multiprocessing.cpu_count()
    elif device_type == "cuda" or device_type == "gpu":
        return _get_num_gpus(instance_type)
    else:
        log.warning("An unknown device_type was passed: {}".format(device_type))
        return None


def _sanitize_inputs(inputs, batch_sizes: Union[int, List[int]], dataset_inputs=False) -> List[int]:
    """Return inputs and batch_sizes with matching lengths, or throw an error."""
    if not isinstance(inputs, list):
        inputs = [inputs]
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if not batch_sizes:
        log.warning(
            "Batch sizes were not provided, so assuming 1 and only the first input will be benchmarked."
        )
        batch_sizes = [1]
    if not dataset_inputs:
        if len(batch_sizes) < len(inputs):
            delta = len(inputs) - len(batch_sizes)
            log.warning(
                "Received {} inputs, but only {} batch sizes. Discarding last {} inputs.".format(
                    len(inputs), len(batch_sizes), delta
                )
            )
            inputs = inputs[: len(batch_sizes)]
        elif len(inputs) < len(batch_sizes):
            delta = len(batch_sizes) - len(inputs)
            log.warning(
                "Received {} batch sizes, but only {} inputs. Discarding last {} batch sizes.".format(
                    len(batch_sizes), len(inputs), delta
                )
            )
            batch_sizes = batch_sizes[: len(inputs)]
    return inputs, batch_sizes


def set_verbosity(verbosity: int):
    r"""
    Controls the verbosty of NeuronPerf logging.

    :param int verbosity: 0 = error, 1 = info, 2 = debug
    """
    if 0 == verbosity:
        log.setLevel(logging.ERROR)
    elif 1 == verbosity:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.DEBUG)


def compile(
    compile_fn,
    model,
    inputs,
    batch_sizes: Union[int, List[int]] = None,
    pipeline_sizes: Union[int, List[int]] = None,
    performance_levels: Union[str, List[int]] = None,
    models_dir: str = "models",
    model_name: str = None,
    filename: str = None,
    compiler_args: dict = None,
    verbosity: int = 1,
    **kwargs,
) -> str:
    r"""
    Compiles the provided model with each provided example input, pipeline size, and performance level.

    :param model: The model to compile.
    :param list inputs: A list of example inputs.
    :param Union[int, List[int]] batch_sizes: A list of batch sizes that correspond to the example inputs.
    :param Union[int, List[int]] pipeline_sizes: A list of pipeline sizes to use. See :ref:`neuroncore-pipeline`.
    :param Union[int, List[int]] performance_levels: A list of performance levels to try. Options are: 0 (max accuracy), 1, 2, 3 (max performance, default).  See :ref:`mixed-precision`.
    :param str models_dir: The directory where compilation artifacts will be stored.
    :param str model_name: An optional model name tag to apply to compiled artifacts.
    :param str filename: The name of the model index to write out. If not provided, a name will be generated and returned.
    :param dict compiler_args: Additional compiler arguments to be forwarded with every compilation.
    :param int verbosity: 0 = error, 1 = info, 2 = debug
    :return: A model index filename. If a configuration fails to compile, it will not be included in the index and an error will be logged.
    :rtype: str
    """
    # Set NeuronPerf logging verbosity.
    set_verbosity(verbosity)

    # Standardize arguments.
    if not pipeline_sizes:
        pipeline_sizes = [1]
    if not performance_levels:
        performance_levels = []
    if not compiler_args:
        compiler_args = {}
    if not model_name:
        if isinstance(model, str):
            model_name = model
        else:
            try:
                model_name = model.__name__
            except AttributeError:
                log.warning("Unable to determine a model name, using 'Model'.")
                model_name = "Model"
    if isinstance(pipeline_sizes, int):
        pipeline_sizes = [pipeline_sizes]
    if isinstance(performance_levels, int):
        performance_levels = [performance_levels]

    inputs, batch_sizes = _sanitize_inputs(inputs, batch_sizes)

    # Sanity check and sanitize compiler_args.
    if NEURONCORE_PIPELINE_CORES in compiler_args:
        if pipeline_sizes:
            log.warning(
                (
                    "You provided NeuronCore Pipeline Core sizes using both "
                    "compiler_args and pipeline_sizes. Ignoring flag in compiler_args."
                )
            )
        else:
            pipeline_sizes = [compiler_args[NEURONCORE_PIPELINE_CORES]]
        del compiler_args[NEURONCORE_PIPELINE_CORES]

    if FAST_MATH in compiler_args:
        if performance_levels:
            log.warning(
                (
                    f"You provided performance_levels and {FAST_MATH}. "
                    "Ignoring flag in compiler_args."
                )
            )
        del compiler_args[FAST_MATH]

    # Check if performance levels are within expected bounds.
    max_performance = max(FAST_MATH_OPTIONS)
    performance_levels_invalid = list(
        filter(
            lambda level: level < min(FAST_MATH_OPTIONS) or level > max_performance,
            performance_levels,
        )
    )
    if performance_levels_invalid:
        log.warning(
            "You provided some invalid performance_levels. Ignoring: {}".format(
                performance_levels_invalid
            )
        )
        performance_levels = [
            level
            for level in performance_levels
            if (level in performance_levels) and (level not in performance_levels_invalid)
        ]

    # If we still have no values, set default to max performance.
    if not performance_levels:
        performance_levels.append(max_performance)

    # Create standard output dir, if it doesn't exit.
    os.makedirs(models_dir, exist_ok=True)

    # Compile all requested model combinations.
    model_idxs = []

    # TODO: Support appending to existing index by filtering already-compiled configs.
    def make_index():
        """Create a model index file that contains info about all compiled models."""
        index = model_index.append(*model_idxs)
        # Return the name of the new index file.
        return model_index.save(index, filename=filename)

    compile_idx = 1
    n_compiles = len(inputs) * len(pipeline_sizes) * len(performance_levels)
    for input_idx, example_input in enumerate(inputs):
        batch_size = batch_sizes[input_idx]
        for pipeline_size in pipeline_sizes:
            for performance_level in performance_levels:
                _compiler_args = copy.copy(compiler_args)
                _compiler_args[FAST_MATH] = FAST_MATH_OPTIONS[performance_level]
                if pipeline_size != 1:
                    _compiler_args[NEURONCORE_PIPELINE_CORES] = str(pipeline_size)

                # Construct a more informative model name with some config info
                model_name_ex = "{}_b{}_p{}_{}".format(
                    model_name,
                    batch_size,
                    pipeline_size,
                    model_index.generate_id(),
                )
                log.info(
                    (
                        f"Compiling batch size {batch_size} for {pipeline_size} NeuronCore(s) with performance level "
                        f"{performance_level}/{max_performance}. [{compile_idx}/{n_compiles}]"
                    )
                )
                status = "ready"
                timer = Timer()
                with timer:
                    try:
                        model_filename = compile_fn(
                            model,
                            example_input,
                            models_dir,
                            model_name_ex,
                            compiler_args=_compiler_args,
                            **kwargs,
                        )
                        status = "finished"
                    except KeyboardInterrupt:
                        status = "error"
                        model_filename = None
                        log.error("Compilation interrupted, terminating.")
                        return make_index()
                    except:
                        status = "error"
                        model_filename = None
                        log.exception(
                            (
                                f"Failed to compile input={input_idx}, "
                                f"batch_size={batch_size}, "
                                f"pipeline_size={pipeline_size}, "
                                f"performance_level={performance_level}."
                            )
                        )
                    finally:
                        model_idx = model_index.create(
                            model_filename,
                            model_name=model_name,
                            batch_size=batch_size,
                            pipeline_size=pipeline_size,
                            performance_level=performance_level,
                            compile_s=round(timer.total_duration("s"), 2),
                            status=status,
                        )
                        model_idxs.append(model_idx)
                        filename = make_index()
                compile_idx += 1
    return filename


def run_benchmarker(benchmarker, duration, pipe=None):
    def _send(results):
        if pipe:
            pipe.send(results)
            pipe.close()
        else:
            return results

    try:
        log.debug(f"Benchmarker {benchmarker.id} started.")
        check_freq = 0.1  # Check progress every 0.1 seconds.
        start_time = time.time()
        benchmarker.start()
        elapsed = 0
        while (elapsed < duration) and benchmarker.benchmarking:
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            time.sleep(min(check_freq, remaining))
        benchmarker.stop()
    except:
        trace = "".join(traceback.format_exception(*sys.exc_info()))
        error = BenchmarkerErrorWrapper(trace)
        return _send(error)
    else:
        results = benchmarker.results() if benchmarker.status != ERROR else benchmarker.error
        return _send(results)
    finally:
        log.debug(f"Benchmarker {benchmarker.id} finished.")


def _run_benchmarker_new_interpreter(benchmarker, duration):
    """
    This function is a workaround for frameworks that cannot be safely forked.
    The premise is to launch a new Python interpreter and run benchmarking
    from within the new interpreter. It works by writing serialized benchmarkers
    to temporary files, and then launching run_benchmark_file.py. The script
    writes back serialized results.
    """

    # Temporary serialization workaround. This attribute is inherited from Thread.
    # TODO: Separate data from benchmarking.
    setattr(benchmarker, "_stderr", None)

    script = run_benchmark_file.__file__

    # Serialize the benchmarker to a file.
    f = tempfile.NamedTemporaryFile(delete=False)
    log.debug("Dumping Benchmarker {} to file '{}'.".format(benchmarker.id, f.name))
    try:
        dill.dump(benchmarker, f)
    except dill.PicklingError:
        raise dill.PicklingError(
            (
                "NeuronPerf was unable to serialize the benchmarker. This is probably becuause your model "
                "could not be serialized. Make sure to use top-level classes instead of locals. You may "
                "need to wrap your model and manually load it using Python's importlib."
            )
        )
    f.close()

    # Run the benchmarking script in a clean Python process.
    command = [
        sys.executable,
        script,
        f.name,
        str(duration),
    ]

    # If we are manually loading a model class file in subprocesses, we need to let them know.
    if benchmarker.model_class_name and benchmarker.model_class_file:
        command.append(f"--model_class_name={benchmarker.model_class_name}")
        command.append(f"--model_class_file={benchmarker.model_class_file}")

    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )

    # Interpreter and framework overhead add a delay to processing. We should ensure
    # that during multiinterpreter benchmarking, sufficient time is waited for results.
    timeout = 60 + duration

    try:
        outs, errs = proc.communicate(timeout=timeout)
        with open(f.name, "rb") as fp:
            result = dill.load(fp)
        if isinstance(result, BenchmarkerErrorWrapper):
            raise ChildProcessError(
                "Benchmarker {} encountered an error:\n{}".format(benchmarker.id, result.trace)
            )
        if isinstance(result, Benchmarker):
            # If we still have a benchmarker object instead of results, something
            # went wrong that wasn't handled by the benchmarker routine.
            from pathlib import Path

            path = Path(f.name)
            logs = os.path.join(path.parent, "neuronperf_error_{}".format(str(path.stem)))
            if os.path.exists(logs):
                with open(logs, "rt") as logs_fp:
                    err_logs = logs_fp.readlines()
                os.unlink(logs)
                raise ChildProcessError(
                    "Benchmarker {} failed. Logs from child process:\n{}".format(
                        benchmarker.id, "".join(err_logs)
                    )
                )
            else:
                raise ChildProcessError(
                    (
                        "Benchmarker {} failed and no error logs were found. A child process may have "
                        "aborted. To obtain a stack trace, try running a single configuration inside a "
                        "single process by passing multiprocess=False, multiinterpreter=False"
                    )
                )

        return result
    except subprocess.TimeoutExpired:
        proc.kill()
        raise ChildProcessError(
            "Benchmarker {} stopped responding after {} seconds.".format(benchmarker.id, timeout)
        )
    finally:
        os.unlink(f.name)


def _run_benchmarkers_multiprocess(
    benchmarkers: List[Benchmarker], duration: int, benchmark_func=run_benchmarker
) -> dict:
    results = []
    # Hand each benchmarker object to a subprocess.
    pipes, procs = [], []
    for benchmarker in benchmarkers:
        parent_pipe, child_pipe = multiprocessing.Pipe()
        pipes.append(parent_pipe)
        proc = multiprocessing.Process(
            target=benchmark_func, args=(benchmarker, duration, child_pipe)
        )
        procs.append(proc)
    # Launch benchmarking.
    for proc in procs:
        proc.start()
    # Collect results.
    for id, (pipe, proc) in enumerate(zip(pipes, procs)):
        try:
            proc_result = pipe.recv()
            if isinstance(proc_result, BenchmarkerErrorWrapper):
                log.error("Child process encountered an error:\n{}".format(proc_result.trace))
                raise ChildProcessError()
            proc.join()
            results.append(proc_result)
        except KeyboardInterrupt:
            log.error("Benchmarking interrupted, terminating.")
            for proc in procs:
                proc.terminate()
            raise KeyboardInterrupt()
        except EOFError:
            log.error(
                (
                    f"Child process {id} was killed by the host OS during benchmarking.\n"
                    "You may have run out of memory.\n"
                    "Verify that your model can perform inference without NeuronPerf or try n_models=1."
                )
            )
    return _combine_results(results)


def _run_benchmarkers_multithreaded(
    benchmarkers: List[Benchmarker], duration: int, benchmark_func=run_benchmarker
) -> dict:
    results = []
    timeout = 60 + duration  # Add some time for setup overhead and cleanup.
    try:
        args = ((benchmarker, duration) for benchmarker in benchmarkers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(benchmarkers)) as exe:
            results.extend(exe.map(lambda arg: benchmark_func(*arg), args, timeout=timeout))
        for result in results:
            if isinstance(result, BenchmarkerErrorWrapper):
                raise RuntimeError("Worker thread encountered an error:\n{}".format(result.trace))
    except concurrent.futures.TimeoutError:
        log.error("Benchmarking timed out after {} seconds.".format(timeout))
    except KeyboardInterrupt:
        raise KeyboardInterrupt("Benchmarking interrupted, terminating.")
    return _combine_results(results)


def run_benchmarkers(
    benchmarkers: List[Benchmarker],
    duration: int,
    stats_interval: float = 0.5,
    multiprocess: bool = True,
    multiinterpreter: bool = False,
) -> dict:
    results = {}

    # Launch a background thread to collect system stats during benchmarking.
    stats_thread = StatsThread(stats_interval)
    stats_thread.start()

    try:
        if multiinterpreter:
            if not sys.executable:
                raise ValueError(
                    (
                        "Unable to benchmark in multi-interpreter mode because "
                        "the Python interpreter cannot be located (sys.executable is empty)."
                    )
                )
            # We can safely re-use the multithreaded path here by using a custom benchmarking
            # function that spawns fresh interpreters.
            results = _run_benchmarkers_multithreaded(
                benchmarkers, duration, benchmark_func=_run_benchmarker_new_interpreter
            )
        elif multiprocess:
            results = _run_benchmarkers_multiprocess(benchmarkers, duration)
        else:
            results = _run_benchmarkers_multithreaded(benchmarkers, duration)
    finally:
        stats_thread.join()
        results["cpu_percents"] = stats_thread.cpu_percents
        results["mem_percents"] = stats_thread.mem_percents

    return results


def _get_env_setup_fn(benchmarker_id: int, benchmarker_config: dict, env_setup_fn):
    """Wrap an environment setup function with device-specific requirements."""
    device_type = str(benchmarker_config["device_type"]).lower().strip()
    legacy = bool(os.environ.get("NEURONCORE_GROUP_SIZES"))
    if "neuron" == device_type:

        @functools.wraps(env_setup_fn)
        def _env_setup_fn():
            import os

            id = benchmarker_id
            config = benchmarker_config
            pipeline_size = config["pipeline_size"]
            if config["multiprocess"] or config["multiinterpreter"]:
                # In multiprocess mode, need to specify the exact cores for the process.
                min_core = pipeline_size * id
                max_core = min_core + (pipeline_size - 1)
                visible_cores = f"{min_core}-{max_core}"

                if legacy:
                    os.environ["NEURONCORE_GROUP_SIZES"] = str(pipeline_size)
                else:
                    os.environ["NEURON_RT_VISIBLE_CORES"] = visible_cores
            else:
                # In multithreaded mode, all required cores are allocated in this process.
                n_models = config["n_models"]
                if legacy:
                    os.environ["NEURONCORE_GROUP_SIZES"] = ",".join([str(pipeline_size)] * n_models)
                else:
                    os.environ["NEURON_RT_VISIBLE_CORES"] = "0-{}".format(
                        n_models * pipeline_size - 1
                    )

            # Finally, call any additional custom setup function provided.
            if env_setup_fn:
                env_setup_fn(id, config)

        return _env_setup_fn
    elif device_type == "cpu":
        return env_setup_fn
    elif device_type == "cuda" or device_type == "gpu":

        @functools.wraps(env_setup_fn)
        def _env_setup_fn():
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = str(benchmarker_id)

            if env_setup_fn:
                env_setup_fn(benchmarker_id, benchmarker_config)

        return _env_setup_fn
    else:
        log.warning(
            (
                f"NeuronPerf does not implement a proper environment setup for {device_type}. "
                "You may need to provide your own."
            )
        )
        return env_setup_fn


def _get_setup_fn(benchmarker_id: int, benchmarker_config: dict, setup_fn):
    """Wraps a customer provided setup function with additional info from the benchmarker."""
    if not setup_fn:
        return None

    @functools.wraps(setup_fn)
    def _setup_fn(model):
        setup_fn(benchmarker_id, benchmarker_config, model)

    return _setup_fn


def _get_device_id(benchmarker_id: int, benchmarker_config: dict):
    """Calculate an appropriate device id for a benchmarker object."""
    device_id = benchmarker_id
    device_type = str(benchmarker_config["device_type"]).lower().strip()
    if device_type in SUPPORTED_DEVICE_TYPES:
        if not (benchmarker_config["multiprocess"] or benchmarker_config["multiinterpreter"]):
            device_id = benchmarker_id * benchmarker_config["pipeline_size"]
        return device_id
    else:
        log.warning(
            "Assuming device_id={} for benchmarker_id={} for unknown device_type={}".format(
                device_id, benchmarker_id, device_type
            )
        )
    return device_id


def benchmark(
    load_fn: Callable[[str, int], Any],
    model_filename: str,
    inputs: Any,
    batch_sizes: Union[int, List[int]] = None,
    duration: float = BENCHMARK_SECS,
    n_models: Union[int, List[int]] = None,
    pipeline_sizes: Union[int, List[int]] = None,
    performance_levels: Union[int, List[int]] = None,
    workers_per_model: Union[int, None] = None,
    env_setup_fn: Callable[[int, Dict], None] = None,
    setup_fn: Callable[[int, Dict, Any], None] = None,
    preprocess_fn: Callable[[Any], Any] = None,
    postprocess_fn: Callable[[Any], Any] = None,
    dataset_loader_fn: Callable[[Any, int], Any] = None,
    multiprocess: bool = True,
    multiinterpreter: bool = False,
    return_timers: bool = False,
    stats_interval: float = 0.5,
    device_type: str = "neuron",
    cost_per_hour: float = None,
    model_name: str = None,
    model_class_name: str = None,
    model_class_file: str = None,
    verbosity: int = 1,
) -> List[Dict]:
    r"""
    Benchmarks the model index or individiual model using the provided inputs.
    If a model index is provided, additional fields such as ``pipeline_sizes`` and
    ``performance_levels`` can be used to filter the models to benchmark. The default
    behavior is to benchmark all configurations in the model index. Any additional
    compiler_args passed will be forwarded to the compiler on every invocation.

    :param Callable[[str, int], Any] load_fn: A function that accepts a model filename and device id, and returns a loaded model. This is automatically passed through the subpackage calls (e.g. ``neuronperf.torch.benchmark``).
    :param str model_filename: A path to a model index from compile or path to an individual model. For CPU benchmarking, a class should be passed that can be instantiated with a default constructor (e.g. ``MyModelClass``).
    :param list inputs: A list of example inputs. If the list contains tuples, they will be destructured on inference to support multiple arguments.
    :param Union[int, List[int]] batch_sizes: A list of ints indicating batch sizes that correspond to the inputs. Assumes 1 if not provided.
    :param duration float: The number of seconds to benchmark each model.
    :param n_models Union[int, List[int]]: The number of models to run in parallel. Default behavior runs 1 model and the max number of models possible, determined by a best effort from ``device_type``, instance size, or other environment state.
    :param Union[int, List[int]] pipeline_sizes: A list of pipeline sizes to use. See :ref:`neuroncore-pipeline`.
    :param Union[int, List[int]] performance_levels: A list of performance levels to try. Options are: 0 (max accuracy), 1, 2, 3 (max performance, default). See :ref:`mixed-precision`.
    :param Union[int, List[int]] workers_per_model: The number of workers to use per model loaded. If ``None``, this is automatically selected.
    :param Callable[[int, Dict], None] env_setup_fn: A custom environment setup function to run in each subprocess before model loading. It will receive the benchmarker id and config.
    :param Callable[[int, Dict, Any], None] setup_fn: A function that receives the benchmarker id, config, and model to perform last minute configuration before inference.
    :param Callable[[Any], Any]: preprocess_fn: A custom preprocessing function to perform on each input before inference.
    :param Callable[[Any], Any]: postprocess_fn: A custom postprocessing function to perform on each input after inference.
    :param bool multiprocess: When True, model loading is dispatched to forked subprocesses. Should be left alone unless debugging.
    :param bool multiinterpreter: When True, benchmarking is performed in a new python interpreter per model. All parameters must be serializable. Overrides multiprocess.
    :param bool return_timers: When True, the return of this function is a list of tuples ``(config, results)`` with detailed information. This can be converted to reports with ``get_reports(results)``.
    :param float stats_interval: Collection interval (in seconds) for metrics during benchmarking, such as CPU and memory usage.
    :param str device_type: This will be set automatically to one of the ``SUPPORTED_DEVICE_TYPES``.
    :param float cost_per_hour: The price of this device / hour. Used to estimate cost / 1 million infs in reports.
    :param str model_name: A friendly name for the model to use in reports.
    :param str model_class_name: Internal use.
    :param str model_class_file: Internal use.
    :param int verbosity: 0 = error, 1 = info, 2 = debug
    :return: A list of benchmarking results.
    :rtype: List[Dict]
    """
    # Set NeuronPerf logging verbosity.
    set_verbosity(verbosity)

    # --------------------------------------------
    # Input validation
    # --------------------------------------------
    # Validate that enough information was provided.
    if not load_fn:
        raise ValueError(
            "You should call benchmark() through a framework submodule, e.g. neuronperf.torch.benchmark()."
        )
    if not isinstance(model_filename, str):
        raise ValueError(
            "You must provide the path to a saved model or the path to a model index from neuronperf.compile()."
        )

    # Useful for debugging.
    if not multiprocess and not multiinterpreter:
        log.warning("Benchmarking in a single process.")

    # Standardize inputs.
    dataset_inputs = dataset_loader_fn is not None
    if (not dataset_inputs) and (not isinstance(inputs, list)):
        inputs = [inputs]
    if isinstance(n_models, int):
        n_models = [n_models]
    if isinstance(pipeline_sizes, int):
        pipeline_sizes = [pipeline_sizes]
    if isinstance(performance_levels, int):
        performance_levels = [performance_levels]
    if workers_per_model is None:
        workers_per_model = []
    elif isinstance(workers_per_model, int):
        workers_per_model = [workers_per_model]
    if duration < BENCHMARK_SECS:
        log.warning("Results may be unreliable with short test durations.")

    # If the model_filename is JSON, attempt to interpret it as a model index.
    index = None
    if model_filename.endswith(model_index.MODEL_INDEX_SUFFIX):
        index = model_index.load(model_filename)

    # If we loaded a model_index, ensure provided inputs are compatible
    # and use it to refine the benchmarking combinations we will run.
    if index:
        # Extract a model name from the index, if possible.
        if not model_name:
            model_name = index["model_name"]

        # If batch_sizes, pipeline_sizes and/or performance_levels were provided,
        # treat them as filters on the index. A value of None is treated as no filter.
        # See the docs for model_index.filter().
        index = model_index.filter(
            index,
            status="finished",  # only take compiled models
            batch_size=batch_sizes,  # select all requested batch sizes
            pipeline_size=pipeline_sizes,
            performance_level=performance_levels,
        )

        if 0 == len(index["model_configs"]):
            raise ValueError(
                "No models were found in the model index matching requested criteria. Check that compilation succeeded."
            )

        # If a model index was provided without batch_sizes, extract the sizes from the index.
        if not batch_sizes:
            # Select unique batch_sizes in model index.
            batch_sizes = set(config["batch_size"] for config in index["model_configs"])
            batch_sizes = sorted(list(batch_sizes))

    # Validate batch sizes after attempting to extract from the model index.
    inputs, batch_sizes = _sanitize_inputs(inputs, batch_sizes, dataset_inputs)

    # If we still don't have a model name, use the filename.
    if not model_name:
        model_name = model_filename

    # If no pipeline_sizes are provided, we'll assume it's 1 for a single model unless told otherwise.
    if not pipeline_sizes:
        log.debug("Pipeline size was not specified, assuming 1.")
        pipeline_sizes = [1]

    # Assume max performance is desired.
    if not performance_levels:
        max_performance = max(FAST_MATH_OPTIONS)
        log.debug(f"Performance level was not specified, assuming {max_performance}.")
        performance_levels = [max_performance]

    # If a model was provided directly without a model index, build a dummy model index.
    # A single model can not possibly have been compiled for more than 1 configuration,
    # hence why we can assume index [0].
    if not index:
        index = model_index.create(
            filename=model_filename,
            model_name=model_name,
            batch_size=batch_sizes[0],
            pipeline_size=pipeline_sizes[0],
            performance_level=performance_levels[0],
        )

    model_configs = index["model_configs"]

    # --------------------------------------------
    # Benchmarking
    # --------------------------------------------

    # Estimate time remaining based on configs requested to run.
    # If n_models wasn't provided, the default benchmarks [min, max].
    n_models_est = 2 if not n_models else len(n_models)
    # If workers_per_model wasn't provided, the default benchmarks [1, 2].
    n_models_est *= 2 if not workers_per_model else len(workers_per_model)
    secs_remaining = len(model_configs) * n_models_est * duration
    mins_remaining = None if secs_remaining < 60 else round(secs_remaining / 60.0, 1)
    etr = f"{mins_remaining} minutes" if mins_remaining else f"{int(round(secs_remaining))} seconds"
    log.info("Benchmarking '{}', ~{} remaining.".format(model_filename, etr))

    # Try to determine instance type.
    instance_type = get_instance_type()
    if not instance_type:
        instance_type = "unknown"

    # Try to automatically determine the maximum number of devices available.
    max_devices = _get_num_devices(device_type, instance_type)
    log.debug("Automatically determined number of devices: {}".format(max_devices))

    # Try to detect cost / hour for this device.
    if not cost_per_hour:
        cost_per_hour = _get_cost_per_hour(instance_type)

    # Run through all requested combinations and generate a report.
    # This will produce a list of tuples, (config, results).
    all_results = []

    def make_reports():
        """Helper to generate reports from available results."""
        # If all_results was set, we return the unmodified benchmarking results.
        return all_results if return_timers else get_reports(all_results, cost_per_hour)

    for model_config in model_configs:
        batch_size = model_config["batch_size"]
        pipeline_size = model_config["pipeline_size"]

        # Determine the number of model copies for each benchmarking session.
        model_counts = n_models
        # If the user didn't provide n_models, choose reasonable defaults.
        if not model_counts:
            # Try to run a single model and the max models supported on this hardware.
            if max_devices and (max_devices // pipeline_size > 1):
                model_counts = [1, max_devices // pipeline_size]
            else:
                model_counts = [1]
        # If the user provided model counts and we determine they are too large, emit a warning.
        else:
            if max_devices:
                model_counts_too_large = list(
                    filter(
                        lambda model_count: model_count * pipeline_size > max_devices, model_counts
                    )
                )
                if model_counts_too_large:
                    log.warning(
                        (
                            "Some values of n_models exceed the number of devices available: "
                            f"{model_counts_too_large} > {max_devices}"
                        )
                    )

        # Compute number of workers for this pipeline size, if not specified.
        n_workers = workers_per_model
        if not n_workers:
            n_workers = [_get_num_workers(pipeline_size)]
            # 1 worker thread == min latency
            if 1 not in n_workers:
                n_workers.insert(0, 1)

        for _workers_per_model in n_workers:
            # We now know everything we need to benchmark.
            #   1. Build a comprehensive benchmarker config,
            #   2. build one benchmarker per model,
            #   3. run the benchmarkers in parallel,
            #   4. and collect the results for this configuration.
            for model_count in model_counts:
                # 1. Benchmarker config
                config = {
                    "model_filename": model_config["filename"],
                    "model_name": model_name,
                    "device_type": device_type,
                    "instance_type": instance_type,
                    "batch_size": batch_size,
                    "n_models": model_count,
                    "workers_per_model": _workers_per_model,
                    "pipeline_size": pipeline_size,
                    "n_devices": model_count * pipeline_size,
                    "performance_level": model_config["performance_level"],
                    "multiprocess": multiprocess,
                    "multiinterpreter": multiinterpreter,
                    "stats_interval": str(stats_interval),
                    "start_dts": time.strftime("%Y%m%d-%H%M%S"),
                    "duration": str(duration),
                }

                # 2. Build the benchmarkers
                benchmarkers = []
                for benchmarker_id in range(model_count):
                    benchmarker = Benchmarker(
                        id=benchmarker_id,
                        device_id=_get_device_id(benchmarker_id, config),
                        load_fn=load_fn,
                        model_filename=model_config["filename"],
                        inputs=inputs if dataset_inputs else inputs[batch_sizes.index(batch_size)],
                        workers_per_model=_workers_per_model,
                        env_setup_fn=_get_env_setup_fn(benchmarker_id, config, env_setup_fn),
                        setup_fn=_get_setup_fn(benchmarker_id, config, setup_fn),
                        preprocess_fn=preprocess_fn,
                        postprocess_fn=postprocess_fn,
                        dataset_loader_fn=dataset_loader_fn,
                        model_class_name=model_class_name,
                        model_class_file=model_class_file,
                    )
                    benchmarkers.append(benchmarker)

                # 3. Run benchmarkers in parallel
                log.debug("Running model config: {}".format(config))
                try:
                    results = run_benchmarkers(
                        benchmarkers,
                        duration,
                        stats_interval=stats_interval,
                        multiprocess=multiprocess,
                        multiinterpreter=multiinterpreter,
                    )

                    # 4. Collect results
                    config["stop_dts"] = time.strftime("%Y%m%d-%H%M%S")
                    all_results.append((config, results))
                except KeyboardInterrupt:
                    # If we are interrupted, return whatever we have on hand.
                    return make_reports()
                except:
                    # If something else goes wrong with the model, we should
                    # log this configuration and move on.
                    log.exception("Failure benchmarking config: {}".format(config))

    return make_reports()
