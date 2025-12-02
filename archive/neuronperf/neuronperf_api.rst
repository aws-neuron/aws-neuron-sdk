.. _neuronperf_api:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.
   :date-modified: 12-02-2025

NeuronPerf API
==============

.. contents:: Table of Contents
   :local:
   :depth: 2


Due to a bug in Sphinx, some of the type annotations may be incomplete. You can :download:`download the source code here </src/neuronperf.tar.gz>`. In the future, the source will be hosted in a more browsable way.

.. py:function:: compile(compile_fn, model, inputs, batch_sizes: Union[int, List[int]] = None, pipeline_sizes: Union[int, List[int]] = None, performance_levels: Union[str, List[int]] = None, models_dir: str = "models", filename: str = None, compiler_args: dict = None, verbosity: int = 1, *args, **kwargs) -> str:

    Compiles the provided model with each provided example input, pipeline size, and performance level.
    Any additional compiler_args passed will be forwarded to the compiler on every invocation.

    :param model: The model to compile.
    :param list inputs: A list of example inputs.
    :param batch_sizes: A list of batch sizes that correspond to the example inputs.
    :param pipeline_sizes: A list of pipeline sizes to use. See :ref:`neuroncore-pipeline`.
    :param performance_levels: A list of performance levels to try. Options are: 0 (max accuracy), 1, 2, 3 (max performance, default).  See :ref:`neuron-cc-training-mixed-precision`.
    :param str models_dir: The directory where compilation artifacts will be stored.
    :param str model_name: An optional model name tag to apply to compiled artifacts.
    :param str filename: The name of the model index to write out. If not provided, a name will be generated and returned.
    :param dict compiler_args: Additional compiler arguments to be forwarded with every compilation.
    :param int verbosity: 0 = error, 1 = info, 2 = debug
    :return: A model index filename. If a configuration fails to compile, it will not be included in the index and an error will be logged.
    :rtype: str

.. _neuronperf_api_benchmark:


.. py:function:: benchmark(load_fn: Callable[[str, int], Any], model_filename: str, inputs: Any, batch_sizes: Union[int, List[int]] = None, duration: float = BENCHMARK_SECS, n_models: Union[int, List[int]] = None, pipeline_sizes: Union[int, List[int]] = None, cast_modes: Union[str, List[str]] = None, workers_per_model: Union[int, None] = None, env_setup_fn: Callable[[int, Dict], None] = None, setup_fn: Callable[[int, Dict, Any], None] = None, preprocess_fn: Callable[[Any], Any] = None, postprocess_fn: Callable[[Any], Any] = None, dataset_loader_fn: Callable[[Any, int], Any] = None, verbosity: int = 1, multiprocess: bool = True, multiinterpreter: bool = False, return_timers: bool = False, device_type: str = "neuron") -> List[Dict]:

    Benchmarks the model index or individiual model using the provided inputs.
    If a model index is provided, additional fields such as ``pipeline_sizes`` and
    ``performance_levels`` can be used to filter the models to benchmark. The default
    behavior is to benchmark all configurations in the model index.

    :param load_fn: A function that accepts a model filename and device id, and returns a loaded model. This is automatically passed through the subpackage calls (e.g. ``neuronperf.torch.benchmark``).
    :param str model_filename: A path to a model index from compile or path to an individual model. For CPU benchmarking, a class should be passed that can be instantiated with a default constructor (e.g. ``MyModelClass``).
    :param list inputs: A list of example inputs. If the list contains tuples, they will be destructured on inference to support multiple arguments.
    :param batch_sizes: A list of ints indicating batch sizes that correspond to the inputs. Assumes 1 if not provided.
    :param float duration: The number of seconds to benchmark each model.
    :param n_models: The number of models to run in parallel. Default behavior runs 1 model and the max number of models possible, determined by a best effort from ``device_type``, instance size, or other environment state.
    :param pipeline_sizes: A list of pipeline sizes to use. See :ref:`neuroncore-pipeline`.
    :param performance_levels: A list of performance levels to try. Options are: 0 (max accuracy), 1, 2, 3 (max performance, default). See :ref:`neuron-cc-training-mixed-precision`.
    :param workers_per_model: The number of workers to use per model loaded. If ``None``, this is automatically selected.
    :param env_setup_fn: A custom environment setup function to run in each subprocess before model loading. It will receive the benchmarker id and config.
    :param setup_fn: A function that receives the benchmarker id, config, and model to perform last minute configuration before inference.
    :param preprocess_fn: A custom preprocessing function to perform on each input before inference.
    :param postprocess_fn: A custom postprocessing function to perform on each input after inference.
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
    :rtype: list[dict]


.. py:function:: get_reports(results)

   Summarizes and combines the detailed results from ``neuronperf.benchmark``, when run with ``return_timers=True``. One report dictionary is produced per model configuration benchmarked. The list of reports can be fed directly to other reporting utilities, such as ``neuronperf.write_csv``.

   :param list[tuple] results: The list of results from ``neuronperf.benchmark``.
   :param list[int] batch_sizes: The batch sizes that correspond to the `inputs` provided to ``compile`` and ``benchmark``. Used to correct throughput values in the reports.
   :return: A list of dictionaries that summarize the results for each model configuration.
   :rtype: list[dict]

.. py:function:: print_reports(reports, cols=SUMMARY_COLS, sort_by="throughput_peak", reverse=False)

    Print a report to the terminal.
    Example of default behavior:

    >>> neuronperf.print_reports(reports)
    throughput_avg latency_ms_p50 latency_ms_p99 n_models pipeline_size  workers_per_model batch_size model_filename
    329.667        6.073          6.109          1        1              2                 1          models/model_b1_p1_83bh3hhs.pt

    :param reports: Results from `get_reports`.
    :param cols: The columns in the report to be displayed.
    :param sort_by: Sort the cols by the specified key.
    :param reverse: Sort order.

.. py:function:: write_csv(reports: list[dict], filename: str = None, cols=REPORT_COLS)

    Write benchmarking reports to CSV file.

    :param list[dict] reports: Results from `neuronperf.get_reports`.
    :param str filename: Filename to write. If not provided, generated from model_name in report and current timestamp.
    :param list[str] cols: The columns in the report to be kept.
    :return: The filename written.
    :rtype: str

.. py:function:: write_json(reports: list[dict], filename: str = None)

    Writes benchmarking reports to a JSON file.

	:param list[dict] reports: Results from `neuronperf.get_reports`.
	:param str filename: Filename to write. If not provided, generated from model_name in report and current timestamp.
	:return: The filename written.
	:rtype: str


.. py:function:: model_index.append(*model_indexes: Union[str, dict]) -> dict:

    Appends the model indexes non-destructively into a new model index, without
    modifying any of the internal data.

    This is useful if you have benchmarked multiple related models and wish to
    combine their respective model indexes into a single index.

    Model name will be taken from the first index provided.
    Duplicate configs will be filtered.

    :param model_indexes: Model indexes or paths to model indexes to combine.
    :return: A new dictionary representing the combined model index.
    :rtype: dict


.. py:function:: model_index.copy(old_index: Union[str, dict], new_index: str, new_dir: str) -> str:

    Copy an index to a new location. Will rename ``old_index``
    to ``new_index`` and copy all model files into ``new_dir``,
    updating the index paths.

    This is useful for pulling individual models out of a pool.

    Returns the path to the new index.


.. py:function:: model_index.create(filename, input_idx=0, batch_size=1, pipeline_size=1, cast_mode=DEFAULT_CAST, compile_s=None)

    Create a new model index from a pre-compiled model.

    :param str filename: The path to the compiled model.
    :param int input_idx: The index in your inputs that this model should be run on.
    :param int batch_size: The batch size at compilation for this model.
    :param int pipeline_size: The pipeline size used at compilation for this model.
    :param str cast_mode: The casting option this model was compiled with.
    :param float compile_s: Seconds spent compiling.
    :return: A new dictionary representing a model index.
    :rtype: dict


.. py:function:: model_index.delete(filename: str):

    Deletes the model index and all associated models referenced by the index.


.. py:function:: model_index.filter(index: Union[str, dict], **kwargs) -> dict:

    Filters provided model index on provided criteria and returns a new index.
    Each kwarg is a standard (k, v) pair, where k is treated as a filter name
    and v may be one or more values used to filter model configs.


.. py:function:: model_index.load(filename) -> dict:

    Load a NeuronPerf model index from a file.


.. py:function:: model_index.move(old_index: str, new_index: str, new_dir: str) -> str:

    This is the same as ``copy`` followed by ``delete`` on the old index.


.. py:function:: model_index.save(model_index, filename: str = None, root_dir=None) -> str:

    Save a NeuronPerf model index to a file.


