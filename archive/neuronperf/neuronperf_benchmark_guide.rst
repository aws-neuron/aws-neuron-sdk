.. _neuronperf_benchmark_guide:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.
   :date-modified: 12-02-2025

==========================
NeuronPerf Benchmark Guide
==========================

The call to ``neuronperf[torch/tensorflow/mxnet/cpu].benchmark`` is used to measure your model performance. It will choose reasonable defaults if none are provided, and will return back reports that summarize the benchmarking results.

What is the default behavior of ``benchmark``?
----------------------------------------------

That will depend how you provided your model and how your model was compiled.

The two most common ways to provide your model are:

#. Provide the path to your compiled model
#. Provide the path to a model index from ``neuronperf.compile`` (a JSON file)


Data Parallel
~~~~~~~~~~~~~

Your model is benchmarked on provided ``inputs`` in 4 different configurations:
   #. A single model on 1 NeuronCore with one worker (min. latency)
   #. A single model on 1 NeuronCore with two workers (max. throughput / NC)
   #. ``MAX`` models on ``MAX`` NeuronCores with one worker (min. latency + max. instance usage)
   #. ``MAX`` models on ``MAX`` NeuronCores with two workers (max. throughput + max. instance usage)

The value ``MAX`` is automatically determined by your instance size. If it can't be identified, those configurations will be skipped.

The primary benefit of (3) and (4) is to verify that your model scales well at maximum instance usage.

.. note::

   If you provided the path to a model index from ``compile``:
      * Your input parameters to ``benchmark`` (``batch_sizes``, etc.) are treated as filters on the index
      * Each remaining model configuration is benchmarked as described in (1)


Pipeline
~~~~~~~~

Pipeline mode is active when using a Neuron device and ``pipeline_sizes > 1``. The same behavior as described in Data Parallel applies, except that only one worker configuration is executed: the optimal number of workers for your pipeline size, unless manually overridden.


Parameters
----------

Below are some useful and common parameters to tweak. Please see the :ref:`neuronperf_api` for full details.

* ``n_models`` controls how many models to load. The default behavior is ``n_models=[1, MAX]``.
* ``workers_per_model`` controls how many worker threads will be feeding inputs to each model. The default is automatically determined.
* ``pipeline_sizes`` tells the benchmarker how many cores are needed for your model so that each model instance can be loaded properly. Default is 1.
* ``duration`` controls how long to run each configuration.
* ``batch_sizes`` is used to inform the benchmarker of your input shape so that throughput can be computed correctly.

Almost all NeuronPerf behaviors are controllable via arguments found in the :ref:`neuronperf_api`. This guide attempts to provide some context and examples for those arguments.

Inputs
------

Models accept one or more inputs to operate on. Since NeuronPerf needs to support multiple inputs for multiple models, as well as multi-input models, there are some details that may need your attention. See the :ref:`neuronperf_framework_notes` for details.

Multi-input Models
~~~~~~~~~~~~~~~~~~

If your model accepts multiple inputs, you must provide them in a ``tuple``. For example, suppose you have a model like this:

.. code:: python


	class Model(torch.nn.Module):
		def forward(self, x, y, z):
			...
			return output


In order for NeuronPerf to pass along your multiple inputs correctly, you should provide them as a ``tuple``:

.. code:: python

	inputs = (x, y, z)
	npf.torch.benchmark(model_filename, inputs, ...)

If you are compiling and/or benchmarking multiple models, you can pass different sized inputs as a list of tuples:

.. code:: python

	inputs = [(x1, y1, z1), (x2, y2, z2), ...]
	npf.torch.benchmark(model_filename, inputs, ...)


Preprocessing and Postprocessing
--------------------------------

Many models have additional preprocessing and postprocessing steps involved that may add non-negligible overhead to inference time. NeuronPerf supports these use cases through the use of custom functions.

Preprocessing
~~~~~~~~~~~~~

Recall that NeuronPerf expects (or wraps) each model input into a ``tuple``. These tuples will be unpacked before calling your model.

Here is an example for a model with one input. The example multiples the input by 5 before inference.

.. code:: python

    def preprocess_fn(x):
        return x * 5

    ...

    # Benchmark with custom preprocessing function
    reports = npf.torch.benchmark(
            filename,
            inputs,
            ...,
            preprocess_fn = preprocess_fn,
    )

Or if your model expects multiple inputs:

.. code:: python

    def preprocess_fn(x, y, z):
        return x / 255, y / 255, z / 255

    ...

    # Benchmark with custom preprocessing function
    reports = npf.torch.benchmark(
            filename,
            inputs,
            ...,
            preprocess_fn = preprocess_fn,
    )

Postprocessing
~~~~~~~~~~~~~~

Postprocessing is almost identical to preprocessing, except that your function will receive whatever the output of your model is, exactly as returned without modification. There are no type guarantees.

.. code:: python

   def postprocess_fn(x):
      return x.argmax()

   ...

   # Benchmark with custom preprocessing function
   reports = npf.torch.benchmark(
         filename,
         inputs,
         ...,
         postprocess_fn = postprocess_fn,
   )

Minimal Latency
---------------

Suppose you are interested in the minimal latency achievable with your model. In this case, there is no need for more than one worker to execute at a time. We can manually specify the number of workers to use. See below :ref:`neuronperf_worker_threads`.


.. _neuronperf_worker_threads:

Worker Threads
--------------

The argument ``workers_per_model`` controls the number of worker threads that are trying to prepare and load examples onto a single NeuronCore at a time. Therefore, a value of 1 corresponds to 1 thread / model. If ``n_models=16``, then there would be 16 worker threads, one per model. This number is selected based upon whether you are using DataParallel (i.e. ``pipeline_sizes == 1``), or Pipeline Mode (``pipeline_sizes != 1``).

By default, NeuronPerf will try to pick try multiple combinations of model copies and workers. You may be interested in controlling this manually.

.. code:: python

   reports = npf.torch.benchmark('model_neuron_b1.pt', ..., workers_per_model=1)


You may also pass a list, as with other parameters:

.. code:: python

   workers_per_model = [1, 2] # Same as the default for data parallel
   reports = npf.torch.benchmark('model_neuron_b1.pt', ..., workers_per_model=workers_per_model)

With the default number of :ref:`neuronperf_model_copies`, a call to ``print_results`` might look like this:

.. code:: bash

   throughput_avg latency_ms_p50 latency_ms_p99 n_models       pipeline_size  workers_per_model batch_size     model_filename
   307.25         3.251          3.277          1              1              1                 1              models/a5cff386-89ca-4bbf-9087-d0e624c3c604.pt
   2746.0         5.641          6.82           16             1              1                 1              models/a5cff386-89ca-4bbf-9087-d0e624c3c604.pt
   329.5          6.053          6.108          1              1              2                 1              models/a5cff386-89ca-4bbf-9087-d0e624c3c604.pt
   2809.0         10.246         12.52          16             1              2                 1              models/a5cff386-89ca-4bbf-9087-d0e624c3c604.pt


.. _neuronperf_model_copies:

Model Copies
------------

By default, NeuronPerf will benchmark two settings for ``n_models``:
   1. A single copy
   2. The maximum number number of copies for your instance size

You can override this behavior by passing ``n_models`` to ``benchmark``, as shown below:

.. code:: python

   reports = npf.torch.benchmark('model_neuron_b1.pt', ..., n_models=6)

or

.. code:: python

   n_models = list(range(1, 10))
   reports = npf.torch.benchmark('model_neuron_b1.pt', ..., n_models=n_models)

.. _neuronperf_pipeline_mode:

Pipeline Mode
-------------

By default, NeuronPerf will assume you intend to use DataParallel, with two exceptions:

* You compiled your model using NeuronPerf for pipeline mode
* You constructed a :ref:`neuronperf_model_index` that uses pipeline mode

You can also manually tell NeuronPerf that your model was compiled for pipeline mode. It is similar to how other arguments are passed.

.. code:: python

   reports = npf.torch.benchmark('model_neuron_b1.pt', ..., pipeline_sizes=2)

If you are passing multiple models in an index, then you should pass a list for ``pipeline_sizes``.

.. code:: python

   reports = npf.torch.benchmark('model_index.json', ..., pipeline_sizes=[1, 2, 3])


Duration
--------

NeuronPerf will benchmark each configuration specified for 60 seconds by default. You can control the duration by passing ``duration`` (in seconds).

.. code:: python

   reports = npf.torch.benchmark('model_index.json', ..., duration=10)

.. warning::

   If you make the duration too short, it may expire before all models are loaded and have had time to execute.


Custom Datasets (Beta)
----------------------

Currently, only PyTorch supports custom datasets, and the interface is subject to change. If you provide a custom dataset, it will be fully executed on each loaded model copy. So if you provide ``n_models=2``, your dataset will be run through twice in parallel.

To use this API, call ``benchmark`` passing a ``torch.utils.data.Dataset`` to ``inputs``. You can easily create your own ``Dataset`` by implementing the interface, or use one of the available datasets. For example:

.. code:: python

   import torchvision

   dataset = torchvision.datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=ToTensor()
   )

   reports = npf.torch.benchmark('model_index.json', inputs=dataset, batch_sizes=[8], preprocess_fn=lambda x: x[0], loop_dataset=False)

.. note::

   The ``preprocess_fn`` is required here to extract image input from the ``(image, label)`` tuple generated by dataloader. If the length of dataset is not sufficient to get the runtime performance, one can set ``loop_dataset=True`` to rerun dataset until certain duration. 

Results
-------

Viewing and Saving
~~~~~~~~~~~~~~~~~~

There are currently three ways to view results.

- ``neuronperf.print_reports(...)``
   - Dump abbrieviated results in your terminal
- ``neuronperf.write_csv(...)``
   - Store metrics of interest as CSV
- ``neuronperf.write_json(...)``
   - Store everything as JSON

See the :ref:`neuronperf_api` for full details.

Full Timing Results
~~~~~~~~~~~~~~~~~~~

NeuronPerf automatically combines and summarizes the detailed timing information collecting during benchmarking. If you wish to receive everything back yourself, you can use:

.. code:: python

   results = npf.torch.benchmark('model_index.json', ..., return_timers=True)

If you later wish to produce reports the same way that NeuronPerf does internally, you can call:

.. code:: python

   reports = npf.get_reports(results)

Verbosity
---------

Verbosity is an integer, currently one of ``{0, 1, 2}``, where:

* 0 = SILENT
* 1 = INFO (default)
* 2 = VERBOSE / DEBUG

Example:

.. code:: python

   reports = npf.torch.benchmark(..., n_models=1, duration=5, verbosity=2)

.. code:: bash

   DEBUG:neuronperf.benchmarking - Cast mode was not specified, assuming default.
   INFO:neuronperf.benchmarking - Benchmarking 'resnet50.json', ~5 seconds remaining.
   DEBUG:neuronperf.benchmarking - Running model config: {'model_filename': 'models/model_b1_p1_83bh3hhs.pt', 'device_type': 'neuron', 'input_idx': 0, 'batch_size': 1, 'n_models': 1, 'workers_per_model': 2, 'pipeline_size': 1, 'cast_mode': None, 'multiprocess': True, 'multiinterpreter': False, 'start_dts': '20211111-062818', 'duration': '5'}
   DEBUG:neuronperf.benchmarking - Benchmarker 0 started.
   DEBUG:neuronperf.benchmarking - Benchmarker 0, Worker 0 started.
   DEBUG:neuronperf.benchmarking - Benchmarker 0, Worker 1 started.
   DEBUG:neuronperf.benchmarking - Benchmarker 0, Worker 0 finished after 738 inferences.
   DEBUG:neuronperf.benchmarking - Benchmarker 0, Worker 1 finished after 738 inferences.
   DEBUG:neuronperf.benchmarking - Benchmarker 0 finished.
   throughput_avg latency_ms_p50 latency_ms_p99 n_models       pipeline_size  workers_per_model batch_size     model_filename
   329.667        6.073          6.109          1              1              2                 1              models/model_b1_p1_83bh3hhs.pt


Internal Process Model
----------------------

For each model loaded (see :ref:`neuronperf_model_copies`), a process is spawned. Each process may use multiple threads (see :ref:`neuronperf_worker_threads`). The threads will continue to load examples and keep the hardware busy.

NeuronPerf spawns processes slightly differently between frameworks. For PyTorch and Apache MXNet, processes are forked. For Tensorflow/Keras, a fresh interpreter is launched, and benchmarkers are serialized and run as a script.

If you suspect you are having trouble due to the way processes are managed, you have two mechanisms of control:

.. code:: python

   reports = npf.torch.benchmark(..., multiprocess=False)

Default is ``True``, and ``False`` will disable multiprocessing and run everything inside a single parent process. This may not work for all frameworks beyond the first model configuration, because process teardown is used to safely deallocate models from the hardware. It is not recommeneded to benchmark this way.


.. code:: python

   reports = npf.torch.benchmark(..., multiinterpreter=True)

This flag controls whether a fresh interpreter is used instead of forking. Defaults to ``False`` except with Tensorflow/Keras.


.. _npf-cpu-gpu:

Benchmark on CPU or GPU
-----------------------

When benchmarking on CPU or GPU, the API is slightly different. With CPU or GPU, there is no compiled model to benchmark, so instead we need to directly pass a reference to the model class that will be instantiated.

.. note::

   GPU benchmarking is currently only available for PyTorch.

CPU:

.. code:: python

   cpu_reports = npf.cpu.benchmark(YourModelClass, ...)

GPU:

.. code:: python

   gpu_reports = npf.torch.benchmark(YourModelClass, ..., device_type="gpu")


Your model class will be instantiated in a subprocess, so there are some things to keep in mind.

* Your model class must be defined at the top level inside a Python module
   * i.e. don't place your model class definition inside a function or other nested scope
* If your model class has special Python module dependencies, consider importing them inside your class ``__init__``
* If your model class expects constructor arguments, wrap your class so that it has no constructor arguments


Example of a wrapped model class for CPU/GPU benchmarking:

.. code:: python

   class ModelWrapper(torch.nn.Module):
      def __init__(self):
         super().__init__()
         from transformers import AutoModelForSequenceClassification
         model_name = "bert-base-cased"
         self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)
         self.add_module(model_name, self.bert)

      def forward(self, *inputs):
         return self.bert(*inputs)


   reports = npf.torch.benchmark(ModelWrapper, inputs, device_type="gpu")
