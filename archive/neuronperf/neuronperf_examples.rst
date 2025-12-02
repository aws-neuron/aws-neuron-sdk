.. _neuronperf_examples:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.
   :date-modified: 12-02-2025

NeuronPerf Examples
===================

This page walks through several examples of using NeuronPerf, starting with the simplest way---using a compiled model. We will also see how we can use NeuronPerf to perform a hyperparameter search, and manage the artifacts produced, as well as our results.

Benchmark a Compiled Model
--------------------------

This example assumes you have already compiled your model for Neuron and saved it to disk.
You will need to adapt the batch size, input shape, and filename for your model.

.. code:: python

   import torch  # or tensorflow, mxnet

   import neuronperf as npf
   import neuronperf.torch  # or tensorflow, mxnet

   # Construct dummy inputs
   batch_sizes = 1
   input_shape = (batch_sizes, 3, 224, 224)
   inputs = torch.ones(input_shape)  # or numpy array for TF, MX

   # Benchmark and save results
   reports = npf.torch.benchmark("your_model_file.pt", inputs, batch_sizes)
   npf.print_reports(reports)
   npf.write_json(reports)


.. code:: bash

   INFO:neuronperf.benchmarking - Benchmarking 'your_model_file.pt', ~8.0 minutes remaining.
   throughput_avg    latency_ms_p50    latency_ms_p99    n_models          pipeline_size     workers_per_model batch_size        model_filename
   296766.5          0.003             0.003             1                 1                 1                 1                 your_model_file.pt
   3616109.75        0.005             0.008             24                1                 1                 1                 your_model_file.pt
   56801.0           0.035             0.04              1                 1                 2                 1                 your_model_file.pt
   3094419.4         0.005             0.051             24                1                 2                 1                 your_model_file.pt


Let's suppose you only wish to test two specific configurations. You wish to benchmark  1 model and 1 worker thread, and also with 2 worker threads for 15 seconds each. The call to ``benchmark`` becomes:

.. code:: python

   reports = npf.torch.benchmark(filename, inputs, batch_sizes, n_models=1, workers_per_model=[1, 2], duration=15)

You can also add a custom model name to reports.

.. code:: python

   reports = npf.torch.benchmark(..., model_name="MyFancyModel")

See the :ref:`neuronperf_benchmark_guide` for further details.


Benchmark a Model from Source
-----------------------------

In this example, we define, compile, and benchmark a simple (dummy) model using PyTorch.

We'll assume you already have a PyTorch model compiled for Neuron with the filename ``model_neuron_b1.pt``. Furthermore, let's assume the model was traced with a batch size of 1, and has an input shape of (3, 224, 224).

.. literalinclude:: test_simple_pt.py
    :language: python
    :caption: :download:`test_simple_pt.py <test_simple_pt.py>`
    :linenos:


.. code:: bash

   (aws_neuron_pytorch_p36) ubuntu@ip-172-31-11-122:~/tmp$ python test_simple_pt.py
   INFO:neuronperf.benchmarking - Benchmarking 'model_neuron_b1.pt', ~8.0 minutes remaining.
   throughput_avg    latency_ms_p50    latency_ms_p99    n_models          pipeline_size     workers_per_model batch_size        model_filename
   296766.5          0.003             0.003             1                 1                 1                 1                 model_neuron_b1.pt
   3616109.75        0.005             0.008             24                1                 1                 1                 model_neuron_b1.pt
   56801.0           0.035             0.04              1                 1                 2                 1                 model_neuron_b1.pt
   3094419.4         0.005             0.051             24                1                 2                 1                 model_neuron_b1.pt

Compile and Benchmark a Model
-----------------------------

Here is an end-to-end example of compiling and benchmarking a ResNet-50 model from ``torchvision``.

.. literalinclude:: test_resnet50_pt.py
    :language: python
    :caption: :download:`test_resnet50_pt.py <test_resnet50_pt.py>`
    :linenos:


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


Please refer to :ref:`npf-cpu-gpu` for details and an example of providing your model class.
