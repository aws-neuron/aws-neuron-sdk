.. _pytorch-tutorials-torchserve:

BERT TorchServe Tutorial
========================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates the use of `TorchServe <https://pytorch.org/serve>`_ with Neuron, the SDK for Amazon Inf1 instances. By the end of this tutorial, you will understand how TorchServe can be used to serve a model backed by EC2 Inf1 instances. We will use a pretrained BERT-Base model to determine if one sentence is a paraphrase of another.

.. _torchserve-compile:


Run the tutorial
----------------

First run the HuggingFace Pretrained BERT tutorial :ref:`[html] </src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb>` :pytorch-neuron-src:`[notebook] <bert_tutorial/tutorial_pretrained_bert.ipynb>`.


You should now have a compiled ``bert_neuron_b6.pt`` file, which is required going forward.

Open a shell on the instance you prepared earlier, create a new directory named ``torchserve``. Copy your compiled model from the previous tutorial into this new directory.

.. code:: bash

  $ cd torchserve
  $ ls

::

  bert_neuron_b6.pt

Prepare a new Python virtual environment with the necessary Neuron and TorchServe components. Use a virtual environment to keep (most of) the various tutorial components isolated from the rest of the system in a controlled way.

.. code:: bash

  $ python3 -m venv env
  $ . env/bin/activate
  $ pip install -U pip
  $ pip install torch-neuron 'neuron-cc[tensorflow]' --extra-index-url=https://pip.repos.neuron.amazonaws.com
  $ pip install transformers==4.12.5 torchserve==0.5.0 torch-model-archiver==0.5.0

Install the system requirements for TorchServe.

.. code:: bash

  $ sudo apt install openjdk-11-jdk
  $ java -version

::

  openjdk 11.0.11 2021-04-20
  OpenJDK Runtime Environment (build 11.0.11+9-Ubuntu-0ubuntu2.18.04)
  OpenJDK 64-Bit Server VM (build 11.0.11+9-Ubuntu-0ubuntu2.18.04, mixed mode, sharing)

.. code:: bash

  $ javac -version

::

  javac 11.0.11

Verify that TorchServe is now available.

.. code:: bash

  $ torchserve --version

::

  TorchServe Version is 0.5.0


.. _torchserve-setup:

Setup TorchServe
----------------

During this tutorial you will need to download various files onto your instance. The simplest way to accomplish this is to paste the download links provided above each file into a ``wget`` command. (We don't provide the links directly because they are subject to change.) For example, right-click and copy the download link for ``config.json`` shown below.

.. literalinclude:: /src/examples/pytorch/torchserve/config.json
    :language: JSON
    :caption: :download:`config.json </src/torchserve/config.json>`


Now execute the following in your shell:

.. code:: bash

  $ wget <paste link here>
  $ ls

::

  bert_neuron_b6.pt  config.json

Download the `custom handler script <https://pytorch.org/serve/custom_service.html>`_ that will eventually respond to inference requests.

.. literalinclude:: /src/examples/pytorch/torchserve/handler_bert.py
    :language: python
    :caption: :download:`handler_bert.py </src/examples/pytorch/torchserve/handler_bert.py>`
    :linenos:

Next, we need to associate the handler script with the compiled model using ``torch-model-archiver``. Run the following commands in your terminal:

.. code:: bash

  $ mkdir model_store
  $ MAX_LENGTH=$(jq '.max_length' config.json)
  $ BATCH_SIZE=$(jq '.batch_size' config.json)
  $ MODEL_NAME=bert-max_length$MAX_LENGTH-batch_size$BATCH_SIZE
  $ torch-model-archiver --model-name "$MODEL_NAME" --version 1.0 --serialized-file ./bert_neuron_b6.pt --handler "./handler_bert.py" --extra-files "./config.json" --export-path model_store

.. note::

  If you modify your model or a dependency, you will need to rerun the archiver command with the ``-f`` flag appended to update the archive.

The result of the above will be a ``mar`` file inside the ``model_store`` directory.

.. code:: bash

  $ ls model_store

::

  bert-max_length128-batch_size6.mar

This file is essentially an archive associated with a fixed version of your model along with its dependencies (e.g. the handler code).

.. note::

  The version specified in the ``torch-model-archiver`` command can be appended to REST API requests to access a specific version of your model. For example, if your model was hosted locally on port 8080 and named "bert", the latest version of your model would be available at ``http://localhost:8080/predictions/bert``, while version 1.0 would be accessible at ``http://localhost:8080/predictions/bert/1.0``. We will see how to perform inference using this API in Step 6.

Create a `custom config <https://pytorch.org/serve/configuration.html>`_ file to set some parameters. This file will be used to configure the server at launch when we run ``torchserve --start``.

.. literalinclude:: /src/examples/pytorch/torchserve/torchserve.config
    :language: properties
    :caption: :download:`torchserve.config </src/examples/pytorch/torchserve/torchserve.config>`

.. note::

  This will cause TorchServe to bind on all interfaces. For security in real-world applications, you’ll probably want to use port 8443 and `enable SSL <https://pytorch.org/serve/configuration.html#enable-ssl>`_.


.. _torchserve-run:

Run TorchServe
--------------

It's time to start the server. Typically we'd want to launch this in a separate console, but for this demo we’ll just redirect output to a file.

.. code:: bash

  $ torchserve --start --ncs --model-store model_store --ts-config torchserve.config 2>&1 >torchserve.log

Verify that the server seems to have started okay.

.. code:: bash

  $ curl http://127.0.0.1:8080/ping

::

  {
    "status": "Healthy"
  }

.. note::

  If you get an error when trying to ping the server, you may have tried before the server was fully launched. Check ``torchserve.log`` for details.

Use the Management API to instruct TorchServe to load our model.

.. code:: bash

  $ MAX_BATCH_DELAY=5000 # ms timeout before a partial batch is processed
  $ INITIAL_WORKERS=4 # number of models that will be loaded at launch
  $ curl -X POST "http://localhost:8081/models?url=$MODEL_NAME.mar&batch_size=$BATCH_SIZE&initial_workers=$INITIAL_WORKERS&max_batch_delay=$MAX_BATCH_DELAY"

::

  {
    "status": "Model \"bert-max_length128-batch_size6\" Version: 1.0 registered with 4 initial workers"
  }

.. note::

  Any additional attempts to configure the model after the initial curl request will cause the server to return a 409 error. You’ll need to stop/start/configure the server to realize any changes.

The ``MAX_BATCH_DELAY`` is a timeout value that determines how long to wait before processing a partial batch. This is why the handler code needs to check the batch dimension and potentially add padding. TorchServe will instantiate the number of model handlers indicated by ``INITIAL_WORKERS``, so this value controls how many models we will load onto Inferentia in parallel. This tutorial was performed on an inf1.xlarge instance (one Inferentia chip), so there are four NeuronCores available. If you want to control worker scaling more dynamically, `see the docs <https://pytorch.org/serve/management_api.html#scale-workers>`_.

.. warning::
  If you attempt to load more models than NeuronCores available, one of two things will occur. Either the extra models will fit in device memory but performance will suffer, or you will encounter an error on your initial inference. You shouldn't set ``INITIAL_WORKERS`` above the number of NeuronCores. However, you may want to use fewer cores if you are using the :ref:`neuroncore-pipeline` feature.

It looks like everything is running successfully at this point, so it's time for an inference.

Create the ``infer_bert.py`` file below on your instance.

.. literalinclude:: /src/examples/pytorch/torchserve/infer_bert.py
    :language: python
    :caption: :download:`infer_bert.py </src/examples/pytorch/torchserve/infer_bert.py>`
    :linenos:

This script will send a ``batch_size`` number of requests to our model. In this example, we are using a model that estimates the probability that one sentence is a paraphrase of another. The script sends positive examples in the first half of the batch and negative examples in the second half.

Execute the script in your terminal.

.. code:: bash

  $ python infer_bert.py

::

  1 ['paraphrase']
  3 ['not paraphrase']
  4 ['not paraphrase']
  0 ['paraphrase']
  5 ['not paraphrase']
  2 ['paraphrase']

We can see that the first three threads (0, 1, 2) all report ``paraphrase``, as expected. If we instead modify the script to send an incomplete batch and then wait for the timeout to expire, the excess padding results will be discarded.


.. _torchserve-benchmark:

Benchmark TorchServe
--------------------

We've seen how to perform a single batched inference, but how many inferences can we process per second? A separate upcoming tutorial will document performance tuning to maximize throughput. In the meantime, we can still perform a simple naïve stress test. The code below will spawn 64 worker threads, with each thread repeatedly sending a full batch of data to process. A separate thread will periodically print throughput and latency measurements.

.. literalinclude:: /src/examples/pytorch/torchserve/benchmark_bert.py
    :language: python
    :caption: :download:`benchmark_bert.py </src/examples/pytorch/torchserve/benchmark_bert.py>`
    :linenos:

Run the benchmarking script.

.. code:: bash

  $ python benchmark_bert.py

::

  pid 26980: current throughput 0.0, latency p50=0.000 p90=0.000
  pid 26980: current throughput 584.1, latency p50=0.099 p90=0.181
  pid 26980: current throughput 594.2, latency p50=0.100 p90=0.180
  pid 26980: current throughput 598.8, latency p50=0.095 p90=0.185
  pid 26980: current throughput 607.9, latency p50=0.098 p90=0.182
  pid 26980: current throughput 608.6, latency p50=0.096 p90=0.181
  pid 26980: current throughput 611.3, latency p50=0.096 p90=0.185
  pid 26980: current throughput 610.2, latency p50=0.096 p90=0.185
  ...

**Congratulations!** By now you should have successfully served a batched model over TorchServe.



