.. _pytorch-tutorials-torchserve-neuronx:

BERT TorchServe Tutorial
========================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------
Update 10/02:This tutorial is currently broken and the AWS Neuron team is working on providing the fix.

This tutorial demonstrates the use of `TorchServe <https://pytorch.org/serve>`_ with Neuron, the SDK for EC2 Inf2 and Trn1 instances. By the end of this tutorial, you will understand how TorchServe can be used to serve a model backed by EC2 Inf2/Trn1 instances. We will use a pretrained BERT-Base model to determine if one sentence is a paraphrase of another.

.. _torchserve-compile:


Run the tutorial
----------------

Open a terminal, log into your remote instance, and activate a Pytorch virtual environment setup (see the:ref:`Install PyTorch Neuron <setup-torch-neuronx>`). To complete this tutorial, you will also need a compiled BERT model. You can run :download:`trace_bert_neuronx.py </src/examples/pytorch/torchserve/trace_bert_neuronx.py>` to obtain a traced BERT model.

You should now have a compiled ``bert_neuron_b6.pt`` file, which is required going forward.

Open a shell on the instance you prepared earlier, create a new directory named ``torchserve``. Copy your compiled model from the previous tutorial into this new directory.

.. code:: bash

  cd torchserve
  ls

::

  bert_neuron_b6.pt

Prepare a new Python virtual environment with the necessary Neuron and TorchServe components. Use a virtual environment to keep (most of) the various tutorial components isolated from the rest of the system in a controlled way.

.. code:: bash

  pip install transformers==4.26.0 torchserve==0.7.0 torch-model-archiver==0.7.0 captum==0.6.0

Install the system requirements for TorchServe.

.. tab-set::

   .. tab-item:: Amazon Linux 2 DLAMI Base

      .. code-block:: bash

        sudo yum -y install jq java-11-amazon-corretto-headless
        sudo alternatives --config java
        sudo alternatives --config javac

   .. tab-item:: Ubuntu 20 DLAMI Base

      .. code-block:: bash

        sudo apt -y install openjdk-11-jdk


.. code:: bash

  java -version

::

  openjdk version "11.0.17" 2022-10-18
  OpenJDK Runtime Environment (build 11.0.17+8-post-Ubuntu-1ubuntu218.04)
  OpenJDK 64-Bit Server VM (build 11.0.17+8-post-Ubuntu-1ubuntu218.04, mixed mode, sharing)

.. code:: bash

  javac -version

::

  javac 11.0.17

Verify that TorchServe is now available.

.. code:: bash

  torchserve --version

::

  TorchServe Version is 0.7.0


.. _torchserve-setup:

Setup TorchServe
----------------

During this tutorial you will need to download a few files onto your instance. The simplest way to accomplish this is to paste the download links provided above each file into a ``wget`` command. (We don't provide the links directly because they are subject to change.) For example, right-click and copy the download link for ``config.json`` shown below.

.. literalinclude:: /src/examples/pytorch/torchserve/config.json
    :language: JSON
    :caption: :download:`config.json </src/examples/pytorch/torchserve/config.json>`


Now execute the following in your shell:

.. code:: bash

  wget <paste link here>
  ls

::

  bert_neuron_b6.pt  config.json

Download the `custom handler script <https://pytorch.org/serve/custom_service.html>`_ that will eventually respond to inference requests.

.. literalinclude:: /src/examples/pytorch/torchserve/handler_bert_neuronx.py
    :language: python
    :caption: :download:`handler_bert_neuronx.py </src/examples/pytorch/torchserve/handler_bert_neuronx.py>`
    :linenos:

Next, we need to associate the handler script with the compiled model using ``torch-model-archiver``. Run the following commands in your terminal:

.. code:: bash

  mkdir model_store
  MAX_LENGTH=$(jq '.max_length' config.json)
  BATCH_SIZE=$(jq '.batch_size' config.json)
  MODEL_NAME=bert-max_length$MAX_LENGTH-batch_size$BATCH_SIZE
  torch-model-archiver --model-name "$MODEL_NAME" --version 1.0 --serialized-file ./bert_neuron_b6.pt --handler "./handler_bert_neuronx.py" --extra-files "./config.json" --export-path model_store

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

  torchserve --start --ncs --model-store model_store --ts-config torchserve.config 2>&1 >torchserve.log

Verify that the server seems to have started okay.

.. code:: bash

  curl http://127.0.0.1:8080/ping

::

  {
    "status": "Healthy"
  }

.. note::

  If you get an error when trying to ping the server, you may have tried before the server was fully launched. Check ``torchserve.log`` for details.

Use the Management API to instruct TorchServe to load our model.

First, determine the number of NeuronCores available based on your instance size.

.. tab-set::

   .. tab-item:: Inf2

      .. list-table::
        :header-rows: 1

        * - Instance Size
          - # of NeuronCores
        * - xlarge
          - 2
        * - 8xlarge
          - 2
        * - 24xlarge
          - 12
        * - 48xlarge
          - 24

   .. tab-item:: Trn1

      .. list-table::
        :header-rows: 1

        * - Instance Size
          - # of NeuronCores
        * - 2xlarge
          - 2
        * - 32xlarge
          - 32


.. code:: bash

  MAX_BATCH_DELAY=5000 # ms timeout before a partial batch is processed
  INITIAL_WORKERS=<number of NeuronCores from table above>
  curl -X POST "http://localhost:8081/models?url=$MODEL_NAME.mar&batch_size=$BATCH_SIZE&initial_workers=$INITIAL_WORKERS&max_batch_delay=$MAX_BATCH_DELAY"

::

  {
    "status": "Model \"bert-max_length128-batch_size6\" Version: 1.0 registered with X initial workers"
  }


.. warning::
  You shouldn't set ``INITIAL_WORKERS`` above the number of NeuronCores. If you attempt to load more models than NeuronCores available, one of two things will occur. Either the extra models will fit in device memory but performance will suffer, or you will encounter an error on your initial inference. However, you may want to use fewer cores if you are using the :ref:`neuroncore-pipeline` feature.


.. note::

  Any additional attempts to configure the model after the initial curl request will cause the server to return a 409 error. You’ll need to stop/start/configure the server to realize any changes.

The ``MAX_BATCH_DELAY`` is a timeout value that determines how long to wait before processing a partial batch. This is why the handler code needs to check the batch dimension and potentially add padding. TorchServe will instantiate the number of model handlers indicated by ``INITIAL_WORKERS``, so this value controls how many models we will load onto Inferentia in parallel. If you want to control worker scaling more dynamically, `see the docs <https://pytorch.org/serve/management_api.html#scale-workers>`_.

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

  python benchmark_bert.py

::

  pid 1214554: current throughput 0.0, latency p50=0.000 p90=0.000
  pid 1214554: current throughput 713.9, latency p50=0.071 p90=0.184
  pid 1214554: current throughput 737.9, latency p50=0.071 p90=0.184
  pid 1214554: current throughput 731.6, latency p50=0.068 p90=0.192
  pid 1214554: current throughput 732.2, latency p50=0.070 p90=0.194
  pid 1214554: current throughput 733.9, latency p50=0.070 p90=0.187
  pid 1214554: current throughput 739.3, latency p50=0.071 p90=0.184
  ...

.. note::

  Your throughput numbers may differ from these based on instance type and size.

**Congratulations!** By now you should have successfully served a batched model over TorchServe.

You can now shutdown torchserve.

.. code:: bash

  torchserve --stop

