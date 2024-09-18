.. _hf_llama3_8B_pretraining:

HuggingFace Llama3-8B Pretraining
=================================

In this example, we will compile and train a HF Llama3-8B model on a single instance
with the ``NxD Training (NxDT)`` library.
The example has the following main sections:

.. contents:: Table of contents
   :local:
   :depth: 2

Setting up the environment
--------------------------

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

Once you have launched a Trn1 instance,
please follow this guide on how to install the latest Neuron packages:
`PyTorch Neuron Setup Guide
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx>`_.

Next, we will need to install ``NxDT`` and its dependencies.
Please see the following installation guide for installing ``NxDT``:
:ref:`NxDT Installation Guide <nxdt_installation_guide>`


Download the dataset
--------------------

This tutorial makes use of a preprocessed and tokenized Wikicorpus
dataset that is stored in S3.
The dataset can be downloaded to your cluster or instance by running
the following AWS CLI commands on the head node or your Trn1 instance:

.. code-block:: bash

    export DATA_DIR=~/examples_datasets/
    mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
    aws s3 cp s3://neuron-s3/training_datasets/llama3/wikicorpus_llama3_tokenized_8k .  --no-sign-request
    aws s3 cp s3://neuron-s3/training_datasets/llama3/llama3_8B_config.json ~/config.json  --no-sign-request


Pre-compile the model
---------------------

By default, PyTorch Neuron uses a just in time (JIT) compilation flow that sequentially
compiles all of the neural network compute graphs as they are encountered during a training job.
The compiled graphs are cached in a local compiler cache so that subsequent training jobs
can leverage the compiled graphs and avoid compilation
(so long as the graph signatures and Neuron version have not changed).

An alternative to the JIT flow is to use the included ``neuron_parallel_compile``
command to perform ahead of time (AOT) compilation. In the AOT compilation flow,
the compute graphs are first identified and extracted during a short simulated training run,
and the extracted graphs are then compiled and cached using parallel compilation,
which is considerably faster than the JIT flow.

First, ensure that you are using the proper config file in the ``conf/`` directory.
In the ``train.sh`` file, ensure that the ``CONF_FILE`` variable is properly
set to the config for the model you want to use. In our case,
it will be ``hf_llama3_8B_config``. The default config here is a 8B parameter model,
but users can also add their own ``conf/*.yaml`` files and run different configs and
hyperparameters if desired. Please see :ref:`Config Overview <nxdt_config_overview>`
for examples and usage for the ``.yaml`` config files.

Next, run the following commands to launch an AOT pre-compilation job on your instance:

.. code-block:: bash

    cd ~/neuronx-distributed-training/examples
    export COMPILE=1
    ./train.sh

The compile output and logs will be shown directly in the terminal
and you will see a message similar to this:

.. code-block:: bash

    2024-08-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total graphs: 22
    2024-08-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total successful compilations: 22
    2024-08-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total failed compilations: 0

Then, you know your compilation has successfully completed.

.. note::
    The number of graphs will differ based on package versions, models, and other factors.
    This is just an example.


Training the model
------------------

The pre-training job is launched almost exactly the same as the compile job.
We now turn off the ``COMPILE`` environment variable and
run the same training script to start pre-training.

On a single instance:

.. code-block:: bash

    export COMPILE=0
    ./train.sh

Once the model is loaded onto the Trainium accelerators and training has commenced,
you will begin to see output indicating the job progress:

Example:

.. code-block:: bash

    Epoch 0:   0%|          | 189/301501 [59:12<1573:03:24, 18.79s/it, loss=7.75, v_num=3-16, reduced_train_loss=7.560, global_step=188.0, consumed_samples=24064.0]
    Epoch 0:   0%|          | 190/301501 [59:30<1572:41:13, 18.79s/it, loss=7.74, v_num=3-16, reduced_train_loss=7.560, global_step=189.0, consumed_samples=24192.0]
    Epoch 0:   0%|          | 191/301501 [59:48<1572:21:28, 18.79s/it, loss=7.73, v_num=3-16, reduced_train_loss=7.910, global_step=190.0, consumed_samples=24320.0]


Monitoring Training
-------------------

Tensorboard monitoring
^^^^^^^^^^^^^^^^^^^^^^

In addition to the text-based job monitoring described in the previous section,
you can also use standard tools such as TensorBoard to monitor training job progress.
To view an ongoing training job in TensorBoard, you first need to identify the
experiment directory associated with your ongoing job.
This will typically be the most recently created directory under
``~/neuronx-distributed-training/examples/nemo_experiments/hf_llama3_8B/``.
Once you have identifed the directory, cd into it, and then launch TensorBoard:

.. code-block:: bash

    cd ~/neuronx-distributed-training/examples/nemo_experiments/hf_llama3_8B/
    tensorboard --logdir ./

With TensorBoard running, you can then view the TensorBoard dashboard by browsing to
``http://localhost:6006`` on your local machine. If you cannot access TensorBoard at this address,
please make sure that you have port-forwarded TCP port 6006 when SSH'ing into the head node,

.. code-block:: bash

    ssh -i YOUR_KEY.pem ubuntu@HEAD_NODE_IP_ADDRESS -L 6006:127.0.0.1:6006

neuron-top / neuron-monitor / neuron-ls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `neuron-top <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-top-user-guide.html>`_
tool can be used to view useful information about NeuronCore utilization, vCPU and RAM utilization,
and loaded graphs on a per-node basis. To use neuron-top during on ongoing training job, run ``neuron-top``:

.. code-block:: bash

    ssh compute1-dy-queue1-i1-1  # to determine which compute nodes are in use, run the squeue command
    neuron-top

Similarly, once you are logged into one of the active compute nodes,
you can also use other Neuron tools such as
`neuron-monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html>`_
and `neuron-ls <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html>`_
to capture performance and utilization statistics and to understand NeuronCore allocation.

Troubleshooting Guide
---------------------

For issues with ``NxDT``, please see:
:ref:`NxDT Known Issues <nxdt_known_issues>`