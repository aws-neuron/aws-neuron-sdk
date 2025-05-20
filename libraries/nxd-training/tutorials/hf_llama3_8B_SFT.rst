.. _hf_llama3_8B_SFT:

HuggingFace  Llama3.1/Llama3-8B Supervised Fine-tuning
======================================================

In this example, we will compile and finetune pre-trained HF  Llama3.1/Llama3-8B
model on a single instance with the NxD Training library.
The pre-trained Llama3-8B model serves as the foundation, and we will
build upon this solid base by fine-tuning the model to adapt
it to a specific task or dataset.
The example has the following main sections:

.. contents:: Table of contents
   :local:
   :depth: 2

Setting up the environment
--------------------------

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

Please follow this guide on how to install the latest Neuron packages:
`PyTorch Neuron Setup Guide
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx>`_.

Next, we will need to install NxD Training and its dependencies.
Please see the following installation guide for installing NxD Training:
:ref:`NxD Training Installation Guide <nxdt_installation_guide>`.


SFT-YAML Configuration Overview
-------------------------------

You can configure a variety of SFT-specific and model parameters for finetuning through the YAML file.

.. code-block:: yaml

    exp_manager:
        resume_from_checkpoint: /pretrained_ckpt

    data:
        train_dir: /example_datasets/llama3_8b/training.jsonl
        val_dir: /example_datasets/llama3_8b/validation.json
        dev_choose_samples: 2250
        seq_length: 4096
        alignment_strategy:
            sft:
                packing: True
        tokenizer:
            type: /llama3_tokenizer

    model:
        weight_init_only: True


**exp_manager**
    **resume_from_checkpoint**

    Manually set the checkpoint file (pretrained checkpoint) to load from

        * **Type**: str
        * **Default**: ``/pretrained_ckpt``
        * **Required**: True (start with pretrained checkpoint)

**data**

    **train_dir**

    SFT training data - jsonl or arrow file

    As for SFT we use HF style ModelAlignment dataloader, we also use HF style data file paths

        * **Type**: str
        * **Required**: True

    **val_dir**

    SFT validation data - jsonl or arrow file

    As for SFT we use HF style ModelAlignment dataloader, we also use HF style data file paths

        * **Type**: str
        * **Required**: False

    **dev_choose_samples**

    If set, will use that many number of records from the
    head of the dataset instead of using all. Set to null to use full dataset

        * **Type**: integer
        * **Default**: null
        * **Required**: False

    **seq_length**

    Set sequence length for the training job.

        * **Type**: integer
        * **Required**: True

    **alignment_strategy**

    Set only when using finetuning specific algorithms (SFT, DPO, etc) and related hyperparameters
    SFT-specific parameters.

        **sft**
            **packing**

            Appends multiple records in a single record until seq length
            supported by model, if false uses pad tokens to reach seq length.
            Setting it to True increases throughput but might impact accuracy.

                * **Type**: bool
                * **Default**: False
                * **Required**: False

    **tokenizer**
        **type**

        Set tokenizer path/type

            * **Type**: str
            * **Default**: ``/llama3_tokenizer``
            * **Required**: True

 **model**
        **weight_init_only**

        Load only model states and ignore the optim states from ckpt directory

            * **Type**: bool
            * **Default**: True


Download the dataset
--------------------

This tutorial makes use of a preprocessed version of `databricks-dolly` instruction-following
dataset that is stored in S3. The dataset can be downloaded to your cluster or instance
by running the following AWS CLI commands on the head node or your Trn1 instance:

.. code-block:: bash

    export DATA_DIR=~/examples_datasets/llama3_8b
    mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
    aws s3 cp s3://neuron-s3/training_datasets/llama/sft/training.jsonl .  --no-sign-request
    aws s3 cp s3://neuron-s3/training_datasets/llama/sft/validation.jsonl .  --no-sign-request


Download pretrained model checkpoint and tokenizer
--------------------------------------------------

In this tutorial, we will use a pretrained Llama3-8B checkpoint from the original repository.
Follow the steps to download tokenizer and model checkpoint from
the pretraining stage: `<https://llama.meta.com/llama-downloads/>`_

Alternatively, the model checkpoint and tokenizer can also be downloaded
from HuggingFace by following this `guide <https://huggingface.co/meta-llama/Llama-3.1-8B#use-with-llama>`_

You can also directly download and covert the HuggingFace
model checkpoint using :ref:`Direct HuggingFace Model Conversion <checkpoint_conversion>`

Create a folder ``llama3_tokenizer`` and copy the tokenizer contents to it.

Modify the following paths in YAML file based on your specific directory configuration:

1. ``model.model_config``
2. ``exp_manager.resume_from_checkpoint``
3. ``tokenizer.type``
4. ``train_dir`` and ``val_dir``

You can use your custom model, pretrained checkpoint and tokenizer by
modifying the ``hf_llama3_8B_SFT_config.yaml`` file.


Checkpoint Conversion
^^^^^^^^^^^^^^^^^^^^^

Follow this :ref:`Checkpoint Conversion Guide <checkpoint_conversion>` to convert the
HF-style Llama3-8B checkpoint
to NxDT supported format and store it in  ``pretrained_ckpt`` directory.
Modify the config parameter ``exp_manager.resume_from_checkpoint`` path to the
converted pretrained checkpoint path.

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

First, clone the open-source ``neuronx-distributed-training`` library

.. code:: ipython3

   git clone https://github.com/aws-neuron/neuronx-distributed-training
   cd neuronx-distributed-training/examples

Now, ensure that you are using the proper config file in the ``conf/`` directory.
In the ``train.sh`` file, ensure that the ``CONF_FILE`` variable is properly
set to the config for the model you want to use. In our case,
it will be ``hf_llama3_8B_SFT_config``. The default config here is a 8B parameter model,
but users can also add their own ``conf/*.yaml`` files and run different configs and
hyperparameters if desired. Please see :ref:`Config Overview <nxdt_config_overview>`
for examples and usage for the ``.yaml`` config files.

Next, run the following commands to launch an AOT pre-compilation job on your instance:

.. code-block:: bash

    export COMPILE=1
    ./train.sh

The compile output and logs will be shown directly in the terminal
and you will see logs similar to this:

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

The fine-tuning job is launched almost exactly in the same way as the compile job.
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

For issues with NxD Training, please see:
:ref:`NxD Training Known Issues <nxdt_known_issues>`