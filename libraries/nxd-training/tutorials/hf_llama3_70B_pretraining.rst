.. _hf_llama3_70B_pretraining:

HuggingFace Llama3.1/Llama3-70B Pretraining
=================================

In this example, we will compile and train a HuggingFace Llama3.1/Llama3-70B model
on multiple trn1 or newly launched trn2 instances using ParallelCluster with the ``NxD Training (NxDT)`` library.
The example has the following main sections:

.. contents:: Table of contents
   :local:
   :depth: 2

Setting up the environment
--------------------------

ParallelCluster Setup
^^^^^^^^^^^^^^^^^^^^^

In this example, we will use 16 trn1.32xlarge instances or 8 trn2.48xlarge instances with ParallelCluster.
Please follow the instructions here to create a cluster:
`Train your model on ParallelCluster
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`_

ParallelCluster automates the creation of trainium clusters,
and provides the Slurm job management system for scheduling and managing distributed training jobs.
Please note that the home directory on your ParallelCluster
head node will be shared with all of the worker nodes via NFS.

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

Once you have launched ParallelCluster,
please follow this guide on how to install the latest Neuron packages:
`PyTorch Neuron Setup Guide
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx>`_.

Next, we will need to install ``NxDT`` and its dependencies.
Please see the following installation guide for installing ``NxDT``:
:ref:`NxDT Installation Guide <nxdt_installation_guide>`


Download the dataset
--------------------

Let's download training-data scripts for our experiments

.. code:: ipython3

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/get_dataset.py

Then download ``config.json`` file:

For Llama-3.1-70B:

.. code-block:: bash

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/70B_config_llama3.1/config.json ~/

For Llama-3-70B:

.. code-block:: bash

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/70B_config_llama3/config.json ~/

To tokenize the data, we must request the tokenizer from Hugging Face and Meta by following the
instructions at the following link: `HuggingFace Llama 3.1 70B Model <https://huggingface.co/meta-llama/Meta-Llama-3.1-70B>`__ . 

Use of the Llama models is governed by the Meta license.
In order to download the model weights and tokenizer, please visit the above website
and accept their License before requesting access. After access has been granted,
you may use the following python3 script along with your own hugging face token to download and save the tokenizer.


.. code:: ipython3

   from huggingface_hub import login
   from transformers import AutoTokenizer

   login(token='your_own_hugging_face_token')

   tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-70B')  
   # For llama3 uncomment line below
   # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B')

   tokenizer.save_pretrained(".")

For Llama3.1/Llama3, make sure your base directory has the following files:

.. code:: ipython3

   './tokenizer_config.json', './special_tokens_map.json', './tokenizer.json'

Next, let’s download and pre-process the dataset:

.. code:: ipython3

   mkdir ~/examples_datasets/
   python3 get_dataset.py --llama-version 3


`Note:` In case you see an error of the following form when downloading data: ``huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'. Use `repo_type` argument if needed.`` 
This could be because of a stale cache. Try deleting the cache using: 

.. code:: ipython3

   sudo rm -rf ~/.cache/


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
it will be ``hf_llama3_70B_config.yaml`` for training on trn1 cluster, and ``hf_llama3_70B_trn2_config.yaml`` for trn2.

In this tutorial, we will train Llama3-70B model on multiple compute nodes. For training on trn1, please make sure ``hf_llama3_70B_config`` has the right configuration:

.. code-block:: bash

    trainer:
      devices: 32
      num_nodes: 16

For pretraining on trn2, ``hf_llama3_70B_trn2_config`` would contain:

.. code-block:: bash

    trainer:
      devices: 64
      lnc: 2 # default for trn2 workloads
      num_nodes: 8

On trn2 instances, the configuration `lnc: 2` indicates that there is a 2-to-1 mapping between logical Neuron Core (lnc) and physical Neuron Core.
Another supported configuration is `lnc: 1`, in which case each node would expose 128 logical devices.

The default config here is a 70B parameter model,
but users can also add their own ``conf/*.yaml`` files and run different configs and
hyperparameters if desired. Please see :ref:`Config Overview <nxdt_config_overview>`
for examples and usage for the ``.yaml`` config files.

On trn1 cluster, run the following commands to launch an AOT pre-compilation job on your instance:

.. code-block:: bash

    export COMPILE=1
    export CONF_FILE=hf_llama3_70B_config
    sbatch --exclusive \
        --nodes 16 \
        --cpus-per-task 128 \
        --wrap="srun ./train.sh"

On trn2 cluster, run the following:

.. code-block:: bash

    export COMPILE=1
    export CONF_FILE=hf_llama3_70B_trn2_config
    sbatch --exclusive \
        --nodes 8 \
        --cpus-per-task 128 \
        --wrap="srun ./train.sh"


Once you have launched the precompilation job, run the squeue command to view the
Slurm job queue on your cluster. If you have not recently run a job on your cluster,
it may take 4-5 minutes for the requested trn1.32xlarge or trn2.48xlarge nodes nodes to
be launched and initialized.
Once the job is running, squeue should show output similar to the following:


.. code-block:: bash

    JOBID  PARTITION  NAME      USER    ST  TIME  NODES NODELIST(REASON)
    7      compute1   wrap      ubuntu  R   5:11  16    compute1-st-queue1-i1-[1-16]

You can view the output of the precompilation job by examining the file named
``slurm-ZZ.out``,
where ZZ represents the JOBID of your job in the squeue output above.

.. code-block:: bash

    tail -f slurm-7.out

Once the precompilation job is complete, just like the above output
you should see a message similar to the following in the logs:

.. code-block:: bash

    2024-11-07 09:57:13.000144:  39810  INFO ||NEURON_PARALLEL_COMPILE||: Total graphs: 36
    2024-11-07 09:57:13.000144:  39810  INFO ||NEURON_PARALLEL_COMPILE||: Total successful compilations: 36
    2024-11-07 09:57:13.000144:  39810  INFO ||NEURON_PARALLEL_COMPILE||: Total failed compilations: 0

At this point, you can press ``CTRL-C`` to exit the tail command.

.. note::
    The number of graphs will differ based on package versions, models, and other factors.
    This is just an example.


Training the model
------------------

You can launch pre-training job similar to compilation by using the same
training script but now turning off the ``COMPILE`` environment variable

On trn1 ParallelCluster:

.. code-block:: bash

    export COMPILE=0
    export CONF_FILE=hf_llama3_70B_config
    sbatch --exclusive \
        --nodes 16 \
        --cpus-per-task 128 \
        --wrap="srun ./train.sh"

On trn2 ParallelCluster:

.. code-block:: bash

    export COMPILE=0
    export CONF_FILE=hf_llama3_70B_trn2_config
    sbatch --exclusive \
        --nodes 8 \
        --cpus-per-task 128 \
        --wrap="srun ./train.sh"

As outlined above, you can again use the ``squeue`` command to view the job queue,
and also monitor the job in the same way with the ``tail`` command to see the training logs.
Once the model is loaded onto the Trainium accelerators and training has commenced,
you will begin to see output indicating the job progress:

Example:

.. code-block:: bash

    Epoch 0:   3%|▎         | 3/91 [16:05<7:52:06, 321.89s/it, loss=6.7, v_num=2, reduced_train_loss=13.40, lr=7.5e-9, parameter_norm=5536.0, global_step=1.000, consumed_samples=2048.0]
    Epoch 0:   3%|▎         | 3/91 [16:05<7:52:06, 321.89s/it, loss=4.47, v_num=2, reduced_train_loss=13.40, lr=7.5e-9, parameter_norm=5536.0, global_step=2.000, consumed_samples=3072.0]
    Epoch 0:   4%|▍         | 4/91 [21:20<7:44:18, 320.22s/it, loss=4.47, v_num=2, reduced_train_loss=13.40, lr=7.5e-9, parameter_norm=5536.0, global_step=2.000, consumed_samples=3072.0]
    Epoch 0:   4%|▍         | 4/91 [21:20<7:44:18, 320.22s/it, loss=3.35, v_num=2, reduced_train_loss=13.40, lr=7.5e-9, parameter_norm=5536.0, global_step=3.000, consumed_samples=4096.0]


.. note::
    The convergence is for demonstration and would differ based on instance type, model, and other factors.


Monitoring Training
-------------------

Tensorboard monitoring
^^^^^^^^^^^^^^^^^^^^^^

In addition to the text-based job monitoring described in the previous section,
you can also use tools such as TensorBoard to monitor training job progress.
To view an ongoing training job in TensorBoard, you first need to identify the
experiment directory associated with your ongoing job.
This will typically be the most recently created directory under
``~/neuronx-distributed-training/examples/nemo_experiments/hf_llama/``.
Once you have identifed the directory, ``cd`` into it, and then launch TensorBoard:

.. code-block:: bash

    cd ~/neuronx-distributed-training/examples/nemo_experiments/hf_llama/8/
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

    ssh compute1-st-queue1-i1-1  # to determine which compute nodes are in use, run the squeue command
    neuron-top

Similarly, once you are logged into one of the active compute nodes,
you can also use other Neuron tools such as
`neuron-monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html>`_
and `neuron-ls <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html>`_
to capture performance and utilization statistics and to understand NeuronCore allocation.


Continual Pre-training with Downloaded Meta Model Weights
---------------------------------------------------------
If you want to perform contiual pre-training using the model weights provided by Meta, follow these steps:

Ensure you have the ``config.json`` file, which should have been downloaded as described in the `Download the dataset`_ section.


Download the model and convert the ``state_dict`` to NxDT checkpoint format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the conversion scripts described in the :ref:`Checkpoint Conversion <checkpoint_conversion>`. 
Mention the ``hf_model_name`` argument to specify the HuggingFace model identifier for
the model you want to download and convert the checkpoint to NxDT format.

Run the following to download the model and convert the ``state_dict`` to NxDT sharded checkpoint.

On trn1 cluster:

.. code-block:: bash

   python3 ./checkpoint_converter_scripts/checkpoint_converter.py \
     --model_style hf \
     --hf_model_name meta-llama/Meta-Llama-3-70B \
     --hw_backend trn1 \
     --tp_size 32 --pp_size 8 --n_layers 80 \
     --output_dir /fsx/pretrained_weight/ \
     --convert_from_full_state --save_xser True \
     --kv_size_multiplier 4 --qkv_linear True \
     --config ~/config.json

On trn2 cluster:

.. code-block:: bash

   python3 ./checkpoint_converter_scripts/checkpoint_converter.py \
     --model_style hf \
     --hf_model_name meta-llama/Meta-Llama-3-70B \
     --hw_backend trn2 \
     --tp_size 32 --pp_size 4 --n_layers 80 \
     --output_dir /fsx/pretrained_weight/ \
     --convert_from_full_state --save_xser True \
     --kv_size_multiplier 4 --qkv_linear True \
     --config ~/config.json


.. note::
    This conversion process requires larger host memory. Please run it on a trn1.32xlarge or trn2.48xlarge compute node. 
    In this example, the converted model is stored on FSx for Lustre to be accessed by all compute nodes.

Start the continual training job by loading converted checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to start the continual training job with loading this converted model as initial weights, please update the config file (``hf_llama3_70B_config.yaml`` or ``hf_llama3_70B_trn2_config.yaml``)  as below:

.. code-block:: bash

    exp_manager:
    .
    .
      resume_from_checkpoint: /fsx/pretrained_weight/ # manually set the checkpoint file to load from
    .
    .
    model:
      # Miscellaneous
      use_cpu_initialization: False # Init weights on the CPU (slow for large models) 
      weight_init_only: True 

Compared to initial pre-training loss value, you should see lower loss value when the training starts with Meta's model weights. Logs for one such sample run look like below.

.. code-block:: bash

    Epoch 0:   3%|▎         | 3/91 [16:09<7:53:59, 323.17s/it, loss=0.834, v_num=7, reduced_train_loss=1.670, lr=7.5e-9, parameter_norm=4736.0, global_step=1.000, consumed_samples=2048.0]
    Epoch 0:   3%|▎         | 3/91 [16:09<7:53:59, 323.17s/it, loss=0.556, v_num=7, reduced_train_loss=1.670, lr=7.5e-9, parameter_norm=4736.0, global_step=2.000, consumed_samples=3072.0]
    Epoch 0:   4%|▍         | 4/91 [21:25<7:46:02, 321.41s/it, loss=0.556, v_num=7, reduced_train_loss=1.670, lr=7.5e-9, parameter_norm=4736.0, global_step=2.000, consumed_samples=3072.0]
    Epoch 0:   4%|▍         | 4/91 [21:25<7:46:02, 321.41s/it, loss=0.417, v_num=7, reduced_train_loss=1.670, lr=7.5e-9, parameter_norm=4736.0, global_step=3.000, consumed_samples=4096.0]


Pretraining with Context Paralellism
------------------------------------

To run pretraining with context parallelism, use the following yaml config file: ``hf_llama3_70B_CP_config.yaml``.
This YAML file has the following changes to enable context parallelism:


.. code-block:: yaml

    distributed_strategy:
        context_parallel_size: 2

    fusions:
        flash_attention: False
        ring_attention: True


**distributed_strategy**
    **context_parallel_size**

    Context parallel degree to be used for sharding sequence.

    * **Type**: int
    * **Required**: False
    * **Default**: 1


**fusions**
    **ring_attention**

    Setting this flag to ``True`` will use the ring attention module for
    both forward and backward.
    This parameter must be true when context parallel is
    ```context_parallel_size`` is greater than 1.

    * **Type**: bool
    * **Required**: False


In the config file, ``context_parallel_size`` is set to the desired degree, and as
context parallelism leverages ring attention instead of flash attention, we set ``ring_attention: True``,
and ``flash_attention: False``.

Context parallelism currently supports sequence lengths up to 32k and is supported on TRN1.

Compile with:

.. code-block:: bash

    export COMPILE=1
    export CONF_FILE=hf_llama3_70B_CP_config
    sbatch --exclusive \
        --nodes 16 \
        --cpus-per-task 128 \
        --wrap="srun ./train.sh"

and pre-training with:

.. code-block:: bash

    export COMPILE=0
    export CONF_FILE=hf_llama3_70B_CP_config
    sbatch --exclusive \
        --nodes 16 \
        --cpus-per-task 128 \
        --wrap="srun ./train.sh"


Troubleshooting Guide
---------------------

For issues with ``NxDT``, please see:
:ref:`NxDT Known Issues <nxdt_known_issues>`
