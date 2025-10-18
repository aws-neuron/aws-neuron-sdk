.. _hf_llama3_8B_DPO_ORPO:

HF Llama3.1/Llama3-8B Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO) based Fine-tuning (Beta)
=================================================================================================================================

In this example, we will show how to compile and finetune a pre-trained
HF Llama3.1/Llama3-8B model on a single instance with the ``NxD Training (NxDT)`` library
using `Direct Preference Optimization (DPO) <https://arxiv.org/pdf/2305.18290>`_ and
`Odds Ratio Preference Optimization (ORPO) <https://arxiv.org/abs/2403.07691>`_
based fine-tuning. The pre-trained Llama3-8B model serves as the foundation, and we will
build upon this base by fine-tuning and aligning the model to adapt
it to a specific task or dataset.
The example has the following main sections:

.. contents:: Table of contents
   :local:
   :depth: 2

Setting up the environment
--------------------------

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

Once you have launched a Trn1 instance,
Please follow this guide on how to install the latest Neuron packages:
`PyTorch Neuron Setup Guide
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/torch-neuronx.html#setup-torch-neuronx>`_.

Next, we will need to install ``NxDT`` and its dependencies.
Please see the following installation guide for installing ``NxDT``:
:ref:`NxDT Installation Guide <nxdt_installation_guide>`.

For DPO and ORPO tests, We have to first install ``requirements.txt`` and then install ``alignment_requirements.txt``. We can use the following commands for the same:

.. code-block:: shell

    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/master/requirements.txt
    pip install -r requirements.txt
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/master/alignment_requirements.txt
    pip install -r alignment_requirements.txt

DPO-YAML Configuration Overview
-------------------------------

You can configure a variety of DPO-specific and model parameters for finetuning through the YAML file.

.. code-block:: yaml

    exp_manager:
        resume_from_checkpoint: /pretrained_ckpt

    data:
        train_dir: /example_datasets/llama3_8b/data_dpo.jsonl
        val_dir: null
        dev_choose_samples: null
        seq_length: 4096
        tokenizer:
            type: /llama3_tokenizer

    model:
        weight_init_only: True

    model_alignment_strategy:
        dpo:
            kl_beta: 0.01
            loss_type: sigmoid
            max_prompt_length: 2048
            precompute_ref_log_probs: True
            truncation_mode: keep_start


**exp_manager**
    **resume_from_checkpoint**

    Manually set the checkpoint file (pretrained/post-SFT checkpoint) to load from

        * **Type**: str
        * **Default**: ``/pretrained_ckpt``
        * **Required**: True (start with pretrained checkpoint)

**data**
    **train_dir**

    DPO training data - jsonl or arrow file

    As for DPO we use HF style ModelAlignment dataloader, we also use HF style data file paths

        * **Type**: str
        * **Required**: True

    **val_dir**

    DPO validation data - jsonl or arrow file

    As for DPO we use HF style ModelAlignment dataloader, we also use HF style data file paths

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
    For DPO, it is total sequence length of prompt and (chosen/rejected) response concatenated together

        * **Type**: integer
        * **Required**: True

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

 **model_alignment_strategy**

    Set only when using finetuning specific algorithms (SFT, DPO, etc) and and parameter-efficient
    fine-tuning methods like LoRA (Low-Rank Adaptation).

        **dpo**
            Direct Preference Optimization (DPO) specific parameters.

            **kl_beta**

            KL-divergence beta to control divergence of policy model from reference model

                * **Type**: float
                * **Default**: 0.01
                * **Required**: True

            **loss_type**

            Currently support sigmoid version of optimized DPO loss

                * **Type**: str
                * **Default**: ``sigmoid``
                * **Required**: True

            **max_prompt_length**

            Set maximum length of prompt in the concatenated prompt and (chosen/rejected) response input

                * **Type**: integer
                * **Required**: True

            **precompute_ref_log_probs**

            To enable precomputation of reference model log probabilities using pre-fit hook,
            False is not supported currently

                * **Type**: bool
                * **Required**: True

            **truncation_mode**

            To define how to truncate if size (prompt+response) exceeds seq_length
            options: ["keep_start", "keep_end"]

                * **Type**: str
                * **Default**: ``keep_start```
                * **Required**: True

ORPO-YAML Configuration Overview
--------------------------------

Here we show the ORPO-specific model parameters which can be configured
for finetuning through the YAML file.
And below we explain the parameters that are new as compared to DPO-specific
parameters.

.. code-block:: yaml

    exp_manager:
        checkpoint_callback_params:
            every_n_train_steps: 10
        resume_from_checkpoint: /pretrained_ckpt

    data:
        train_dir: /example_datasets/llama3_8b/data_orpo.jsonl
        val_dir: null
        dev_choose_samples: null
        seq_length: 4096
        tokenizer:
            type: /llama3_tokenizer

    model:
        encoder_seq_len: 4096
        weight_init_only: True
        optim:
            lr: 1.5e-4
            sched:
                name: CosineAnnealing

    model_alignment_strategy:
        orpo:
            beta: 0.1
            max_prompt_length: 2048
            truncation_mode: keep_start


**exp_manager**

    **checkpoint_callback_params.every_n_train_steps**

    How often we want to checkpoint.

        * **Type**: int
        * **Required**: True

**model**
    **encoder_seq_length**

    Setting the sequence length for the training job. This parameter is common for all
    models supported in the library.

        * **Type**: int
        * **Required**: True

    **optim.sched**

    This is where the LR schedulers can be set. We can configure the schedulers supported by
    ``NeMo``. All the schedulers can be configured according to the
    `parameters specified here <https://github.com/NVIDIA/NeMo/blob/v1.14.0/nemo/core/config/schedulers.py>`__.

        * **Type**: config
        * **Possible Values**: ``LinearAnnealingWithWarmUp``, ``CosineAnnealing``, ``WarmupPolicy``,
        *  ``WarmupHoldPolicy``, ``SquareAnnealing``, ``NoamAnnealing``, ``WarmupAnnealing``,
        *   ``StepLR``, ``rprop``, ``ExponentialLR``
        * **Required**: True


 **model_alignment_strategy**

    Set only when using finetuning specific algorithms (SFT, DPO, ORPO, etc) and parameter-efficient
    fine-tuning methods like LoRA (Low-Rank Adaptation).

        **orpo**
            Odds Ratio Preference Optimization (ORPO) specific parameters.

            **beta**

            KL-divergence beta to control divergence of policy model from reference model

                * **Type**: float
                * **Default**: 0.01
                * **Required**: True

Download the dataset
--------------------

The DPO (& ORPO) tutorial makes use of the same preprocessed version of `intel-orca_dpo_pairs`
preference dataset that is stored in S3. The dataset can be downloaded to your cluster or
instance by running the following AWS CLI commands on the head node or your Trn1 instance:

.. code-block:: bash

    export DATA_DIR=~/examples_datasets/llama3_8b
    mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
    aws s3 cp s3://neuron-s3/training_datasets/llama/dpo/data_dpo.jsonl .  --no-sign-request

Then, download the ``config.json`` file:

For Llama-3.1-8B:

.. code-block:: bash

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3.1/config.json ~/


For Llama-3-8B:

.. code-block:: bash

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3/config.json ~/


Convert data to DPO-specific Preference data format
---------------------------------------------------

If you directly downloaded the `Intel ORCA_dpo_pairs dataset <https://huggingface.co/datasets/Intel/orca_dpo_pairs>`_, then you need to convert the
data into preference dataset format using the script below.

.. note::
    For different datasets with different field names, make necessary changes to the script accordingly.

.. code-block:: python

    from datasets import load_dataset
    from transformers import AutoTokenizer

    def preference_data_format(example):

        system = "<|im_start|>\n" + example['system'] + "<|im_end|>\n"

        # Format instruction
        prompt = "<|im_start|> " + example['question'] + "<|im_end|>\n<|im_start|>assistant\n"

        # Format chosen answer
        chosen = example['chosen'] + "<|im_end|>\n"

        # Format rejected answer
        rejected = example['rejected'] + "<|im_end|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    # Particular dataset with following fields: "system", "question", "chosen", "rejected"
    dataset = load_dataset("json", data_files="orca_rlhf.jsonl", split="train")

    # Save columns
    original_columns = dataset.column_names

    # Format dataset
    dataset = dataset.map(
        preference_data_format,
        remove_columns=original_columns
        )

    # save converted preference dataset
    dataset.to_json("data_dpo.jsonl")


Download pretrained model checkpoint and tokenizer
--------------------------------------------------

In this tutorial, we will use a pretrained Llama3-8B checkpoint (post-SFT checkpoint preferred)
from the original repository.
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

You can use your Llama model, pretrained checkpoint and tokenizer by
modifying the ``hf_llama3_8B_<DPO/ORPO>_config.yaml`` file.


Checkpoint Conversion
^^^^^^^^^^^^^^^^^^^^^

Follow this :ref:`Checkpoint Conversion Guide <checkpoint_conversion>` to convert the
HF-style Llama3-8B checkpoint
to NxDT supported format and store it in ``pretrained_ckpt`` directory.
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
it will be ``hf_llama3_8B_<DPO/ORPO>_config.yaml``. The default config here is a 8B parameter model,
but users can also add their own ``conf/*.yaml`` files and run different configs and
hyperparameters if desired. Please see :ref:`Config Overview <nxdt_config_overview>`
for examples and usage for the ``.yaml`` config files.

Next, run the following commands to launch an AOT pre-compilation job on your instance:

.. code-block:: bash

    export COMPILE=1
    export CONF_FILE=hf_llama3_8B_<DPO/ORPO>_config
    ./train.sh

The compile output and logs will be shown directly in the terminal
and you will see logs similar to this:

.. code-block:: bash

    2024-10-24 18:49:49.000950: INFO ||NEURON_PARALLEL_COMPILE||: Total graphs: 32
    2024-10-24 18:49:49.000950: INFO ||NEURON_PARALLEL_COMPILE||: Total successful compilations: 32
    2024-10-24 18:49:49.000950: INFO ||NEURON_PARALLEL_COMPILE||: Total failed compilations: 0

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
    export CONF_FILE=hf_llama3_8B_<DPO/ORPO>_config
    ./train.sh

Once the model is loaded onto the Trainium accelerators and training has commenced,
you will begin to see output indicating the job progress:

Example:

.. code-block:: bash

    Epoch 0:   5%|â–         | 3/62 [02:59<58:44,  0.02it/s, v_num=8-06, reduced_train_loss=6.930, chosen_rewards=-0.81, rejected_rewards=-0.675, lr=2.73e-5, parameter_norm=1.95e+3, global_step=1.000, consumed_samples=32.00, throughput=0.108, throughput_peak=0.0677, gradient_norm=8.600]
    Epoch 0:   6%|â–‹         | 4/62 [03:24<49:27,  0.02it/s, v_num=8-06, reduced_train_loss=6.790, chosen_rewards=-0.628, rejected_rewards=-0.64, lr=5.45e-5, parameter_norm=1.95e+3, global_step=3.000, consumed_samples=64.00, throughput=0.181, throughput_peak=0.146, gradient_norm=6.590]
    Epoch 0:   8%|â–Š         | 5/62 [03:50<43:42,  0.02it/s, v_num=8-06, reduced_train_loss=6.790, chosen_rewards=-0.628, rejected_rewards=-0.64, lr=5.45e-5, parameter_norm=1.95e+3, global_step=3.000, consumed_samples=64.00, throughput=0.181, throughput_peak=0.146, gradient_norm=6.590]

.. note::
    The values in the above logs will differ based on config used, package versions,
    models, and other factors. This is just an example.

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