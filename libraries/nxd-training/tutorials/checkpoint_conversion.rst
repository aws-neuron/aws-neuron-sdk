.. _checkpoint_conversion:

Checkpoint Conversion
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

The NxD Training library provides a versatile checkpoint conversion functionality,
allowing seamless transition between different model styles. This tutorial aims to provide a
comprehensive guide through the various use cases and demonstrate how to perform the checkpoint conversions.

Supported Model Architectures
-----------------------------

The checkpoint conversion functionality supports conversion of the following model styles to/from NxDT checkpoints:

1. **HuggingFace (HF) style models**
2. **Megatron style models**

Extends support for both GQA (Llama-3) and non-GQA models (Llama-2).

Conversion Scenarios and Usage
------------------------------

The tool supports the following conversion scenarios. It internally
uses ``NeuronxDistributed (NxD)`` to convert to/from checkpoints.
Run the following commands from the ``/examples/checkpoint_conversion_scripts/`` directory:

.. note::

   1. Please ensure that the model configuration ``config.json`` file is present,
      as it is required for checkpoint conversions.
      If not present, you will need to create it.

   2. If your HF/custom checkpoint has multiple ``.bin`` or ``.pt`` files
      then merge and convert to a single file before conversion.

For conversion of non-GQA based models (e.g. Llama2), just set the ``--qkv_linear`` argument to ``False``.

1. **HF style model**:

   a. **HF to NxDT checkpoint**:

      **Command**:

      .. code-block:: bash

        python3 checkpoint_converter.py --model_style hf --input_dir /home/ubuntu/pretrained_llama_3_8B_hf/pytorch_model.bin --output_dir /home/ubuntu/converted_hf_style_hf_to_nxdt_tp8pp4/ --save_xser True --config /home/ubuntu/pretrained_llama_3_8B_hf/config.json --tp_size 8 --pp_size 4 --n_layers 32 --kv_size_multiplier 1 --qkv_linear True --convert_from_full_state

     This converts an HF-style checkpoint to an NxDT checkpoint.

   b. **NxDT to HF checkpoint**:

    **Command**:

    .. code-block:: bash

       python3 checkpoint_converter.py --model_style hf --input_dir ~/examples/nemo_experiments/hf_llama3_8B_SFT/2024-07-19_23-07-40/checkpoints/hf_llama3_8B--step=5-consumed_samples=160.0.ckpt/model --output_dir ~/converted_hf_style_nxdt_to_hf_tp8pp4/ --load_xser True --config ~/config.json --tp_size 8 --pp_size 4 --kv_size_multiplier 1 --qkv_linear True --convert_to_full_state

    This converts an NxDT checkpoint to an HF-style checkpoint.

2. **Megatron style model (non-GQA models: e.g., Llama-2, and GQA models: e.g., Llama-3)**:

   a. **HF to NxDT Megatron checkpoint**:

    **Command**:

    .. code-block:: bash

       python3 checkpoint_converter.py --model_style megatron --input_dir ~/megatron-tp8pp4-nxdt-to-hf4/checkpoint.pt --output_dir ~/meg_nxdt_hf3_nxdt3 --config ~/llama_gqa/config.json --save_xser True --tp_size 8 --pp_size 4 --n_layers 32 --kv_size_multiplier 1 --qkv_linear True --convert_from_full_state

    This converts an HF-style checkpoint to an NxDT Megatron-style checkpoint.

   b. **NxDT Megatron checkpoint to HF**:

    **Command**:

    .. code-block:: bash

       python3 checkpoint_converter.py  --model_style megatron --input_dir ~/examples/nemo_experiments/megatron_llama/2024-07-23_21-07-30/checkpoints/megatron_llama--step=5-consumed_samples=5120.0.ckpt/model --output_dir ~/megatron-tp8pp4-nxdt-to-hf4 --load_xser True --config ~/llama_gqa/config.json --tp_size 8 --pp_size 4 --kv_size_multiplier 1 --qkv_linear True --convert_to_full_state

    This converts an NxDT Megatron-style checkpoint to an HF-style checkpoint (GQA-based model, see: ``--qkv_linear`` set to ``True``).


Key Arguments
^^^^^^^^^^^^^

The ``checkpoint_converter.py`` script supports the following key arguments:

- ``--model_style``: Specifies the model style, either `hf` (HuggingFace: default) or `megatron`
- ``--input_dir``: (required) directory containing the input checkpoint
- ``--output_dir``: (required) directory to save the converted checkpoint directory
- ``--save_xser``: Saves the checkpoint with torch_xla serialization
- ``--load_xser``: Loads the checkpoint with torch_xla serialization
- ``--convert_from_full_state``: Converts full model checkpoint to sharded model checkpoint
- ``--convert_to_full_state``: Converts sharded model checkpoint to full model checkpoint
- ``--config``: path to the model configuration file (create `json` file if not present)
- ``--tp_size``: tensor parallelism degree
- ``--pp_size``: pipeline parallelism degree
- ``--n_layers``: number of layers in the model
- ``--kv_size_multiplier``: key-value size multiplier
- ``--qkv_linear``: boolean to specify GQA/non-GQA models

We recommend enabling xser for significantly faster save and load times.
Note that if the checkpoint is saved with xser, it can only be loaded with xser,
and vice versa.

Conversion Example
------------------

Assuming you have a pre-trained HF-style Llama3-8B model checkpoint looking similar to:

``input_dir: /hf/checkpoint/pytorch_model.bin``

.. code-block:: bash

  $ ls /hf/checkpoint

  -rw-r--r-- 1 user group 123 Aug 27 2024 pytorch_model.bin

Convert the HF-style checkpoint to an NxDT checkpoint on a single instance:

.. code-block:: bash

  python3 checkpoint_converter.py --model_style hf --input_dir /hf/checkpoint/pytorch_model.bin --output_dir /nxdt/checkpoint --save_xser True --convert_from_full_state --config /path/to/config.json --tp_size 8 --pp_size 4 --n_layers 32 --kv_size_multiplier 1 --qkv_linear True --convert_from_full_state

This command will create an NxDT checkpoint in ``output_dir: /nxdt/checkpoint``
and it will be sharded with (tp=8, pp=4) like:

.. code-block:: bash

  $ ls /nxdt/checkpoint/model

  -rw-r--r-- 1 user group 123 Aug 27 2024 dp_rank_00_tp_rank_00_pp_rank_00.pt
  -rw-r--r-- 1 user group 456 Aug 27 2024 dp_rank_00_tp_rank_01_pp_rank_00.pt
  ...........................................................................
  -rw-r--r-- 1 user group 789 Aug 27 2024 dp_rank_00_tp_rank_07_pp_rank_02.pt
  -rw-r--r-- 1 user group 122 Aug 27 2024 dp_rank_00_tp_rank_07_pp_rank_03.pt