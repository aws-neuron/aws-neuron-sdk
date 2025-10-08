.. _torch-hf-bert-finetune:

PyTorch Neuron for Trainium Hugging Face BERT MRPC task finetuning using Hugging Face Trainer API
=================================================================================================

.. note::

   Use Hugging Face `Optimum-Neuron <https://huggingface.co/docs/optimum-neuron/index>`_ for the best coverage and support for Hugging Face models running on AWS Trainium and Inferentia devices.

In this tutorial, we show how to run a Hugging Face script that uses Hugging Face Trainer API
to do fine-tuning on Trainium. The example follows the `text-classification
example <https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification>`__
which fine-tunes BERT-base model for sequence classification on the GLUE
benchmark.


.. contents:: Table of Contents
   :local:
   :depth: 2

.. include:: ../note-performance.txt

Setup and compilation
---------------------

Before running the tutorial please follow the installation instructions at:

:ref:`Install PyTorch Neuron on
Trn1 <setup-torch-neuronx>`

Please set the storage of instance to *512GB* or more if you also want to run through the BERT pretraining and GPT pretraining tutorials.

For all the commands below, make sure you are in the virtual environment that you have created above before you run the commands:

.. code:: shell

   source ~/aws_neuron_venv_pytorch/bin/activate

First we install a recent version of HF transformers, scikit-learn and evaluate packages in our environment as well as download the source matching the installed version. In this example, we use the text classification example from HF transformers source:

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_setup_code.sh
   :language: shell
   :lines: 5-10

Single-worker training
----------------------

We will run MRPC task fine-tuning following the example in README.md located in the path ``~/transformers/examples/pytorch/text-classification``. In this part of the tutorial we will use the Hugging Face model hub's pretrained ``bert-large-uncased`` model.

.. note::

    If you are using older versions of transformers <4.27.0 or PyTorch Neuron <1.13.0, please see section :ref:`workarounds_for_older_versions` for necessary workarounds.

We use BF16 mixed-precision casting using trainer API ``--bf16`` option and compiler flag ``--model-type=transformer`` to enable best performance.
We also launch the ``run_glue.py`` script with ``torchrun`` using ``--nproc_per_node=N`` option to specify the number of workers. Here we start off with 1 worker.

.. note::

    With transformers version 4.44 and up, please use torchrun even for one worker (``--nproc_per_node=1``) to avoid execution hang.

First, paste the following script into your terminal to create a “run.sh” file and change it to executable:

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_single_worker_training.sh
   :language: shell
   :lines: 7-29

We optionally precompile the model and training script using neuron_parallel_compile to warm up the persistent
graph cache (Neuron Cache) such that the actual run has fewer compilations (faster run
time):

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_single_worker_training.sh
   :language: shell
   :lines: 32

Please ignore the results from this precompile run as it is only for
extracting and compiling the XLA graphs.

.. note::

   With both train and evaluation options (``--do_train`` and ``--do_eval``), you will encounter harmless error
   ``ValueError: Target is multiclass but average='binary'`` when using neuron_parallel_compile.

Precompilation is optional and only needed to be done once unless hyperparameters such as batch size are modified.
After the optional precompilation, the actual run will be faster with minimal
additional compilations.

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_single_worker_training.sh
   :language: shell
   :lines: 34

If precompilation was not done, the first execution of ./run.sh will be slower due to serial compilations. Rerunning the same script a second time would show quicker execution as the compiled graphs will be already cached in persistent cache.

.. _multi_worker_training_parallel:

Multi-worker data-parallel training
-----------------------------------

The above script would run one worker on one Logical NeuronCore. To run on
multiple Logical NeuronCores in data-parallel configuration, launch the ``run_glue.py`` script with ``torchrun`` using ``--nproc_per_node=N`` option to specify the number of workers
(N=2 for trn1.2xlarge, and N=2, 8, or 32 for trn1.32xlarge).

.. note::

    If you are using older versions of transformers <4.27.0 or PyTorch Neuron <1.13.0, please see section :ref:`workarounds_for_older_versions` for necessary workarounds.

The following example runs 2 workers.
Paste the following script into your terminal to create a “run_2w.sh” file and change it to executable:

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_multi_worker_training_code.sh
   :language: shell
   :lines: 7-29

Again, we optionally precompile the model and training script using neuron_parallel_compile to warm up the persistent
graph cache (Neuron Cache), ignoring the results from this precompile run as it is only for
extracting and compiling the XLA graphs:

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_multi_worker_training_code.sh
   :language: shell
   :lines: 32

Precompilation is optional and only needed to be done once unless hyperparameters such as batch size are modified.
After the optional precompilation, the actual run will be faster with minimal
additional compilations.

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_multi_worker_training_code.sh
   :language: shell
   :lines: 34

During run, you will now notice that the "Total train batch size" is now 16 and the "Total optimization steps" is now half the number for one worker training.

Converting BERT pretrained checkpoint to Hugging Face pretrained model format
-----------------------------------------------------------------------------
If you have a pretrained checkpoint (i.e., from the BERT phase 2 pretraining tutorial), you can run the script below (saved as "convert.py") to convert BERT pretrained saved checkpoint to Hugging Face pretrained model format. An example phase 2 pretrained checkpoint can be downloaded from ``s3://neuron-s3/training_checkpoints/pytorch/dp_bert_large_hf_pretrain/ckpt_29688.pt``. Note that here we also use the ``bert-large-uncased`` model configuration to match the BERT-Large model trained following BERT phase 2 pretraining tutorial.

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_converted_checkpoint_training.sh
   :language: python
   :lines: 8-33

Run the conversion script as:

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_converted_checkpoint_training.sh
   :language: shell
   :lines: 35

After conversion, the new Hugging Face pretrained model is stored in the output directory specified by the ``--output_saved_model_path`` option which is ``hf_saved_model`` by default. You will use this directory in the next step.

Paste the following script into your terminal to create a “run_converted.sh” file and change it to executable:
(note that it uses the converted Hugging Face pretrained model in ``hf_saved_model`` directory):

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_converted_checkpoint_training.sh
   :language: shell
   :lines: 38-61

If it is the first time running with ``bert-large-uncased`` model or if hyperparameters have changed, then the optional one-time precompilation step can save compilation time:

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_converted_checkpoint_training.sh
   :language: shell
   :lines: 64

If you have run the single worker training in a previous section, then you can skip the precompilation step and just do:

.. literalinclude:: tutorial_source_code/bert_mrpc_finetuning/bert_mrpc_finetuning_converted_checkpoint_training.sh
   :language: shell
   :lines: 67


.. _known_issues:

Known issues and limitations
----------------------------

``INVALID_ARGUMENT: Input dimension should be either 1 or equal to the output dimension ...`` during precompilation of evaluation phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During precompilation (``neuron_parallel_compile``) of model evaluation phase, you may see the following crash:

.. code:: shell

    Status: INVALID_ARGUMENT: Input dimension should be either 1 or equal to the output dimension it is broadcasting into; the 1th operand dimension is 2, the 1th output dimension is 0.
    *** Begin stack trace ***
        tsl::CurrentStackTrace[abi:cxx11]()
        xla::Shape const* ConsumeValue<xla::Shape const*>(absl::lts_20230802::StatusOr<xla::Shape const*>&&) 
        ...

This is due to output dependent logic in HuggingFace Accelerate's ``pad_across_processes`` utility function. To work-around this issue, please add the following code snippet to the top of your run script (i.e. ``run_glue.py``):

.. code:: python

    import os
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", "0") == "1":
        from accelerate.accelerator import Accelerator
        def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
            return tensor
        Accelerator.pad_across_processes = pad_across_processes


Compilations for every evaluation step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During model evaluation, there can be small compilations for every evaluation step due to a `known transformers issue <https://github.com/huggingface/transformers/issues/37593>`_. The work-around is to set training arguments ``eval_do_concat_batches=False`` and apply the changes in `the PR <https://github.com/huggingface/transformers/pull/37621>`_ which will be in a future release of transformers package (version 4.52 or later).

Running one worker fine-tuning without torchrun would result in a hang
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With transformers>=4.44.0, running one worker fine-tuning without torchrun would result in a hang. To workaround and run one worker fine-tuning, use ``torchrun --nproc_per_node=1 <script>``.


Long compilation times
^^^^^^^^^^^^^^^^^^^^^^

Long compilation times can be alleviated by using the ``neuron_parallel_compile`` tool to extract graphs from a short trial run and compile them in parallel ahead of the actual run, as shown above. Subsequent runs would load compiled graphs from the Neuron Cache and thus avoid long compilation times.

Compilation errors during precompilation using ``neuron_parallel_compile`` on small EC2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When precompiling using batch size of 16 on trn1.2xlarge, you will see ``ERROR ||PARALLEL_COMPILE||: parallel compilation with neuronx-cc exited with error.Received error code: -9``. To workaround this error, please set ``NEURON_PARALLEL_COMPILE_MAX_RETRIES=1`` in the environment.


Variable input sizes leading to timeouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variable input sizes: When fine-tuning models such as dslim/bert-base-NER using the `token-classification example <https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification>`__, you may encounter timeouts (lots of "socket.h:524 CCOM WARN Timeout waiting for RX" messages) and execution hang. This occurs because NER dataset has different sample sizes, which causes many recompilations and compiled graph (NEFF) reloads. Furthermore, different data parallel workers can execute different compiled graph. This multiple-program multiple-data behavior is currently unsupported. To workaround this issue, please pad to maximum length using the Trainer API option ``--pad_to_max_length``.

"ValueError: Your setup doesn't support bf16/gpu."
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using latest HuggingFace transformers version, you may see "ValueError: Your setup doesn't support bf16/gpu." To fix this, please use ``--use_cpu True`` in your scripts.

.. _resolved_hf_issues:

Resolved issues
---------------

-  With torch-neuronx 2.1, HF Trainer API's use of XLA function ``xm.mesh_reduce`` causes ``"EOFError: Ran out of input"`` or ``"_pickle.UnpicklingError: invalid load key, '!'"`` errors during Neuron Parallel Compile. This is an issue with the trial execution of empty NEFFs and should not affect the normal execution of the training script.
-  Multi-worker training using Trainer API resulted in too many graph compilations for HF transformers>=4.35: This is resolved with HF transformers>=4.37 with the additional workarounds as shown in `the ticket <https://github.com/aws-neuron/aws-neuron-sdk/issues/813>`_.
-  Reduced accuracy for RoBERTa-Large is seen with Neuron PyTorch 1.12 (release 2.6) in FP32 mode with compiler BF16 autocast.
   The workaround is to set NEURON_CC_FLAGS="--auto-cast none" or set NEURON_RT_STOCHASTIC_ROUNDING_EN=1.
- When running HuggingFace GPT fine-tuning with transformers version >= 4.21.0 and using XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1, you might see NaNs in the loss immediately at the first step. This issue occurs due to large negative constants used to implement attention masking (https://github.com/huggingface/transformers/pull/17306). To workaround this issue, please use transformers version <= 4.20.0.
-  With release 2.6 and transformers==4.25.1,
   using ``neuron_parallel_compile`` tool to run ``run_glue.py`` script
   with both train and evaluation options (``--do_train`` and ``--do_eval``), you will encounter harmless error
   ``ValueError: Target is multiclass but average='binary'``
-  Using ``neuron_parallel_compile`` tool to run ``run_glue.py`` script
   with both train and evaluation options (``--do_train`` and ``--do_eval``), you will
   encounter INVALID_ARGUMENT error. To avoid this, only enable train for parallel
   compile (``--do_train``). This will cause compilations during evaluation step.
   The INVALID_ARGUMENT error is fixed in release 2.6 together with latest transformers package version 4.25.1.
- When using Trainer API option --bf16, you will see "RuntimeError: No CUDA GPUs are available". To workaround this error, please add "import torch; torch.cuda.is_bf16_supported = lambda: True" to the Python script (i.e. run_glue.py). (Trainer API option --fp16 is not yet supported).
-  When running HuggingFace BERT (any size) fine-tuning tutorial or pretraining tutorial with transformers version >= 4.21.0 and < 4.25.1 and using XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1, you will see NaNs in the loss immediately at the first step. More details on the issue can be found at `pytorch/xla#4152 <https://github.com/pytorch/xla/issues/4152>`_. The workaround is to use transformers version < 4.21.0 or >= 4.25.1, or add ``transformers.modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16`` to your Python script (i.e. run_glue.py).
-  Some recompilation is seen at the epoch boundary even after ``neuron_parallel_compile`` is used. This can be fixed by using the same number of epochs both during precompilation and the actual run.
-  When running multi-worker training, you may see the process getting killed at the time of model saving on trn1.2xlarge.
   This happens because the transformers ``trainer.save_model`` api uses ``xm.save`` for saving models.
   This api is known to cause high host memory usage in multi-worker setting `see Saving and Loading XLA Tensors in  <https://github.com/pytorch/xla/blob/master/API_GUIDE.md>`__ . Coupled with a compilation
   at the same time results in a host OOM. To avoid this issue, we can: Precompile all the graphs in multi-worker
   training. This can be done by running the multi-worker training first with ``neuron_parallel_compile <script>``
   followed by the actual training. This would avoid the compilation at model save during actual training.

.. _workarounds_for_older_versions:

Older versions of transformers <4.27.0 or PyTorch Neuron <1.13.0
----------------------------------------------------------------

If using older versions of transformers package before 4.27.0 or PyTorch Neuron before 1.13.0, please edit the python script run_glue.py and add the following lines after the Python
imports. They set the compiler flag for transformer model type and enable data parallel training using torchrun:

.. code:: python

    # Enable torchrun
    import os
    import torch
    import torch_xla.distributed.xla_backend
    from packaging import version
    from transformers import __version__, Trainer
    if version.parse(__version__) < version.parse("4.26.0") and os.environ.get("WORLD_SIZE"):
        torch.distributed.init_process_group('xla')

    # Disable DDP for torchrun
    import contextlib
    if version.parse(__version__) < version.parse("4.20.0"):
        def _wrap_model(self, model, training=True):
            model.no_sync = lambda: contextlib.nullcontext()
            return model
    else:
        def _wrap_model(self, model, training=True, dataloader=None):
            model.no_sync = lambda: contextlib.nullcontext()
            return model
    Trainer._wrap_model = _wrap_model

    # Workaround for NaNs seen with transformers version >= 4.21.0
    # https://github.com/aws-neuron/aws-neuron-sdk/issues/593
    import transformers
    if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
        transformers.modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16

