.. _torch-hf-bert-finetune:

PyTorch Neuron for Trainium Hugging Face BERT MRPC task finetuning using Hugging Face Trainer API
=================================================================================================

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

* :ref:`Install PyTorch Neuron on Trn1 <pytorch-neuronx-install>`

Please set the storage of instance to *512GB* or more if you also want to run through the BERT pretraining and GPT pretraining tutorials.

For all the commands below, make sure you are in the virtual environment that you have created above before you run the commands:

.. code:: shell

   source ~/aws_neuron_venv_pytorch/bin/activate

First we install a recent version of HF transformers, scikit-learn and evaluate packages in our environment as well as download the source matching the installed version. In this example, we use the text classification example from HF transformers source:

.. code:: bash

    export HF_VER=4.27.4
    pip install -U transformers==$HF_VER datasets evaluate scikit-learn
    cd ~/
    git clone https://github.com/huggingface/transformers --branch v$HF_VER
    cd ~/transformers/examples/pytorch/text-classification

Single-worker training
----------------------

We will run MRPC task fine-tuning following the example in README.md located in the path ``~/transformers/examples/pytorch/text-classification``. In this part of the tutorial we will use the Hugging Face model hub's pretrained ``bert-large-uncased`` model.

.. note::

    If you are using older versions of transformers <4.27.0 or PyTorch Neuron <1.13.0, please see section :ref:`workarounds_for_older_versions` for necessary workarounds.

We use full BF16 casting using XLA_USE_BF16=1 and compiler flag ``--model-type=transformer`` to enable best performance.
First, paste the following script into your terminal to create a “run.sh” file and change it to executable:

.. code:: bash

    tee run.sh > /dev/null <<EOF
    #!/usr/bin/env bash
    export TASK_NAME=mrpc
    export NEURON_CC_FLAGS="--model-type=transformer"
    XLA_USE_BF16=1 python3 ./run_glue.py \\
    --model_name_or_path bert-large-uncased \\
    --task_name \$TASK_NAME \\
    --do_train \\
    --do_eval \\
    --max_seq_length 128 \\
    --per_device_train_batch_size 8 \\
    --learning_rate 2e-5 \\
    --num_train_epochs 5 \\
    --overwrite_output_dir \\
    --output_dir /tmp/\$TASK_NAME/ |& tee log_run
    EOF

    chmod +x run.sh

We optionally precompile the model and training script using neuron_parallel_compile to warm up the persistent
graph cache (Neuron Cache) such that the actual run has fewer compilations (faster run
time):

.. code:: bash

    neuron_parallel_compile ./run.sh

Please ignore the results from this precompile run as it is only for
extracting and compiling the XLA graphs.

.. note::

   With both train and evaluation options (``--do_train`` and ``--do_eval``), you will encounter harmless error
   ``ValueError: Target is multiclass but average='binary'`` when using neuron_parallel_compile.

Precompilation is optional and only needed to be done once unless hyperparameters such as batch size are modified.
After the optional precompilation, the actual run will be faster with minimal
additional compilations.

.. code:: bash

    ./run.sh

If precompilation was not done, the first execution of ./run.sh will be slower due to serial compilations. Rerunning the same script a second time would show quicker execution as the compiled graphs will be already cached in persistent cache.

.. _multi_worker_training:

Multi-worker training
---------------------

The above script would run one worker on one NeuronCore. To run on
multiple cores, launch the ``run_glue.py`` script with ``torchrun`` using ``--nproc_per_node=N`` option to specify the number of workers
(N=2 for trn1.2xlarge, and N=2, 8, or 32 for trn1.32xlarge).

.. note::

    If you are using older versions of transformers <4.27.0 or PyTorch Neuron <1.13.0, please see section :ref:`workarounds_for_older_versions` for necessary workarounds.

The following example runs 2 workers.
Paste the following script into your terminal to create a “run_2w.sh” file and change it to executable:

.. code:: bash

    tee run_2w.sh > /dev/null <<EOF
    #!/usr/bin/env bash
    export TASK_NAME=mrpc
    export NEURON_CC_FLAGS="--model-type=transformer"
    XLA_USE_BF16=1 torchrun --nproc_per_node=2 ./run_glue.py \\
    --model_name_or_path bert-large-uncased \\
    --task_name \$TASK_NAME \\
    --do_train \\
    --do_eval \\
    --max_seq_length 128 \\
    --per_device_train_batch_size 8 \\
    --learning_rate 2e-5 \\
    --num_train_epochs 5 \\
    --overwrite_output_dir \\
    --output_dir /tmp/\$TASK_NAME/ |& tee log_run_2w
    EOF

    chmod +x run_2w.sh

Again, we optionally precompile the model and training script using neuron_parallel_compile to warm up the persistent
graph cache (Neuron Cache), ignoring the results from this precompile run as it is only for
extracting and compiling the XLA graphs:

.. code:: bash

    neuron_parallel_compile ./run_2w.sh

Precompilation is optional and only needed to be done once unless hyperparameters such as batch size are modified.
After the optional precompilation, the actual run will be faster with minimal
additional compilations.

.. code:: bash

    ./run_2w.sh

During run, you will now notice that the "Total train batch size" is now 16 and the "Total optimization steps" is now half the number for one worker training.

Converting BERT pretrained checkpoint to Hugging Face pretrained model format
-----------------------------------------------------------------------------
If you have a pretrained checkpoint (i.e., from the BERT phase 2 pretraining tutorial), you can run the script below (saved as "convert.py") to convert BERT pretrained saved checkpoint to Hugging Face pretrained model format. An example phase 2 pretrained checkpoint can be downloaded from ``s3://neuron-s3/training_checkpoints/pytorch/dp_bert_large_hf_pretrain/ckpt_29688.pt``. Note that here we also use the ``bert-large-uncased`` model configuration to match the BERT-Large model trained following BERT phase 2 pretraining tutorial.

.. code:: python

    import os
    import sys
    import argparse
    import torch
    import transformers
    from transformers import (
        BertForPreTraining,
    )
    import torch_xla.core.xla_model as xm
    from transformers.utils import check_min_version
    from transformers.utils.versions import require_version

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='bert-large-uncased',  help="Path to model identifier from huggingface.co/models")
        parser.add_argument('--output_saved_model_path', type=str, default='./hf_saved_model', help="Directory to save the HF pretrained model format.")
        parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to pretrained checkpoint which needs to be converted to a HF pretrained model format")
        args = parser.parse_args(sys.argv[1:])

        model = BertForPreTraining.from_pretrained(args.model_name)
        check_point = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(check_point['model'], strict=False)
        model.save_pretrained(args.output_saved_model_path, save_config=True, save_function=xm.save)
        print("Done converting checkpoint {} to HuggingFace saved model in directory {}.".format(args.checkpoint_path, args.output_saved_model_path))

Run the conversion script as:

.. code:: bash

    python convert.py --checkpoint_path ckpt_29688.pt

After conversion, the new Hugging Face pretrained model is stored in the output directory specified by the ``--output_saved_model_path`` option which is ``hf_saved_model`` by default. You will use this directory in the next step.

Paste the following script into your terminal to create a “run_converted.sh” file and change it to executable:
(note that it uses the converted Hugging Face pretrained model in ``hf_saved_model`` directory):

.. code:: bash

    tee run_converted.sh > /dev/null <<EOF
    #!/usr/bin/env bash
    export TASK_NAME=mrpc
    export NEURON_CC_FLAGS="--model-type=transformer"
    XLA_USE_BF16=1 python3 ./run_glue.py \\
    --model_name_or_path hf_saved_model \\
    --tokenizer_name bert-large-uncased \\
    --task_name \$TASK_NAME \\
    --do_train \\
    --do_eval \\
    --max_seq_length 128 \\
    --per_device_train_batch_size 8 \\
    --learning_rate 2e-5 \\
    --num_train_epochs 5 \\
    --overwrite_output_dir \\
    --output_dir /tmp/\$TASK_NAME/ |& tee log_run_converted
    EOF

    chmod +x run_converted.sh

If it is the first time running with ``bert-large-uncased`` model or if hyperparameters have changed, then the optional one-time precompilation step can save compilation time:

.. code:: bash

    neuron_parallel_compile ./run_converted.sh

If you have run the single worker training in a previous section, then you can skip the precompilation step and just do:

.. code:: bash

    ./run_converted.sh

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

.. _known_issues:

Known issues and limitations
----------------------------

The following are currently known issues:

-  Long compilation times: this can be alleviated with
   ``neuron_parallel_compile`` tool to extract graphs from a short trial run and
   compile them in parallel ahead of the actual run, as shown above.
-  When precompiling using batch size of 16 on trn1.2xlarge, you will see ``ERROR ||PARALLEL_COMPILE||: parallel compilation with neuronx-cc exited with error.Received error code: -9``. To workaround this error, please set NEURON_PARALLEL_COMPILE_MAX_RETRIES=1 in the environment.
-  With release 2.6 and transformers==4.25.1,
   using ``neuron_parallel_compile`` tool to run ``run_glue.py`` script
   with both train and evaluation options (``--do_train`` and ``--do_eval``), you will encounter harmless error
   ``ValueError: Target is multiclass but average='binary'``
-  Reduced accuracy for RoBerta-Large is seen with Neuron PyTorch 1.12 (release 2.6) in FP32 mode with compiler BF16 autocast.
   The workaround is to set NEURON_CC_FLAGS="--auto-cast none" or set NEURON_RT_STOCHASTIC_ROUNDING_EN=1.
-  When using DDP in PT 1.13, compilation of one graph will fail with "Killed" error message for ``bert-large-uncased``. For ``bert-base-cased``, the final MRPC evaluation accuracy is 31% which is lower than expected. These issues are being investigated and will be fixed in an upcoming release. For now, DDP is disabled with the workaround shown above in :ref:`multi_worker_training`.
-  When using DDP in PT 1.13 with neuron_parallel_compile precompilation, you will hit an error ``Rank 1 has 393 params, while rank 0 has inconsistent 0 params.``. To workaround this error, add the follow code snippet at the top of ``run_glue.py`` to skip the problematic shape verification code during precompilation:

.. code:: python

   import os
   if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
       import torch.distributed as dist
       _verify_param_shape_across_processes = lambda process_group, tensors, logger=None: True

- Variable input sizes: When fine-tune models such as dslim/bert-base-NER using the `token-classification example <https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification>`__, you may encounter timeouts (lots of "socket.h:524 CCOM WARN Timeout waiting for RX" messages) and execution hang. This occurs because NER dataset has different sample sizes, which causes many recompilations and compiled graph (NEFF) reloads. Furthermore, different data parallel workers can execute different compiled graph. This multiple-program multiple-data behavior is currently unsupported. To workaround this issue, please pad to maximum length using the Trainer API option ``--pad_to_max_length``.
- When running HuggingFace GPT fine-tuning with transformers version >= 4.21.0 and using XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1, you might see NaNs in the loss immediately at the first step. This issue occurs due to large negative constants used to implement attention masking (https://github.com/huggingface/transformers/pull/17306). To workaround this issue, please use transformers version <= 4.20.0.
- When using Trainer API option --bf16, you will see "RuntimeError: No CUDA GPUs are available". To workaround this error, please add "import torch; torch.cuda.is_bf16_supported = lambda: True" to the Python script (i.e. run_glue.py). (Trainer API option --fp16 is not yet supported).

The following are resolved issues:

-  Using ``neuron_parallel_compile`` tool to run ``run_glue.py`` script
   with both train and evaluation options (``--do_train`` and ``--do_eval``), you will
   encounter INVALID_ARGUMENT error. To avoid this, only enable train for parallel
   compile (``--do_train``). This will cause compilations during evaluation step.
   The INVALID_ARGUMENT error is fixed in release 2.6 together with latest transformers package version 4.25.1.
-  When running HuggingFace BERT (any size) fine-tuning tutorial or pretraining tutorial with transformers version >= 4.21.0 and < 4.25.1 and using XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1, you will see NaNs in the loss immediately at the first step. More details on the issue can be found at `pytorch/xla#4152 <https://github.com/pytorch/xla/issues/4152>`_. The workaround is to use transformers version < 4.21.0 or >= 4.25.1, or add ``transformers.modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16`` to your Python script (i.e. run_glue.py).
-  Some recompilation is seen at the epoch boundary even after ``neuron_parallel_compile`` is used. This can be fixed by using the same number of epochs both during precompilation and the actual run.
-  When running multi-worker training, you may see the process getting killed at the time of model saving on trn1.2xlarge.
   This happens because the transformers ``trainer.save_model`` api uses ``xm.save`` for saving models.
   This api is known to cause high host memory usage in multi-worker setting `see Saving and Loading XLA Tensors in  <https://github.com/pytorch/xla/blob/master/API_GUIDE.md>`__ . Coupled with a compilation
   at the same time results in a host OOM. To avoid this issue, we can: Precompile all the graphs in multi-worker
   training. This can be done by running the multi-worker training first with ``neuron_parallel_compile <script>``
   followed by the actual training. This would avoid the compilation at model save during actual training.
