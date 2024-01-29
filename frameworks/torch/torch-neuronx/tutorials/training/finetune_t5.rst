.. _torch-hf-t5-finetune:

Fine-tune T5 model on Trn1
================================

.. note:: 
   Update 01/03/24: This tutorial is currently broken and the AWS Neuron team is working on the fix.


In this tutorial, we show how to fine-tune a Hugging Face (HF) T5 model 
using HF trainer API. This example fine-tunes a `T5 model for
a text-summarization <https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization>`__ task on CNN/DailyMail dataset.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. include:: ../note-performance.txt

Setup and compilation
---------------------

Before running the tutorial please follow the installation instructions at:

:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`

Please set the storage of instance to *512GB* or more if you also want to run through the BERT pretraining and GPT pretraining tutorials.

For all the commands below, make sure you are in the virtual environment that you have created above before you run the commands:

.. code:: shell

   source ~/aws_neuron_venv_pytorch/bin/activate

First we install a recent version of HF transformers, scikit-learn and evaluate packages in our environment as well as download the source matching the installed version. In this example, we chose version 4.26.0 and the text summarization example from HF transformers source:

.. code:: bash

    export HF_VER=4.26.0
    pip install -U transformers==$HF_VER datasets evaluate scikit-learn rouge_score pandas==1.4.0
    cd ~/
    git clone https://github.com/huggingface/transformers --branch v$HF_VER
    cd ~/transformers/examples/pytorch/summarization


Single-worker training
----------------------

We will run text-summarization fine-tuning task following the example in
README.md located in the path
`~/transformers/examples/pytorch/summarization.`

We use full BF16 casting using `XLA_USE_BF16=1` to enable best
performance. First, paste the following script into your terminal to
create a “run.sh” file and change it to executable:

.. code:: ipython3

    tee run.sh > /dev/null <<EOF
    #!/bin/bash
    if [ \$NEURON_PARALLEL_COMPILE == "1" ]
    then
        XLA_USE_BF16=1 python3 ./run_summarization.py \
        --model_name_or_path t5-small \
        --dataset_name cnn_dailymail \
        --dataset_config "3.0.0" \
        --do_train \
        --do_eval \
        --source_prefix "summarize: " \
        --max_source_length 512 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --overwrite_output_dir \
        --pad_to_max_length \
        --max_steps 100 \
        --max_eval_samples 100 \
        --gradient_accumulation_steps=32 \
        --output_dir /tmp/tst-summarization |& tee log_run
    else
        XLA_USE_BF16=1 python3 ./run_summarization.py \
        --model_name_or_path t5-small \
        --dataset_name cnn_dailymail \
        --dataset_config "3.0.0" \
        --do_train \
        --do_eval \
        --source_prefix "summarize: " \
        --max_source_length 512 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --overwrite_output_dir \
        --pad_to_max_length \
        --gradient_accumulation_steps=32 \
        --output_dir /tmp/tst-summarization |& tee log_run
    fi
    EOF
    
    chmod +x run.sh

We optionally precompile the model and training script using
`neuron\_parallel\_compile <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html?highlight=neuron_parallel_compile>`__ to warm up the persistent graph cache (Neuron
Cache) such that the actual run has fewer compilations (faster run
time):

.. code:: ipython3

    neuron_parallel_compile ./run.sh

Note: For these auto-regressive models, do not run the
``predict_with_generate`` method when doing the precompile step. This is
because the ``neuron_parallel_compile`` utility will run the training
script in graph extraction mode and no actual execution of the graph
will be done. Hence, the outputs at each step are invalid. Since the
auto-regressive generation at each step is dependent on output of
previous step, the generate step would fail since the outputs from
previous steps are invalid.

Precompilation is optional and only needs to be done once unless
hyperparameters such as batch size are modified. After the optional
precompilation, the actual run will be faster with minimal additional
compilations.

.. code:: ipython3

    ./run.sh

If precompilation was not done, the first execution of ./run.sh will be
slower due to serial compilations. Rerunning the same script a second
time would show quicker execution as the compiled graphs will be already
cached in persistent cache.

Running the above script will run the T5-small fine-tuning on a single
process.

**Note:** As you may have noticed, we are not running the
``predict_with_generate`` as part of training. This is because,
``predict_with_generate`` requires auto-regressive sampling where the
inputs to the decoder are created by appending outputs of previous
steps. This causes the inputs to the decoder to change shape and thereby
resulting in a new graph. In other words, the current ``generate`` api
provided by HF transformers leads to repeated compilations. We are working on
building a Neuron friendly version of ``generate`` api and it will be
made available as part of future release. This will enable us to run
``predict_with_generate`` as part of training script.

As a workaround, we can run the ``predict_with_generate`` on CPU after
the model is trained. Once training is completed, a trained checkpoint
would be saved. We can load the trained model and run the
``predict_with_generate`` to compute the final accuracy.

To do so, in run_summarization.py, add the following before ``transformers`` get imported.
This can be done by adding the below lines before all the ``imports``:

.. code:: ipython3

    import libneuronxla
    # Disable configuring xla env
    def _configure_env():
        pass
    libneuronxla.configure_environment = _configure_env

You can now run the following and it should run the predict method on CPU device.

.. code:: ipython3

    NEURON_NUM_DEVICES=0 python3 ./run_summarization.py \
        --model_name_or_path <CHECKPOINT_DIR> \
        --dataset_name cnn_dailymail \
        --dataset_config "3.0.0" \
        --do_predict \
        --predict_with_generate \
        --source_prefix "summarize: " \
        --per_device_eval_batch_size 4 \
        --max_source_length 512 \
        --pad_to_max_length \
        --no_cuda \
        --output_dir /tmp/tst-summarization |& tee log_run

Note: To run on CPU, we need to make sure that NEURON\_NUM\_DEVICES is
set to 0. This will make sure no xla\_devices are created and the
trainer would use the default device (CPU).

.. _multi_worker_training:

Multi-worker Training
---------------------

The above script will run one worker on one NeuronCore. To run on
multiple cores, first add these lines to top of run\_summarization.py to disable
Distributed Data Parallel (DDP) when using torchrun (see Known issues
and limitations section below):

.. code:: ipython3

    # Disable DDP for torchrun
    from transformers import __version__, Trainer
    Trainer._wrap_model = lambda self, model, training=True, dataloader=None: model

Then launch the run\_summarization.py script with torchrun using
--nproc\_per\_node=N option to specify the number of workers (N=2 for
trn1.2xlarge, and N=2, 8, or 32 for trn1.32xlarge). The following
example runs 2 workers. Paste the following script into your terminal to
create a “run\_2w.sh” file and change it to executable:

.. code:: ipython3

    tee run_2w.sh > /dev/null <<EOF
    #!/bin/bash
    if [ \$NEURON_PARALLEL_COMPILE == "1" ]
    then
        XLA_USE_BF16=1 torchrun --nproc_per_node=2 ./run_summarization.py \
        --model_name_or_path t5-small \
        --dataset_name cnn_dailymail \
        --dataset_config "3.0.0" \
        --do_train \
        --do_eval \
        --source_prefix "summarize: " \
        --max_source_length 512 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --overwrite_output_dir \
        --pad_to_max_length \
        --max_steps 100 \
        --max_eval_samples 100 \
        --gradient_accumulation_steps=32 \
        --output_dir /tmp/tst-summarization |& tee log_run
    else
        XLA_USE_BF16=1 torchrun --nproc_per_node=2 ./run_summarization.py \
        --model_name_or_path t5-small \
        --dataset_name cnn_dailymail \
        --dataset_config "3.0.0" \
        --do_train \
        --do_eval \
        --source_prefix "summarize: " \
        --max_source_length 512 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --overwrite_output_dir \
        --pad_to_max_length \
        --gradient_accumulation_steps=32 \
        --output_dir /tmp/tst-summarization |& tee log_run
    fi
    EOF
    
    chmod +x run_2w.sh

Again, we optionally precompile the model and training script using
neuron\_parallel\_compile to warm up the persistent graph cache (Neuron
Cache), ignoring the results from this precompile run as it is only for
extracting and compiling the XLA graphs:

.. code:: ipython3

    neuron_parallel_compile ./run_2w.sh

Precompilation is optional and only needs to be done once unless
hyperparameters such as batch size are modified. After the optional
precompilation, the actual run will be faster with minimal additional
compilations.

.. code:: ipython3

    ./run_2w.sh

During run, you will notice that the “Total train batch size” is now
8 and the “Total optimization steps” is now half the number for one
worker training. Also, if you open ``neuron-top`` in a separate terminal, 
you should see 2 cores been utilized.

To train T5-large model, you can set the ``model_name_or_path`` argument to ``t5-large``.
Please note, currently running ``t5-large`` on trn1-2xl machine can result in ``HOST OOM`` during 
compilation. Hence, it is recommended that you run a ``t5-large`` model training on a trn1-32xl machine.

On a trn1-32xl machine, you can create a run_32w.sh on the terminal using the following commands:

.. code:: ipython3

    tee run_32w.sh > /dev/null <<EOF
    #!/bin/bash
    if [ \$NEURON_PARALLEL_COMPILE == "1" ]
    then
        XLA_USE_BF16=1 torchrun --nproc_per_node=32 ./run_summarization.py \
        --model_name_or_path t5-large \
        --dataset_name cnn_dailymail \
        --dataset_config "3.0.0" \
        --do_train \
        --do_eval \
        --source_prefix "summarize: " \
        --max_source_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --overwrite_output_dir \
        --pad_to_max_length \
        --max_steps 100 \
        --max_eval_samples 100 \
        --gradient_accumulation_steps=11 \
        --output_dir /tmp/tst-summarization |& tee log_run
    else
        XLA_USE_BF16=1 torchrun --nproc_per_node=32 ./run_summarization.py \
        --model_name_or_path t5-large \
        --dataset_name cnn_dailymail \
        --dataset_config "3.0.0" \
        --do_train \
        --do_eval \
        --source_prefix "summarize: " \
        --max_source_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --overwrite_output_dir \
        --pad_to_max_length \
        --gradient_accumulation_steps=11 \
        --output_dir /tmp/tst-summarization |& tee log_run
    fi
    EOF
    
    chmod +x run_32w.sh

You can now follow the same steps as listed above. This script would run a t5-large model by launching a training script 
using 32 data-parallel workers.


.. _known_issues:

Known issues and limitations
----------------------------

The following are currently known issues:

-  Long compilation times: this can be alleviated with
   ``neuron_parallel_compile`` tool to extract graphs from a short trial run and
   compile them in parallel ahead of the actual run, as shown above.
- T5-Large compilation causing processes to get killed on trn1-2xl: It is recommended 
  to ``t5-large`` model training on a trn1-32xl machine, as it avoids CPU OOM and also provides 
  faster training by making use of 32 data-parallel workers.
