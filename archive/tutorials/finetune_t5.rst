.. _torch-hf-t5-finetune:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.

Fine-tune T5 model on Trn1
================================

.. note:: 
   This page was archived on 7/31/2025.


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

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_setup_code.sh
   :language: shell
   :lines: 5-9

Single-worker training
----------------------

We will run text-summarization fine-tuning task following the example in
README.md located in the path
`~/transformers/examples/pytorch/summarization.`

We use full BF16 casting using `XLA_USE_BF16=1` to enable best
performance. First, paste the following script into your terminal to
create a “run.sh” file and change it to executable:

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_single_worker_training_code.sh
   :language: shell
   :lines: 7-46

We optionally precompile the model and training script using
`neuron\_parallel\_compile <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html?highlight=neuron_parallel_compile>`__ to warm up the persistent graph cache (Neuron
Cache) such that the actual run has fewer compilations (faster run
time):

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_single_worker_training_code.sh
   :language: shell
   :lines: 49

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

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_single_worker_training_code.sh
   :language: shell
   :lines: 51

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

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_single_worker_training_code.sh
   :language: python
   :lines: 55-59

You can now run the following and it should run the predict method on CPU device.

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_single_worker_training_code.sh
   :language: shell
   :lines: 67-78

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

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_modify_run_summarization_code.sh
   :language: python
   :lines: 8-10

Then launch the run\_summarization.py script with torchrun using
--nproc\_per\_node=N option to specify the number of workers (N=2 for
trn1.2xlarge, and N=2, 8, or 32 for trn1.32xlarge). The following
example runs 2 workers. Paste the following script into your terminal to
create a “run\_2w.sh” file and change it to executable:

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_multi_worker_training_code.sh
   :language: shell
   :lines: 7-46

Again, we optionally precompile the model and training script using
neuron\_parallel\_compile to warm up the persistent graph cache (Neuron
Cache), ignoring the results from this precompile run as it is only for
extracting and compiling the XLA graphs:

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_multi_worker_training_code.sh
   :language: python
   :lines: 49

Precompilation is optional and only needs to be done once unless
hyperparameters such as batch size are modified. After the optional
precompilation, the actual run will be faster with minimal additional
compilations.

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_multi_worker_training_code.sh
   :language: python
   :lines: 51

During run, you will notice that the “Total train batch size” is now
8 and the “Total optimization steps” is now half the number for one
worker training. Also, if you open ``neuron-top`` in a separate terminal, 
you should see 2 cores been utilized.

To train T5-large model, you can set the ``model_name_or_path`` argument to ``t5-large``.
Please note, currently running ``t5-large`` on trn1-2xl machine can result in ``HOST OOM`` during 
compilation. Hence, it is recommended that you run a ``t5-large`` model training on a trn1-32xl machine.

On a trn1-32xl machine, you can create a run_32w.sh on the terminal using the following commands:

.. literalinclude:: tutorial_source_code/t5_finetuning/t5_finetuning_32_worker_training_code.sh
   :language: shell
   :lines: 7-46

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
