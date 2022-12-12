.. _megatron-lm-pretraining-tutorial:

Megatron-LM GPT Pretraining Tutorial
====================================

GPT is a large language model that excels at many natural language
processing (NLP) tasks. It is derived from the decoder part of the
Transformer. `Neuron Reference For Megatron-LM <https://github.com/aws-neuron/aws-neuron-reference-for-megatron-lm>`__ is a library
that enables large-scale distributed training of language models such as
GPT and is adapted from `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`__.
This tutorial explains how to run the Neuron reference for Megatron-LM GPT pretraining on Trainium.

The AWS Neuron SDK provides access to Trainium devices through an
extension of PyTorch/XLA - a library that includes the familiar PyTorch
interface along with XLA-specific additions. For Trainium customers,
this means that existing PyTorch training scripts can be executed on
Trn1 instances with minimal code modifications. For additional details
relating to PyTorch/XLA, please refer to the `official PyTorch/XLA
documentation <https://pytorch.org/xla>`__.

To run on Trainium, Neuron Reference For Megatron-LM library includes the following changes:

-  GPU devices are replaced with Pytorch/XLA devices.
-  Pytorch/XLA distributed backend is used to bridge the PyTorch distributed
   APIs to XLA communication semantics.
-  Pytorch/XLA MpDeviceLoader is used for the data ingestion pipelines.
   Pytorch/XLA MpDeviceLoader helps improve performance by overlapping the three
   execution steps: tracing, compilation and data batch loading to the
   device.
-  CUDA APIs are mapped to generic PyTorch APIs.
-  CUDA fused optimizers are replaced with generic PyTorch alternatives.

The GPT example in this tutorial is an adaptation of the original
Megatron-LM GPT example, trained using the Wikipedia dataset.

.. contents:: Table of Contents
   :local:
   :depth: 3

.. include:: ../note-performance.txt


Install PyTorch Neuron
~~~~~~~~~~~~~~~~~~~~~~

Before running the tutorial please follow the installation instructions at:

* :ref:`Install PyTorch Neuron on Trn1 <pytorch-neuronx-install>`

Please set the storage of instance to *512GB* or more if you intent to run multiple experiments and save many checkpoints.

Download Preprocessed Wikipedia Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the vocabulary file, the merge table file, and the preprocessed Wikipedia dataset using the following commands:

::

   export DATA_DIR=~/examples_datasets/gpt2

   mkdir -p ${DATA_DIR} && cd ${DATA_DIR}

   wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
   wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
   aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.bin .  --no-sign-request
   aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.idx .  --no-sign-request
   aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/license.txt .  --no-sign-request

See section ``Preparing Wikipedia dataset from scratch`` if you would like to recreate the preprocessed dataset from scratch.

Setting up the training environment on trn1.32xlarge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please follow the :ref:`instructions <pytorch-neuron-setup>` to setup Python virtual environment
with Neuron packages.

Install Python3 development package needed to build the data helpers tools. If you are on Amazon Linux, do:

::

    sudo yum install -y python3-devel

If you are on Ubuntu, do:

::

    sudo apt install -y python3-dev

Clone the AWS Neuron Reference for Megatron-LM package, install dependencies, and build the data helpers tool:

::

    cd ~/
    git clone https://github.com/aws-neuron/aws-neuron-reference-for-megatron-lm.git
    pip install pybind11 regex
    pushd .
    cd aws-neuron-reference-for-megatron-lm/megatron/data/
    make
    popd

GPT Pretraining Python Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPT pretraining python script is a wrapper that imports the Megatron-LM
library modules and sets up the pieces needed by the Megatron-LM
trainer: GPT model, loss function, forward pass, data provider.
It is adapted from `pretrain_gpt.py <https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py>`__. The
Neuron changes are:

-  Use XLA device
-  Not using mpu.broadcast_data as it is currently unsupported. Instead
   each worker reads the data in parallel.
-  Use int instead of long datatype for token data

The script is available at ``~/aws-neuron-reference-for-megatron-lm/pretrain_gpt.py``

GPT Training Shell Script
~~~~~~~~~~~~~~~~~~~~~~~~~

The GPT training shell script runs the above python script with
following model configurations (for 6.7 billion parameters model):

-  Number of layers: 32
-  Hidden size: 4096
-  Number attention heads: 32
-  Sequence length: 2048
-  Max positional embeddings size: 2048

The following training parameters are used:

-  The number of gradient accumulation microsteps is 64, with worker
   batch size of 1.
-  The tensor parallelism degree is 8.
-  The data parallelism degree is 4.
-  The number of workers is 32.

Additionally, the script uses:

-  CPU intitialization
-  AdamW optimizer (default).
-  Gradient clipping.
-  No CUDA fusions (bias-gelu, masked-softmax, bias-dropout)
-  Disabled contiguous buffer in local DDP 
-  Option ``--distributed-backend xla`` picks the XLA distributed backend
   to bridge the Pytorch distributed APIs to XLA
   communication semantics.

See `this link <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py>`__ for a full list of options and their descriptions.

.. note::

    Not all options are supported. Currently only tensor-parallel and data-parallel modes
    in Neuron Reference For Megatron-LM are supported. We support tensor-parallel degree of 8 
    and data-parallel degree of upto 64. 

The script for running on a single node is available at
``~/aws-neuron-reference-for-megatron-lm/examples/pretrain_gpt3_6.7B_32layers_bf16.sh``

This shell script expects dataset files to be located in ~/examples_datasets/gpt2/ following the steps above. If you place the dataset files in another location, please update the DATA_PATH variable in the shell script.

Initiating a Training Job
~~~~~~~~~~~~~~~~~~~~~~~~~

To run the GPT example, first activate the Python virtual environment
and change to the Megatron-LM package location:

::

   source ~/aws_neuron_venv_pytorch/bin/activate
   cd ~/aws-neuron-reference-for-megatron-lm/

Next, run the parallel compilations of graphs in order to reduce
compilation time during the actual run.

::

   neuron_parallel_compile sh ./examples/pretrain_gpt3_6.7B_32layers_bf16.sh

This command performs a short trial run of the training script to
extract graphs and then do parallel compilations on those graphs before
populating the persistent cache with compiled graphs. This helps reduce
the compilation time during the actual run of the training script. 

.. note::

	Please ignore the results of the trial run as they are not the actual
	execution results.

If some or all the graphs were already compiled and cached in
the persistent cache, then fewer or none of the graphs would need
compilation. To force recompilation, you can remove the cache directory
at ``/var/tmp/neuron-compile-cache/.``

Compilation is recommended if there are some changes in the script (such
as batch size, number of layers, number of workers, etc.). Compilation will only happen if the model graph or its parameters/compilation flags change.

Finally, run the script for the actual run:

::

   sh ./examples/pretrain_gpt3_6.7B_32layers_bf16.sh

During the run, you will see outputs like below, some lines showing
throughput and loss statistics every global step.

::

   `iteration     4873/   10000 | consumed samples:       311872 | elapsed time per iteration (ms): 8718.9 | learning rate: 1.500E-04 | global batch size:    64 | lm loss: 3.296875E+00 | grad norm: 0.430 | throughput: 7.340`

Monitoring Training Job Progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a single Trn1 instance with 32 NeuronCores, the current GPT
pretraining will run for ~81 hours. During this time, you will see the
average loss metric begin at 11 and ultimately converge to ~3.2.
Throughput for the training job will be ~7.3 seq/sec.

Monitoring Training Job Progress using neuron-top
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the training job still running, launch a second SSH connection into
the trn1 instance, and use the ``neuron-top`` command to examine the
aggregate NeuronCore utilization.

Monitoring Training Job Progress using TensorBoard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The demo includes TensorBoard-compatible logging, which allows the
learning rate and training metrics to be monitored in real-time. By
default, the training script logs metrics to the following TensorBoard
log directory ``~/aws-neuron-reference-for-megatron-lm/tb_*``.

In order to view your training metrics in TensorBoard, first run the
following commands in your SSH session:

::

   source ~/aws_neuron_venv_pytorch/bin/activate
   cd ~/aws-neuron-reference-for-megatron-lm/
   tensorboard --logdir ./

Once running, open a new SSH connection to the instance and port-forward
TCP port 6006 (ex: -L 6006:127.0.0.1:6006). Once the tunnel is
established, TensorBoard can then be accessed via web browser at the
following URL: `http://localhost:6006 <http://localhost:6006/>`__.
Please note that you will not be able to access TensorBoard if you
disconnect your port-forwarding SSH session to the Trainium instance.

Finishing the tutorial
~~~~~~~~~~~~~~~~~~~~~~

Once you are ready, and the
training throughput is as expected, there are a couple of options for
finishing the GPT pretraining demo:

**Allow the training script to run to completion**. If you would like to
observe the training script run to completion, it is recommended to
launch the training script from a terminal multiplexer such as ``tmux``
or ``screen``, and then detach the session so that the training script
can run in the background. With this approach, you can safely let the
training script run unattended, without risk of an SSH disconnection
causing the training job to stop running.

**Stop the training job early**. To stop the training job early, press
CTRL-C in the terminal window in which you launched the training script.
In some cases, if you manually cancel a job using CTRL-C and then later
want to run the job again, you might first need to terminate all the
python processes by the command ``killall -9 python3`` .


Running a multi-node GPT 
~~~~~~~~~~~~~~~~~~~~~~~~

We use SLURM to launch multi-node GPT training jobs. Like single node runs, 
we have a precompilation step followed by the actual run. To precompile:

::
   
   sbatch examples/pretrain_gpt3_6.7B_compile.slurm

This will precompile the script ``examples/pretrain_gpt3_6.7B_32layers_bf16_bs1024_slurm.sh`` 
on all the nodes and populate the caches.

To run the compiled model:

::
   
   sbatch examples/pretrain_gpt3_6.7B.slurm

The number of nodes is currently set to 16 and since the tensor-parallel degree used is
8, the data-parallel degree is automatically computed to be 64, resulting in a 8x64 two
dimensional mesh parallelism.

The tensorboard logs are written by the last rank and will be in the TensorBoard
log directory ``~/aws-neuron-reference-for-megatron-lm/tb_*``.

Compared to the single-node script, we use an increased batch size of 1024 which gives us
a throughput bump of ~98 seq/sec. The number of iterations is also increased with changes 
in the hyperparameters pertaining to learning rates, weight decay.

Checkpointing GPT Model
~~~~~~~~~~~~~~~~~~~~~~~

A new mode of checkpointing using serialized tensor and staggered save/load is supported
to alleviate memory pressure. To save the model, add the lines:

::
   
   --save-xser $CHECKPOINT_PATH
   --save-interval 1500

This will save the checkpoint at path variable provided for every 1500 iterations. 

.. note::

	Please note that the model saves all the model weights, optimizer and rng states (~76GB for a
	32 layermodel). And if checkpointed frequently can quickly lead to low disk storage. 
	Make sure there is enough disk space. 


To load the checkpoint, we first need to remove ``--use-cpu-initialization`` from the script 
and then add

::

 --load-xser $CHECKPOINT_PATH

.. note::

	Please note not removing the --use-cpu-initialization flag may lead to out-of-memory
	execution and result in unstable resumption of training.



Preparing Wikipedia Dataset from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process of preparing the Wikipedia dataset follows the original
`Megatron-LM
documentation <https://github.com/NVIDIA/Megatron-LM#user-content-datasets>`__. You
will need a large c5 machine like c5n.18xlarge and using the latest Deep
Learning AMI. First download the Wikipedia dataset. Depending on
the network bandwidth, this is expected to be about ~65 minutes.

::

   export WIKI_DIR=~/examples_datasets/wiki
   mkdir -p $WIKI_DIR && cd $WIKI_DIR

   wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

Download the vocabulary and merge table files for the desired model. This
example uses the GPT-2 model:

::

   export DATA_DIR=~/examples_datasets/gpt2
   export GPT2_DATA=${DATA_DIR}/gpt2

   mkdir -p ${GPT2_DATA} && cd ${GPT2_DATA}

   wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
   wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

   mkdir -p ${GPT2_DATA}/checkpoint
   wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O ${GPT2_DATA}/checkpoint/megatron_lm_345m_v0.0.zip

Extract the downloaded data using WikiExtractor (this step takes about 2
hours):

::

   git clone https://github.com/attardi/wikiextractor.git /tmp/wikiextractor
   cd /tmp/wikiextractor
   python -m wikiextractor.WikiExtractor --json ~/examples_datasets/wiki/enwiki-latest-pages-articles.xml.bz2 --output ~/examples_datasets/wiki/text/ -q --processes 70 2>&1 | tee wikiextract.out &

The Wikiextractor first preprocesses the template of all pages
sequentially, followed by a Map/Reduce process for extracting the pages
and converting to the loose json format required by Megatron-LM.

Once the extraction completes, we merge the text files with (~2
minutes):

::

   conda activate pytorch_latest_p37
   cd ~/examples_datasets/wiki
   find ~/examples_datasets/wiki/text/ -name wiki* | parallel -m -j 70 "cat {} >> mergedfile.json"

The ``mergedfile.json`` size on disk is 16GB. With it, create the binary
data format for Megatron GPT2. NOTE: Refer to `this
solution <https://github.com/NVIDIA/Megatron-LM/issues/62>`__ if an
``IndexError: list index out of range`` occurs. To create the binary
data, type the following command:

::

   python ~/aws-neuron-reference-for-megatron-lm/tools/preprocess_data.py \
       --input ~/examples_datasets/wiki/mergedfile.json \
       --output-prefix my-gpt2 \
       --vocab ~/examples_datasets/gpt2/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ~/examples_datasets/gpt2/gpt2-merges.txt \
       --append-eod \
       --workers 70

Files my-gpt2_text_document.\* are generated after about 12 minutes.

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No broadcast support
--------------------

Currently, the mpu.broadcast_data is unsupported on Trainium.

No pipeline parallel support
-------------------------------------------

Currently, only tensor parallel and data parallel are supported and there is no 
pipeline parallel support in Neuron Reference For Megatron-LM.

Dropout is disabled
-------------------

Currently, dropout is disabled in the example.

Error: cannot import name 'helpers' from 'megatron.data'
--------------------------------------------------------

You may encounter the error "cannot import name 'helpers' from 'megatron.data'" like below:

.. code:: bash

    Exception in device=NEURONT:0: cannot import name 'helpers' from 'megatron.data' (/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/data/__init__.py)
    Traceback (most recent call last):
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37/lib64/python3.7/site-packages/torch_xla/distributed/xla_multiprocessing.py", line 373, in _mp_start_fn
        _start_fn(index, pf_cfg, fn, args)
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37/lib64/python3.7/site-packages/torch_xla/distributed/xla_multiprocessing.py", line 367, in _start_fn
        fn(gindex, *args)
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/pretrain_gpt_mp.py", line 138, in pretrain_mp
        forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/training.py", line 162, in pretrain
        train_valid_test_dataset_provider)
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/training.py", line 1021, in build_train_valid_test_data_iterators
        train_val_test_num_samples)
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/pretrain_gpt_mp.py", line 128, in train_valid_test_datasets_provider
        skip_warmup=(not args.mmap_warmup))
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/data/gpt_dataset.py", line 43, in build_train_valid_test_datasets
        seq_length, seed, skip_warmup)
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/data/gpt_dataset.py", line 118, in _build_train_valid_test_datasets
        train_dataset = build_dataset(0, 'train')
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/data/gpt_dataset.py", line 115, in build_dataset
        seq_length, seed)
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/data/gpt_dataset.py", line 156, in __init__
        num_samples, seq_length, seed)
      File "/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/data/gpt_dataset.py", line 274, in _build_index_mappings
        from megatron.data import helpers
    ImportError: cannot import name 'helpers' from 'megatron.data' (/home/ec2-user/aws-neuron-reference-for-megatron-lm/megatron/data/__init__.py)

To fix this, please go into aws-neuron-reference-for-megatron-lm/megatron/data/ and do "make":

.. code:: bash

   pip install pybind11
   pushd .
   cd aws-neuron-reference-for-megatron-lm/megatron/data/
   make
   popd

Error: Out of space while checkpointing
--------------------------------------------------------

You may seem an error as follows. The model checkpoints are large as they dump all the model weights,
optimizer and rng states. And if these are frequently checkpointed, the storage can run out fast.
Please make sure you have enough disk space.

.. code:: bash

   Traceback (most recent call last):
     File "/home/ec2-user/aws_neuron_venv_pytorch_p37/lib64/python3.7/site-packages/torch/serialization.py", line 380, in save
       _save(obj, opened_zipfile, pickle_module, pickle_protocol)
     File "/home/ec2-user/aws_neuron_venv_pytorch_p37/lib64/python3.7/site-packages/torch/serialization.py", line 604, in _save
       zip_file.write_record(name, storage.data_ptr(), num_bytes)
   OSError: [Errno 28] No space left on device


Troubleshooting
~~~~~~~~~~~~~~~

See :ref:`pytorch-neuron-traning-troubleshooting`
