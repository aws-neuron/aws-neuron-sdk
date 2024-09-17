.. _nxdt_known_issues:

Known Issues and Workarounds
============================

This section covers the common failures that one can see while working with Neuronx Distributed Training library.
Some of the failures regarding installation have been documented in :ref:`nxdt_installation_common_failures`.

.. contents:: Table of contents
   :local:
   :depth: 2

Shared weights error
--------------------

Tieing weights is not supported when using pipeline parallelism.
This means currently, the ``share_embeddings_and_output_weights`` parameter is not supported when using pipeline
parallelism. It would produce an error that looks like this

::

    File "/home/ubuntu/aws_neuron_venv_pytorch/lib/python3.8/site-packages/neuronx_distributed/pipeline/model.py", line 625, in _reduce_shared_weights
    assert p.grad is not None, f"Found shared weight {n} has None grad"
    AssertionError: Found shared weight language_model_embedding_word_embeddings.weight has None grad

Please set this flag to ``False`` when using pipeline parallelism.


HOST OOM issues
---------------

You would see an error log that looks like this without any other error above it.

::

    WARNING:torch.distributed.elastic.agent.server.api:Received 15 death signal, shutting down workers
    WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 3721028 closing signal SIGTERM
    WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 3721029 closing signal SIGTERM
    WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 3721030 closing signal SIGTERM
    WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 3721031 closing signal SIGTERM

You can confirm ``HOST OOM`` by checking ``sudo dmesg`` on the Trn1 node. ``HOST OOM`` can occur because of multiple
reasons:

During checkpoint saving
########################

If you see the above error immediately after a checkpoint saving log, this indicates that the entire checkpoint
is copied to CPU. In this case, please check if the ``save_xser`` parameter is set to ``True``. This mode will
ensure each worker saves only one tensor at a time to disk. Setting this to ``False`` will make all the workers
copy the entire checkpoint to CPU and can result in ``HOST OOM``.

During async_checkpointing
##########################

``async_checkpointing`` when used with a low number of nodes can cause ``HOST OOM`` as it increases memory pressure
per node. When we use more nodes, the memory pressure gets divided among the nodes and hence you would get an OOM.

On a high level, async checkpointing copies data from device memory to host memory, then launch a new process
to save host memory to storage, and let the main process continue with the training. Since we launch
a new process, it requires a lot more extra host memory, because the launched process has the exact copy of memory
space of the parent process. Let's use the following example to demonstrate how much memory we would need. For a llama2
70b training using tp32 on 32 nodes, we launch 32 processes on each node. As baseline, each process uses 5 GB of host
memory. There is also the XRT server, which uses 110 GB of host memory, so in total 270 GB host memory is used
(5*32 + 110). If we enable ``async_checkpointing`` on this setting, the final memory usage can reach as high as
482 GB because of the following reasons:

1. Each training process needs to allocate memory to hold the model. The model weights for llama2 70B would
require 280GB of memory to store the weights. The optimizer state would require twice as much memory. So total
amount of host memory is 840 GB. Because we used all ranks for saving, the 840GB of data was evenly distributed
among 1,024 processes (32 x 32), which means 0.84 GB of memory per process, or 26 GB of memory per instance. So
each process’s host memory usage is 5.8GB.

2. Second, each training process will fork a process for saving. The forked process will have a copy of parent’s
memory. In practice, linux uses a Copy-On-Write mechanism to save memory usage, but still in theory the actual memory
usage of the child process can reach 5.8 GB combined. When ``async_checkpointing`` is enabled, we have 64 processes
each using 5.8 GB of memory, and the XRT server uses 110 GB of memory. Therefore the total memory usage will be 482GB
(64 * 5.8 + 110).

Hence with 32 nodes, we are already on the edge (each Trn1 node has 512GB of host memory) and we could OOM at 32 nodes.
For a more stable run, enabling ``async_checkpointing`` at 64 nodes is recommended.


During Dataloading
##################

Another common reason for ``HOST OOM`` is loading too much data onto CPU. For pipeline-parallel processing, the
library loads the entire global batch onto CPU and then moves it one-by-one to device. If we have a large
batchsize with each batch taking space, it can lead to ``HOST OOM``.


ImportError: ``helpers``
------------------------

If you see an error that looks like:

::

    ImportError: cannot import name 'helpers' from 'nemo.collections.nlp.data.language_modeling.megatron' (/usr/local/lib/python3.8/dist-packages/nemo/collections/nlp/data/language_modeling/megatron/__init__.py)

This could be because the helpers.cpp didn’t get built correctly at the time of execution. We can pre-built it
by running the following code:

.. code-block:: python

    import sys
    import types

    import torch

    if torch.__version__.startswith("2"):
        string_classes = str
        inf = torch.inf
    else:
        string_classes = None
        inf = None


    # conditionally modify the import
    def modify_torch_six_import():
        if string_classes is not None:
            try:
                if "torch._six" not in sys.modules:
                    # Create and add dummy module to sys.modules
                    six_module = types.ModuleType("torch._six")
                    six_module.string_classes = string_classes
                    six_module.inf = inf
                    sys.modules["torch._six"] = six_module
            except Exception as e:
                raise RuntimeError(f"Failed to override torch._six import: {e}")

    modify_torch_six_import()
    from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper
    compile_helper()


Alternatively, if you see

::

    ImportError: /shared/username/aws_neuron_venv_pytorch/lib/python3.10/site-packages/nemo/collections/nlp/data/language_modeling/megatron/helpers.cpython-310-x86_64-linux-gnu.so: file too short

A current workaround for this case is to delete the .so file and run the above snippet explicitly.

Matplotlib error
----------------

If you see an error that looks like:

::

    TimeoutError: Lock error: Matplotlib failed to acquire the following lock file

It means there is some contention in compute/worker nodes to access the matlotlib cache, and hence the lock error.
To resolve this add or run ``python -c 'import matplotlib.pyplot as plt'`` as part of your setup. This will
create a matplotlib cache and avoid the race condition.

Flash Attention not supported for megatron-style models
-------------------------------------------------------

Flash attention kernel is supported only for HF-style models and will be added for megatron-style models in one of
the future releases.
