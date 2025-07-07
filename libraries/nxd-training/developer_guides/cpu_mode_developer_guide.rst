.. _cpu_mode_overview:

This topic discusses the use of CPU mode for processing and debugging parallel primitives using PyTorch without compilation steps. 

CPU Mode Overview
=================

Use CPU mode to run parallel primitives like `RowParallelLinear` and `ColumnParallelLinear` on a compute instance's CPU when
debugging or developing model sharding and check the intermediate results  of sharded layers. CPU mode runs in PyTorch's "eager" mode and does not require the compilation steps of torch-xla and the Neuron compiler. 

Collective communications like all-reduce use PyTorch's  `Gloo backend <https://pytorch.org/docs/stable/distributed.html#backends-that-come-with-pytorch>`_
for communication. Since CPU mode leverages the Gloo backend for communication, you must initialize the distributed environment with the "gloo" backend instead of the "xla" backend. To enable CPU mode, set the environment variable `NXD_CPU_MODE=1`.

Here is an example of a multi-layer perceptron (MLP) built with Tensor Parallel linear layers, configured to use CPU mode to process them without prior compilation: 

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.parallel_layers import initialize_model_parallel
    from neuronx_distributed.utils import cpu_mode, get_device, master_print

    # initialize the distributed environment inside PyTorch
    cc_backend = "gloo" if cpu_mode() else "xla"
    dist.init_process_group(backend=cc_backend)

    # assuming sharding the model with TP=2
    initialize_model_parallel(tensor_model_parallel_size=2)

    hidden_size = 1024
    rand_inputs = torch.rand(4, hidden_size)
    model = nn.Sequential(
        layers.ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            keep_master_weight=True,
        ),
        layers.RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            keep_master_weight=True,
        ),
    )
    model = model.to(get_device())
    rand_inputs = rand_inputs.to(get_device())

    outputs = model(rand_inputs)
    # user can check the outputs are on the CPU
    # and there is no compilation triggered
    master_print(f"Output sum is {outputs.sum()}")


.. code-block:: bash

    # set the environment variable to enable CPU mode
    # if the environment variable is set to 0, 
    # the script will run on Trainium accelerator using XLA
    export NXD_CPU_MODE=1
    # assumign the script show above is saved in test_cpu_mode.py
    exec_file=test_cpu_mode.py
    torchrun --nnodes=1 --nproc-per-node=2 --master_port=1234 ${exec_file}


How to use CPU mode in existing scripts
---------------------------------------

If your scripts previously used the `xla_device` explicitly, 
you must replace the corresponding use of `xla_device` with the 
`get_device()` function call from `neuronx_distributed.utils` to get the suitable device.

To make the scripts general to both CPU mode and XLA mode, and with Trainium as the backend, you 
must replace functions from the `torch-xla` package with a thin wrapper that can 
dispatch the function calls to the native PyTorch counterparts when CPU mode 
is in use. For example, you must replace explicit calls to `xm.master_print` with calls to a wrapped version of `master_print` 
from `neuronx_distributed.utils`. 
