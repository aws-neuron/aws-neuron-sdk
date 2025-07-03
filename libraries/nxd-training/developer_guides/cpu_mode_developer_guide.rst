.. _cpu_mode_overview:

CPU Mode Overview
=================

CPU mode allows users to run parallel primitives
like `RowParallelLinear` and `ColumnParallelLinear` on CPU. This is useful
when debugging or developing model sharding and want to check the intermediate results 
of sharded layers. The CPU mode runs in PyTorch's eager mode and does not require
the compilation steps of torch-xla and Neuron compiler. The collective communications
like all-reduce use the PyTorch's 
`gloo backend <https://pytorch.org/docs/stable/distributed.html#backends-that-come-with-pytorch>`_
for communications.

To enable the CPU mode, we need to set the environment variable `NXD_CPU_MODE=1` to 
enable the CPU mode. As the CPU mode leverages Gloo backend for communication, users 
need to initialize the distributed environment with "gloo" backend instead of "xla" backend.
In the following, we given an example of a MLP with Tensor Parallel linear layers. 

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

If the scripts previously used the `xla_device` explicitly, 
users need to replace the corresponding use of `xla_device` with 
`get_device()` function call from `neuronx_distributed.utils` to get the suitable device. 
Similarly, you need to replace explicit calling of `xm.master_print` with wrapped `master_print`
from `neuronx_distributed.utils`. In principle, to make the 
scripts general to both CPU mode and XLA mode with Trainium as the backend, you 
need to replace functions from torch-xla package with a thin wrapper that can 
dispatch the function calls to the native PyTorch counterparts, when CPU mode 
is in-use.
