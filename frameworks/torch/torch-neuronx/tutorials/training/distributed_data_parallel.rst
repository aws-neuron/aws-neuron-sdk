.. _neuronx-ddp-tutorial:

Distributed Data Parallel Training Tutorial
========================================

Distributed Data Parallel (DDP) is a utility to run models in data
parallel mode. It is implemented at the module level and can help run
the model across multiple devices. As mentioned in the `DDP tutorial on
PyTorch <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__,
DDP requires applications to spawn multiple processes and then create a
single DDP instance. DDP would then make use of ``torch.distributed``
package to synchronize the gradients.

.. contents:: Table of Contents
   :local:
   :depth: 3

.. include:: ../note-performance.txt

Setup environment and download examples
---------------------------------------

Before running the tutorial please follow the installation instructions at:

* :ref:`Install PyTorch Neuron on Trn1 <pytorch-neuronx-install>`

For all the commands below, make sure you are in the virtual environment that you have created above before you run the commands:

.. code:: shell

   source ~/aws_neuron_venv_pytorch/bin/activate
    
To download the DDP examples, do:

.. code:: bash

   git clone https://github.com/aws-neuron/aws-neuron-samples.git
   cd aws-neuron-samples/torch-neuronx/training/ddp

Spawning process using xmp.spawn
--------------------------------

``xmp.spawn`` is a torch-xla utility for spawning multiple processes

Initialize the process group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import os
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.optim as optim
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.xla_backend
    
    
    def setup(init_file: str):
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
        dist.init_process_group(
            "xla",
            init_method=f"file://{init_file}" if init_file is not None else None,
            rank=rank,
            world_size=world_size)
        return rank, world_size

Build a Toy model
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)
    
        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))

Build the training function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we wrap the ``model`` with DDP wrapper. This wrapper would make the
model look like a local model, but it would wrap all the distributed
communication across devices for you.

.. code:: ipython3

    def train_fn(rank):
        setup(None)
        device = xm.xla_device()
        
        # Create the model and move to device
        model = Model().to(device)
        ddp_model = DDP(model, gradient_as_bucket_view=True)
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        num_iterations = 100
        for step in range(num_iterations):
            optimizer.zero_grad()
            outputs = ddp_model(torch.randn(20, 10).to(device))
            labels = torch.randn(20, 5).to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            xm.mark_step()
            if rank == 0:
                print(f"Loss after step {step}: {loss.cpu()}")
    
            
    def run():
        xmp.spawn(train_fn, args=())

Running the script
------------------

Copy the above methods in a script and name the script as
ddp\_xmp\_spawn.py. You can then run the script using the
following command.

::

    NEURON_NUM_DEVICES=2 python ddp_xmp_spawn.py

Using torchrun
--------------

``torchrun`` can be used for spawning processes where each process has a
model replica. This can be done with the following changes:

.. code:: ipython3

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.optim as optim
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.xla_backend
    
    torch.distributed.init_process_group('xla')

Notice how we do the ``init_process_group()`` without specifying the
``rank`` and ``worldsize``. This worldsize setting should be handled by
torchrun command.

Building the training function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice the only change is in the ``run()``. Instead of using
``xmp.spawn``, we directly call the ``train_fn``, as the process
spawning is taken care of by ``torchrun``.

.. code:: ipython3

    def train_fn():
        device = xm.xla_device()
        rank = xm.get_ordinal()
        
        # Create the model and move to device
        model = Model().to(device)
        ddp_model = DDP(model, gradient_as_bucket_view=True)
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        num_iteration = 100
        for step in range(num_iterations):
            optimizer.zero_grad()
            outputs = ddp_model(torch.randn(20, 10).to(device))
            labels = torch.randn(20, 5).to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            xm.mark_step()
            if rank == 0:
                print(f"Loss after step {step}: {loss.cpu()}")
    
            
    def run():
        train_fn()

Running the script
------------------

Running with 2 devices on a single node

::

    torchrun --nproc_per_node=2 ddp_torchrun.py

To run on multiple instances, launch 2 instances with EFA-enabled interfaces, using `EFA-enabled security group <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl-base.html#nccl-start-base-setup>`__, and setup using :ref:`Install PyTorch Neuron on Trn1 <pytorch-neuronx-install>`.
Note: Currently, we see issues with too many workers on the same node. Having 32 workers on the same node results in too many graphs, which eventually
causes some NCCL errors. This should be fixed in the future release.

On the rank-0 Trn1 host (root), run with ``--node_rank=0`` using torchrun utility, and ``--master_addr`` set to rank-0 host's IP address:

.. code:: shell

   export FI_EFA_USE_DEVICE_RDMA=1
   export FI_PROVIDER=efa
   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=<root IP> --master_port=2020 ddp_torchrun.py

On another Trn1 host, run with ``--node_rank=1``, and ``--master_addr`` also set to rank-0 host's IP address:

.. code:: shell

   export FI_EFA_USE_DEVICE_RDMA=1
   export FI_PROVIDER=efa
   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=<root IP> --master_port=2020 ddp_torchrun.py

It is important to launch rank-0 worker with ``--node_rank=0`` to avoid hang.

To train on multiple instances, it is recommended to use either a ParallelCluster or an EKS setup. For a ParallelCluster example, please see `Train a model on AWS Trn1 ParallelCluster <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples>`__ 
and for an EKS example, please see: https://github.com/aws-neuron/aws-neuron-eks-samples/tree/master/dp_bert_hf_pretrain


