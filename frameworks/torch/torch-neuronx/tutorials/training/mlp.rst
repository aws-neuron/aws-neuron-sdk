.. _neuronx-mlp-training-tutorial:

Multi-Layer Perceptron Training Tutorial
========================================



MNIST is a standard dataset for handwritten digit recognition. A
multi-layer perceptron (MLP) model can be trained with MNIST dataset to
recognize hand-written digits. This tutorial starts with a 3-layer MLP
training example in PyTorch on CPU, then show how to modify it to run on
Trainium using PyTorch Neuron. It also shows how to do multiple worker
data parallel MLP training.



.. contents:: Table of Contents
   :local:
   :depth: 2

.. include:: ../note-performance.txt

Setup environment and download examples
---------------------------------------

Before running the tutorial please follow the installation instructions at:

* :ref:`Install PyTorch Neuron on Trn1 <pytorch-neuronx-install>`

Please set the storage of instance to *512GB* or more if you also want to run through the BERT pretraining and GPT pretraining tutorials.

For all the commands below, make sure you are in the virtual environment that you have created above before you run the commands:

.. code:: shell

   source ~/aws_neuron_venv_pytorch_p37/bin/activate

Install needed dependencies in your environment by running:

.. code:: bash

    pip install pillow torchvision==0.12 --no-deps
    
To download the MNIST MLP examples, do:

.. code:: bash

   git clone https://github.com/aws-neuron/aws-neuron-samples.git
   cd aws-neuron-samples/torch-neuronx/training/mnist_mlp

Multi-layer perceptron MNIST model
----------------------------------

In ``model.py``, we define the multi-layer perceptron (MLP) MNIST model with 3
linear layers and ReLU activations, followed by a log-softmax layer.
This model will be used in multiple example scripts.

Single-worker MLP training script in PyTorch on CPU
---------------------------------------------------

We will show how to modify a training script that runs on other platform to run on Trainium.

We begin with a single-worker MLP training script for running on
the host CPUs of the Trainium instance. The training script imports the
MLP model from ``model.py``.

In this training script, we load the MNIST train dataset and, within the
``main()`` method, set the data loader to read batches of 32 training
examples and corresponding labels.

Next we instantiate the MLP model and move it to the device. We use
``device = 'cpu'`` to illustrate the use of device in PyTorch. On GPU
you would use ``device = 'cuda'`` instead.

We also instantiate the other two components of a neural network
trainer: stochastic-gradient-descent (SGD) optimizer and
negative-log-likelihood (NLL) loss function (also known as cross-entropy
loss).

After the optimizer and loss function, we create a training loop to iterate over the training samples and
labels, performing the following steps for each batch in each iteration:

-  Zero gradients using:

.. code:: python

   optimizer.zero_grad()

-  Move training samples and labels to device using the 'tensor.to'
   method.
-  Perform forward/prediction pass using

.. code:: python

   output = model(train_x)

-  The prediction results are compared against the corresponding labels
   using the loss function to compute the loss

.. code:: python

   loss_fn(output, train_label)

-  The loss is propagated back through the model using chain-rule to
   compute the weight gradients

.. code:: python

   loss.backward()

-  The weights are updated with a change that is proportional to the
   computed weights gradients

.. code:: python

   optimizer.step()

At the end of training we compute the throughput, display the final loss
and save the checkpoint.

Expected CPU output:

.. code:: bash

    ----------Training ---------------
    Train throughput (iter/sec): 286.96994718801335
    Final loss is 0.1040
    ----------End Training ---------------

For a full tutorial on training in PyTorch, please see
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html.

Thus far we have used PyTorch without Trainium. Next, we will show how
to change this script to run on Trainium.

Single-worker MLP training on Trainium
--------------------------------------

To run on Trainium, first we modify the CPU training script train_cpu.py to run with
PyTorch Neuron torch_xla as described in :ref:`PyTorch Neuron for Trainium Getting Started Guide <pytorch-neuronx-programming-guide>`
by changing the device:

.. code:: python

   import torch_xla.core.xla_model as xm
   device = xm.xla_device()
   # or
   device = 'xla'

When the model is moved to the XLA device using ``model.to(device)``
method, subsequent operations on the model are recorded for later
execution. This is XLA's lazy execution which is different from
PyTorch's eager execution. Within the training loop, we must mark the
graph to be optimized and run on XLA device (NeuronCore) using
xm.mark_step() (unless MpDeviceLoader is used as you will see in the next section). 
Without this mark, XLA cannot determine where the graph
ends. The collected computational graph also gets compiled and executed
when you request the value of a tensor such as by calling
``loss.item()`` or ``print(loss)``.

To save a checkpoint, it is recommended to use the ``xm.save()``
function instead of ``torch.save()`` to ensure states are moved to CPU.
``xm.save()`` also prevents the "XRT memory handle not found" warning at
the end of evaluation script (if the checkpoint saved using torch.save()
is used for evaluation).

The resulting script ``train.py`` can be executed as 
``python3 train.py``. Again, note that we import the MLP model
from ``model.py``. When you examine the script, the comments that begin with
'XLA' indicate the changes required to make the script compatible with
torch_xla.

Expected output on trn1.32xlarge (start from a fresh compilation cache, located at /var/tmp/neuron-compile-cache by default):

.. code:: bash

    2022-04-12 16:15:00.000947: INFO ||NCC_WRAPPER||: No candidate found under /var/tmp/neuron-compile-cache/USER_neuroncc-1.0.47218.0+162039557/MODULE_18200615679846498221.
    2022-04-12 16:15:00.000949: INFO ||NCC_WRAPPER||: Cache dir for the neff: /var/tmp/neuron-compile-cache/USER_neuroncc-1.0.47218.0+162039557/MODULE_18200615679846498221/MODULE_0_SyncTensorsGraph.318_18200615679846498221_ip-172-31-69-14.ec2.internal-8355221-28940-5dc775cd78aa2/83a0fd4a-b07e-4404-aa55-701ab3b2700c
    ........
    Compiler status PASS
    2022-04-12 16:18:05.000843: INFO ||NCC_WRAPPER||: Exiting with a successfully compiled graph
    2022-04-12 16:18:05.000957: INFO ||NCC_WRAPPER||: No candidate found under /var/tmp/neuron-compile-cache/USER_neuroncc-1.0.47218.0+162039557/MODULE_5000680699473283909.
    2022-04-12 16:18:05.000960: INFO ||NCC_WRAPPER||: Cache dir for the neff: /var/tmp/neuron-compile-cache/USER_neuroncc-1.0.47218.0+162039557/MODULE_5000680699473283909/MODULE_1_SyncTensorsGraph.390_5000680699473283909_ip-172-31-69-14.ec2.internal-8355221-28940-5dc7767e5fc69/7d0a2955-11b4-42e6-b536-6f0f02cc68df
    .
    Compiler status PASS
    2022-04-12 16:18:12.000912: INFO ||NCC_WRAPPER||: Exiting with a successfully compiled graph
    ----------Training ---------------
    Train throughput (iter/sec): 95.06756661972014
    Final loss is 0.1979
    ----------End Training ---------------

If you re-run the training script a second time, you will see messages
indicating that the compiled graphs are cached in the persistent cache
from the previous run and that the startup time is quicker:

.. code:: bash

    (aws_neuron_venv_pytorch_p36) [ec2-user@ip-172-31-69-14 mnist_mlp]$ python train.py |& tee log_trainium
    2022-04-12 16:21:58.000241: INFO ||NCC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/USER_neuroncc-1.0.47218.0+162039557/MODULE_18200615679846498221/MODULE_0_SyncTensorsGraph.318_18200615679846498221_ip-172-31-69-14.ec2.internal-8355221-28940-5dc775cd78aa2/83a0fd4a-b07e-4404-aa55-701ab3b2700c/MODULE_0_SyncTensorsGraph.318_18200615679846498221_ip-172-31-69-14.ec2.internal-8355221-28940-5dc775cd78aa2.neff. Exiting with a successfully compiled graph
    2022-04-12 16:21:58.000342: INFO ||NCC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/USER_neuroncc-1.0.47218.0+162039557/MODULE_5000680699473283909/MODULE_1_SyncTensorsGraph.390_5000680699473283909_ip-172-31-69-14.ec2.internal-8355221-28940-5dc7767e5fc69/7d0a2955-11b4-42e6-b536-6f0f02cc68df/MODULE_1_SyncTensorsGraph.390_5000680699473283909_ip-172-31-69-14.ec2.internal-8355221-28940-5dc7767e5fc69.neff. Exiting with a successfully compiled graph
    ----------Training ---------------
    Train throughput (iter/sec): 93.16748895384832
    Final loss is 0.1979
    ----------End Training ---------------

Multiple graphs can be created during execution since there are
differences between some iterations (first, steady state, last). After
the first iteration, the graph for each iteration should remain the same
from iteration to iteration. This allows XLA runtime to execute a
previous compiled graph that has been cached in XLA runtime cache.

If the inner training loop has some control-flows, for example for
gradient accumulation, the number of compiled graphs may increase due to the
generation and consumption of intermediates as well as additional
operations when the conditional path is taken.

Multi-worker data-parallel MLP training using torchrun
------------------------------------------------------

Data parallel training allows you to replicate your script across
multiple workers, each worker processing a proportional portion of the
dataset, in order to train faster.

The PyTorch distributed utility torchrun can be used to launch multiple
processes in a server node for multi-worker data parallel training.

To run multiple workers in data parallel configuration using torchrun,
modify the single-worker training script train.py as follows (below we use ``xm``
as alias for ``torch_xla.core.xla_model`` and ``xmp`` as alias for
``torch_xla.distributed.xla_multiprocessing``):

1. Import XLA backend for torch.distributed using ``import torch_xla.distributed.xla_backend``.
2. Use ``torch.distributed.init_process_group('xla')``
   to initialize PyTorch XLA runtime and Neuron
   runtime.
3. Use XLA multiprocessing device loader (``MpDeviceLoader``) from
   ``torch_xla.distributed`` to wrap PyTorch data loader.
4. Use ``xm.optimizer_step(optimizer)`` to perform allreduce and take
   optimizer step.

XLA MpDeviceLoader is optimized for XLA and is recommended for best
performance. It also takes care of marking the step for execution
(compile and execute the lazily collected operations for an iteration)
so no separate ``xm.mark_step()`` is needed.

The following are general best-practice changes needed to scale up the
training:

1. Set the random seed to be the same across workers.
2. Scale up the learning rate by the number of workers. Use
   ``xm.xrt_world_size()`` to get the global number of workers.
3. Add distributed sampler to allow different worker to sample different
   portions of dataset.

Also, the ``xm.save()`` function used to save checkpoint automatically
saves only for the rank-0 worker's parameters.

The resulting script is ``train_torchrun.py``
(note again that we import the MLP model from ``model.py``):

Next we use the ``torchrun`` utility that is included with torch
installation to run multiple processes, each using one NeuronCore. Use
the option ``nproc_per_node`` to indicate the number of processes to launch.
For example, to run on two NeuronCores on one Trn1 instance only, do:

``torchrun --nproc_per_node=2 train_torchrun.py``

NOTE: Currently we only support 1 and 2 worker configurations on trn1.2xlarge and 1, 2, 8, and 32-worker configurations on trn1.32xlarge.

Expected output on trn1.32xlarge (second run to avoid compilations):

.. code:: bash

    ----------Training ---------------
    ----------Training ---------------
    ... (Info messages truncated)
    Train throughput (iter/sec): 163.25353269069706
    Train throughput (iter/sec): 163.23261047441036
    Final loss is 0.3469
    Final loss is 0.1129
    ----------End Training ---------------
    ----------End Training ---------------

In another example, we run on two instances, using 32 NeuronCores on each instance.
NOTE: To run on multiple instances, you will need trn1.32xlarge instances
and using all 32 NeuronCores on each instance. 

On the rank-0 Trn1 host (root), run with ``--node_rank=0``  using torchrun utility:

.. code:: shell

    torchrun --nproc_per_node=32 --nnodes=2 --node_rank=0 --master_addr=<root IP> --master_port=2020 train_torchrun.py

On another Trn1 host, run with --node_rank=1 :

.. code:: shell

    torchrun --nproc_per_node=32 --nnodes=2 --node_rank=1 --master_addr=<root IP> --master_port=2020 train_torchrun.py

It is important to launch rank-0 worker with --node_rank=0  to avoid hang.


Known issues and limitations
----------------------------

MLP model is not optimized for performance. For the single-worker training, the performance can be improved by using MpDeviceLoader which exists in the multiprocessing example. For example, by setting ``--nproc_per_node=1`` in the torchrun example, you will see higher MLP performance.

.. code:: bash

    (aws_neuron_venv_pytorch_p36) [ec2-user@ip-172-31-69-14 mnist_mlp]$ torchrun --nproc_per_node=1 train_torchrun.py

    ----------Training ---------------
    ... (Info messages truncated)
    Train throughput (iter/sec): 192.43508922834008
    Final loss is 0.2720
    ----------End Training ---------------
