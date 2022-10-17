.. _pytorch-neuronx-programming-guide:

Developer Guide for Training with PyTorch Neuron  (``torch-neuronx``)
=====================================================================


.. contents:: Table of Contents
   :local:
   :depth: 2


Trainium is designed to speed up model training and reduce training cost. It is available on the Trn1 instances. Each Trainium accelerator has two NeuronCores, which are the main neural network compute units.

PyTorch Neuron enables PyTorch users to train their models on Trainium's
NeuronCores with little code change to their training code. It is based
on the `PyTorch/XLA software package <https://pytorch.org/xla>`__.

This guide helps you get started with single-worker training and
distributed training using PyTorch Neuron.

PyTorch Neuron
--------------

Neuron XLA device
~~~~~~~~~~~~~~~~~

With PyTorch Neuron the default XLA device is mapped to a NeuronCore. By default, one NeuronCore is configured. To use Neuron XLA device, specify
the device as ``xm.xla_device()`` or ``'xla'``:

.. code:: python

   import torch_xla.core.xla_model as xm
   device = xm.xla_device()

or

.. code:: python

   device = 'xla'

PyTorch models and tensors can be mapped to the device as usual:

.. code:: python

   model.to(device)
   tensor.to(device)

To move tensor back to CPU, do :

.. code:: python

   tensor.cpu()

or

.. code:: python

   tensor.to('cpu')

PyTorch Neuron single-worker training quick-start
--------------------------------------------------------------

PyTorch Neuron uses XLA to enable conversion of
PyTorch operations to Trainium instructions. To get started on PyTorch
Neuron, first modify your :ref:`training script <neuronx-mlp-training-tutorial>` to
use XLA in the same manner as described in `PyTorch/XLA
documentation <https://pytorch.org/xla>`__ and
use XLA device:

.. code:: python

   import torch_xla.core.xla_model as xm

   device = xm.xla_device()
   # or
   device = 'xla'

The NeuronCore is mapped to an XLA device. On Trainium instance, the XLA device is automatically mapped to the first available NeuronCore.

By default the above steps will enable the training script to run on one
NeuronCore. NOTE: Each process is mapped to one NeuronCore.

Finally, add ``mark_step`` at the end of the training step to compile
and execute the training step:

.. code:: python

   xm.mark_step()

These changes can be placed in control-flows in order to keep the script
the same between PyTorch Neuron and CPU/GPU. For example, you can use an
environment variable to disable XLA which would cause the script to run
in PyTorch native mode (using CPU on Trainium instances and GPU on GPU
instances):

.. code:: python

   device = 'cpu'
   if not os.environ.get("DISABLE_XLA", None):
       device = 'xla'

   ...

       # end of training step 
       if not os.environ.get("DISABLE_XLA", None):
           xm.mark_step()

More on the need for mark_step is at `Understand the lazy mode in
PyTorch Neuron <#understand-the-lazy-mode-in-pytorch-neuron>`__.

For a full runnable example, please see the :ref:`Single-worker MLP training
on Trainium tutorial
<neuronx-mlp-training-tutorial:single-worker-mlp-training-on-trainium>`.

PyTorch Neuron multi-worker data parallel training using torchrun
-----------------------------------------------------------------

Data parallel training allows you to replicate your script across
multiple workers, each worker processing a proportional portion of the
dataset, in order to train faster.

To run multiple workers in data parallel configuration, with each worker
using one NeuronCore, first add additional imports for parallel
dataloader and multi-processing utilities:

::

   import torch_xla.distributed.parallel_loader as pl

Next we initialize the Neuron distributed context using the XLA backend for torch.distributed:

::

    import torch_xla.distributed.xla_backend
    torch.distributed.init_process_group('xla')

Next, replace ``optimizer.step()`` function call with
``xm.optimizer_step(optimizer)`` which adds gradient synchronization
across workers before taking the optimizer step:

::

   xm.optimizer_step(optimizer)

If you're using a distributed dataloader, wrap your dataloader in the
PyTorch/XLA's ``MpDeviceLoader`` class which provides buffering
to hide CPU to device data load latency:

::

   parallel_loader = pl.MpDeviceLoader(dataloader, device)

Within the training code, use xm.xrt_world_size() to get the world size,
and xm.get_ordinal to get the global rank of the current process.

Then run use `PyTorch
torchrun <https://pytorch.org/docs/stable/elastic/run.html#launcher-api>`__
utility to run the script. For example, to run 32 worker data parallel
training:

``torchrun --nproc_per_node=32 <script and options>``

To run on multiple instances, make sure to use trn1.32xlarge instances
and use all 32 NeuronCores on each instance. For example, with two instances, 
on the rank-0 Trn1 host, run with --node_rank=0  using torchrun utility:

.. code:: shell

    torchrun --nproc_per_node=32 --nnodes=2 --node_rank=0 --master_addr=<root IP> --master_port=<root port> <script and options>

On another Trn1 host, run with --node_rank=1 :

.. code:: shell

    torchrun --nproc_per_node=32 --nnodes=2 --node_rank=1 --master_addr=<root IP> --master_port=<root port> <script and options>

It is important to launch rank-0 worker with --node_rank=0  to avoid hang.

More information about torchrun can be found PyTorch documentation at
https://pytorch.org/docs/stable/elastic/run.html#launcher-api .

See the :ref:`Multi-worker data-parallel MLP training using torchrun
tutorial <neuronx-mlp-training-tutorial:multi-worker-data-parallel-mlp-training-using-torchrun>`
for a full example.

Conversion from Distributed Data Parallel (DDP) application
-----------------------------------------------------------

Distributed Data Parallel (DDP) in torch.distributed module is a wrapper
to help convert a single-worker training to distributed training. To
convert from torch.distributed Distributed Data Parallel (DDP)
application to PyTorch Neuron, first convert the application back to
single-worker training, which simply involves removing the DDP wrapper,
for example ``model = DDP(model, device_ids=[rank])``. After this,
follow the previous section to change to multi-worker training.

PyTorch Neuron environment variables
--------------------------------------

Environment variables allow modifications to PyTorch Neuron behavior
without requiring code change to user script. See :ref:`PyTorch Neuron environment variables <pytorch-neuronx-envvars>` for more details.

Persistent cache for compiled graphs
------------------------------------

PyTorch/XLA has an internal in-memory compilation cache that caches previously
compiled graphs within the same python process. However this internal
cache is not persistent between runs. PyTorch Neuron includes a
persistent cache that enables caching of previously compiled graph on disk so
that subsequent run of the same program do not incur long compilation
time. This cache is enabled by default and the default cache directory
is /var/tmp/neuron-compile-cache.

The cache uses hash of the Neuron compiler flags and XLA graph as the
key. If the Neuron compiler version or XLA graph changes, you will see
recompilation. Examples of changes that would cause XLA graph change
include:

-  Model type and size
-  Batch size
-  Optimizer and optimizer hyperparameters
-  Location of xm.mark_step()

To disable the cache, you can pass ``--no_cache`` option via NEURON_CC_FLAGS:

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --no_cache'

To change the cache's root directory, pass ``cache_dir=<root dir>``
option via NEURON_CC_FLAGS (the actual cache directory will be in
``<root dir>/neuron-compile-cache``:

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --cache_dir=<root dir>'

Stale cached compiled graphs (NEFFs) are deleted from the cache whenever
the size of cache is above default cache size of 100GB . The deletion
order is based on least-recently-used first. To change the cache size,
pass ``--cache_size=SIZE_IN_BYTES``. For example, to change the cache
size to 16 MB:

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --cache_size=16777216'

A cache entry considered stale if the last used time is older than a
time-to-live value, currently default to 30 days. If the last used time
is earlier than the time-to-live value, then it is not deleted even if
cache size exceeds cache size limit. To change cache time-to-live, set
the option ``--cache_ttl`` to the number of days desired:

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --cache_ttl=60'

If in some cases, the compilation failed because of an environment issue, and you want to retry compilation,
you can do so by adding ``--retry_failed_compilation``. This will retry the compilation even if there is a
failed NEFF in the cache.

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --retry_failed_compilation'

You can change the verbose level of the compiler by adding ``log_level`` to either ``WARNING``, ``INFO``
or ``ERROR``. This can be done as follows:

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --log_level=INFO'

Number of graphs
-----------------

PyTorch/XLA converts PyTorch's eager mode execution to lazy-mode
graph-based execution. During this process, there can be multiple graphs
compiled and executed if there are extra mark-steps or functions with
implicit mark-steps. Additionally, more graphs can be generated if there
are different execution paths taken due to control-flows.

Cast all XLA float tensors to BFloat16
--------------------------------------

With PyTorch Neuron, the default behavior is for torch.float (FP32) and torch.double (FP64) tensors
to be mapped to torch.float in hardware. To reduce memory footprint and improve performance,
torch.float and torch.double tensors can automatically be converted to BFloat16 by setting
the environment variable ``XLA_USE_BF16=1``.

Automatic Mixed-Precision
-------------------------

BF16 mixed-precision using PyTorch Autocast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the compiler automatically cast internal FP32 operations to
BF16. You can disable this and allow PyTorch's BF16 mixed-precision to
do the casting. PyTorch's BF16 mixed-precision is achieved by casting
certain operations to operate BF16. We currently use CUDA's list of
operations that can operate in BF16:

(NOTE: convolution is currently supported.)

.. code:: bash

   _convolution
   _convolution
   _convolution_nogroup
   conv1d
   conv2d
   conv3d
   conv_tbc
   conv_transpose1d
   conv_transpose2d
   conv_transpose3d
   convolution
   cudnn_convolution
   cudnn_convolution_transpose
   cudnn_convolution
   cudnn_convolution_transpose
   cudnn_convolution
   cudnn_convolution_transpose
   prelu
   addmm
   addmv
   addr
   matmul
   mm
   mv
   linear
   addbmm
   baddbmm
   bmm
   chain_matmul
   linalg_multi_dot

To enable PyTorch's BF16 mixed-precision, first turn off the Neuron
compiler auto-cast:

.. code:: python

   os.environ["NEURON_CC_FLAGS"] = "--fast-math=none"

Next, overwrite torch.cuda.is_bf16_supported to return True:

.. code:: python

   torch.cuda.is_bf16_supported = lambda: True

Next, per recommendation from official PyTorch documentation, place only
the forward-pass of the training step in the torch.autocast scope:

.. code:: python

   with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
       # forward pass

The device type is CUDA because we are using CUDA's list of BF16
compatible operations as mentioned above.

Example showing the original training code snippet:

.. code:: python

   def train_loop_fn(train_loader):
       for i, data in enumerate(train_loader):
           inputs = data[0]
           labels = data[3]
           outputs = model(inputs, labels=labels)
           loss = outputs.loss/ flags.grad_acc_steps
           loss.backward()
           optimizer.step()
           xm.mark_step()               

The following shows the training loop modified to use BF16 autocast:

.. code:: python

   os.environ["NEURON_CC_FLAGS"] = "--fast-math=none"

   def train_loop_fn(train_loader):
       for i, data in enumerate(train_loader):
           torch.cuda.is_bf16_supported = lambda: True
           with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
               inputs = data[0]
               labels = data[3]
               outputs = model(inputs, labels=labels)
           loss = outputs.loss/ flags.grad_acc_steps
           loss.backward()
           optimizer.step()
           xm.mark_step()        

For a full example of BF16 mixed-precision, see :ref:`PyTorch Neuron BERT Pretraining Tutorial <hf-bert-pretraining-tutorial>`.

See official PyTorch documentation for more details about
`torch.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`__
.

Tips and Best Practices
-----------------------

Understand the lazy mode in PyTorch Neuron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One significant difference between PyTorch Neuron and native PyTorch is
that the PyTorch Neuron system runs in lazy mode while the native
PyTorch runs in eager mode. Tensors in lazy mode are placeholders for
building the computational graph until they are materialized after the
compilation and evaluation are complete. The PyTorch Neuron system
builds the computational graph on the fly when you call PyTorch APIs to
build the computation using tensors and operators. The computational
graph gets compiled and executed when ``xm.mark_step()`` is called
explicitly or implicitly by ``pl.MpDeviceLoader/pl.ParallelLoader``, or
when you explicitly request the value of a tensor such as by calling
``loss.item()`` or ``print(loss)``.

.. _minimize-the-number-of-compilation-and-executions:

Minimize the number of compilation-and-executions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For best performance, you should keep in mind the possible ways to
initiate compilation-and-executions as described in `Understand the lazy
mode in PyTorch/XLA <#understand-the-lazy-mode-in-pytorch-neuron>`__ and
should try to minimize the number of compilation-and-executions.
Ideally, only one compilation-and-execution is necessary per training
iteration and is initiated automatically by
``pl.MpDeviceLoader/pl.ParallelLoader``. The ``MpDeviceLoader`` is
optimized for XLA and should always be used if possible for best
performance. During training, you might want to examine some
intermediate results such as loss values. In such case, the printing of
lazy tensors should be wrapped using ``xm.add_step_closure()`` to avoid
unnecessary compilation-and-executions.

Ensure common initial weights across workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To achieve best accuracy during data parallel training, all workers need
to have the same initial parameter states. This can be achieved by using
the same seed across the workers. In the case of HuggingFace library,
the set_seed function can be used.
(https://github.com/pytorch/xla/issues/3216).

Use PyTorch/XLA's model save function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid problems with saving and loading checkpoints, make sure you use
PyTorch/XLA's model save function to properly checkpoint your model. For
more information about the function, see
`torch_xla.core.xla_model.save <https://pytorch.org/xla/release/1.9/index.html#torch_xla.core.xla_model.save>`__
in the *PyTorch on XLA Devices* documentation.

FAQ
---

What is the difference between Trainium and Inferentia?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trainium is an accelerator designed to speed up training, whereas
Inferentia is an accelerator designed to speed up inference.

Debugging and troubleshooting
-----------------------------

To debug on PyTorch Neuron, please follow the :ref:`debug
guide <./pytorch-neuron-debug.html>`.
