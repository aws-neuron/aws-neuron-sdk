.. _nki_getting_started:

Getting Started with NKI
--------------------------

In this guide, we will implement a simple "Hello World" style NKI kernel and run it on a NeuronDevice
(Trainium/Inferentia2 or beyond device).
We will showcase how to invoke a NKI kernel standalone through NKI baremetal mode
and also through ML frameworks (PyTorch and JAX).
Before diving into kernel implementation, let's make sure you have the correct environment setup
for running NKI kernels.


Environment Setup
~~~~~~~~~~~~~~~~~~~~

You need a `Trn1 <https://aws.amazon.com/ec2/instance-types/trn1/>`__ or
`Inf2 <https://aws.amazon.com/ec2/instance-types/inf2/>`__ instance set up
on AWS to run NKI kernels on a NeuronDevice.
Once logged into the instance, follow steps below to ensure you have all the
required packages installed in your Python environment.

NKI is shipped as part of the Neuron compiler package. To make sure you have the latest compiler
package, see `Setup
Guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/index.html>`__
for an installation guide.

You can verify that NKI is available in your compiler installation by
running the following command:

::

   python -c 'import neuronxcc.nki'

This attempts to import the NKI package. It will error out if NKI is not included in
your Neuron compiler version or if the Neuron
compiler is not installed. The import might take about a minute the first
time you run it. Whenever possible, we recommend using local instance NVMe volumes instead of EBS for
executable code.

If you intend to run NKI kernels without any ML framework for quick prototyping, you must have 
`NumPy <https://numpy.org/install/>`__ installed.

To call NKI kernels from PyTorch, you must have ``torch_neuronx``
installed. For details on installation, see
`PyTorch Neuron Setup <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx>`__.

Verify that you have ``torch_neuronx`` installed by
running the following command:

::

   python -c 'import torch_neuronx'

To call NKI kernels from JAX, you must have both ``libneuronxla`` and ``jax`` installed.  For details on installing JAX, see
`JAX Neuron Setup <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/jax/setup/jax-setup.html>`_.

Verify that you have ``libneuronxla`` installed by running the following command:

::
	  		  
	python -c 'import libneuronxla'

Implementing your first NKI kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In current NKI release, all input tensors must be passed into the kernel as device memory (HBM) tensors
on a NeuronDevice. Similarly, output tensors returned from the kernel must also reside in device memory.
The body of the kernel typically consists of three main phases:

1. Load the inputs from device memory to on-chip memory (SBUF).
2. Perform the desired computation.
3. Store the outputs from on-chip memory to device memory.

For more details on the above terms, see
:doc:`NKI Programming Model <programming_model>`.

Below is a small NKI kernel example. In this example, we take two tensors and add them element-wise to
produce an output tensor of the same shape.

.. We start by creating a ``nki_tensor_add.py`` file with
.. the following code. We'll discuss different parts of
.. the code in details later in this section.

.. *nki_tensor_add.py:*

.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_0


Now let us walk through the above code:

.. _importing-nki:

Importing NKI
^^^^^^^^^^^^^^^^^^^^

We start by importing :doc:`neuronxcc.nki<api/nki>` which includes function decorators to compile
NKI kernels and also :doc:`neuronxcc.nki.language <api/nki.language>` which implements the
NKI language. We will go into more detail regarding the NKI language
in :doc:`NKI Programming Model <programming_model>`,
but for now you can think of it as a tile-level
domain-specific language.


.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :marker: NKI_EXAMPLE_1

.. _defining-a-kernel:

Defining a kernel
^^^^^^^^^^^^^^^^^^^^^^^^

Next we define the ``nki_tensor_add_kernel`` Python function, which contains the NKI kernel code.
The kernel is `decorated <https://www.geeksforgeeks.org/decorators-in-python/>`__
with :doc:`nki.jit <api/generated/nki.jit>`, which allows Neuron compiler to recognize this
is NKI kernel code and trace it correctly.
Input tensors (``a_input`` and ``b_input``) are passed by reference into the
kernel, just like any other Python function input parameters.

.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :marker: NKI_EXAMPLE_2

.. This instructs the NKI toolset to translate this Python function into an
.. intermediate representation, which is then passed to the Neuron
.. compiler.

.. .. _tensor-indexing:

.. Indexing tensors
.. ^^^^^^^^^^^^^^^^^^^^^^

.. NKI performs tile-level operations, where a tile can be created using a
.. combination of a tensor and corresponding indices. So, we first define
.. the desired indices:

.. ::

..      i_x = nl.arange(4)[:, None]
..      i_y = nl.arange(3)[None, :]

.. The above code defines the indices ``i_x`` and ``i_y``, of shapes
.. ``(4,1)`` and ``(1,3)`` respectively. When ``i_x`` and ``i_y`` are used
.. together to access a tensor, they form a ``(4,3)`` 2D index-grid,
.. following NumPy's *broadcasting* conventions to handle
.. arrays with different shapes during arithmetic operations. For further details on this, see
.. `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ in the NumPy User Guide
.. for further details.

.. In this example, ``arange(4)``
.. creates a 1D array with four evenly spaced values from 0 to 3
.. (``[0,1,2,3]``), and ``[:, None]`` adds an additional axis to explicitly
.. make ``i_x`` a *row-index* in a 2D array, with shape of ``(4, 1)``.
.. Similarly, ``i_y`` is the *column-index* in a 2D array, with shape
.. ``(1,3)``. Since ``i_x`` and ``i_y`` have their values on different
.. axes, putting them together forms a 2D array with 4x3=12 values.
.. :numref:`Fig. %s <nki-fig-md-tensor-indexing>` below visualizes
.. the tensor indices.

.. .. _nki-fig-md-tensor-indexing:

.. .. figure:: img/getting-started-indexing.png
..    :align: center

..    Multi-dimensional tensor indexing in NKI

Checking input shapes
^^^^^^^^^^^^^^^^^^^^^^^
To keep this getting started guide simple, this kernel example expects all input and output tensors
have the same shapes for an element-wise
addition operation. We further restrict the first dimension of the input/output tensors to not exceed
``nl.tile_size.pmax == 128``. More detailed discussion on tile size limitation is available
in :ref:`NKI Programming Model <nki-tile-size>`. Note, all of these restrictions *can* be lifted with
tensor broadcasting/reshape and tensor tiling with loops in NKI. For more kernel examples, check out
:doc:`NKI tutorials <tutorials>`.


.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :marker: NKI_EXAMPLE_3


.. _loading-inputs:

Loading inputs
^^^^^^^^^^^^^^^^^^^^^

Most NKI kernels start by loading inputs from device memory to on-chip
memory. We need to do that because
computation can only be performed on data in the on-chip memory.

.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :marker: NKI_EXAMPLE_4


.. _5-defining-the-desired-computation:

Defining the desired computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the two input tiles, it is time to define the desired
computation. In this case, we perform a simple element-wise addition
between two tiles:


.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :marker: NKI_EXAMPLE_5


Note that ``c_tile = a_tile + b_tile`` will also work, as NKI overloads
simple Python operators such as ``+``, ``-``, ``*``, and ``/``. For a
complete set of available NKI APIs, refer to
:doc:`NKI API Reference Manual <api/index>`.

.. _storing-outputs:

Storing and returning outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To return the output tensor of the kernel, we first declare a NKI tensor ``c_output`` in
device memory (HBM) and then store the output tile ``c_tile`` from on-chip memory to
``c_output`` using :doc:`nl.store <api/generated/nki.language.store>`.
We end the kernel execution by returning ``c_output`` using a standard Python return call.
This will allow the host to access the output tensor.

.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :marker: NKI_EXAMPLE_6

.. _running-the-kernel:

Running the kernel
~~~~~~~~~~~~~~~~~~~~~~~

Next, we will cover three unique ways to run the above NKI kernel on a NeuronDevice:

1. NKI baremetal: run NKI kernel with no ML framework involvement
2. PyTorch: run NKI kernel as a PyTorch operator
3. JAX: run NKI kernel as a JAX operator

All three run modes can call the same kernel function decorated with the
``nki.jit`` decorator as discussed above:

.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_2

The ``nki.jit`` decorator automatically chooses the correct run mode by checking the incoming
tensor type:

1. NumPy arrays as input: run in NKI baremetal mode
2. PyTorch tensors as input: run in PyTorch mode
3. JAX tensors: run in JAX mode

See :doc:`nki.jit <api/generated/nki.jit>` API doc for more details.

.. note::
   NKI baremetal mode is the most convenient way to prototype and optimize performance
   a NKI kernel alone. For production ML workloads, we highly recommend invoking NKI kernels
   through a ML framework (PyTorch or JAX). This allows you to integrate NKI kernels
   in your regular compute graph to accelerate certain operators
   (see :doc:`NKI Kernel as a Framework Custom Operator <framework_custom_op>` for details)
   and leverage the more optimized host-to-device data transfer
   handling available in ML frameworks.

NKI baremetal
^^^^^^^^^^^^^^^^^^

Baremetal mode expects input tensors of the NKI kernel to be **NumPy arrays**. The kernel also
converts its NKI output tensors to **NumPy arrays**. To invoke the kernel, we first initialize the
two input tensors ``a`` and ``b`` as NumPy arrays. Finally, we call the NKI kernel just like any
other Python function:

.. nki_example:: examples/getting_started_baremetal.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_8

.. note::
   Alternatively, we can decorate the kernel with :doc:`nki.baremetal <api/generated/nki.baremetal>` or pass
   the ``mode`` parameter to the ``nki.jit`` decorator, ``@nki.jit(mode='baremetal')``, to bypass
   the dynamic mode detection. See
   :doc:`nki.baremetal <api/generated/nki.baremetal>` API doc for more available input arguments for the
   baremetal mode.

PyTorch
^^^^^^^^^

To run the above ``nki_tensor_add_kernel`` kernel using PyTorch, we initialize
the input and output tensors as PyTorch ``device`` tensors instead.

.. nki_example:: examples/getting_started_torch.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_10

Running the above code for the first time will trigger compilation of the NKI kernel, which might
take a few minutes before printing any output. The printed output should be as follows:

::

   tensor([[2., 2., 2.],
           [2., 2., 2.],
           [2., 2., 2.],
           [2., 2., 2.]], device='xla:1', dtype=torch.float16)

.. note::
   Alternatively, we can pass the ``mode='torchxla'`` parameter into the ``nki.jit`` decorator to
   bypass the dynamic mode detection.

JAX
^^^^^^^^^
To run the above ``nki_tensor_add_kernel`` kernel using JAX, we initialize the input tensors
as JAX tensors:

.. nki_example:: examples/getting_started_jax.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_11

.. note::
   Alternatively, we can pass the ``mode='jax'`` parameter into the ``nki.jit`` decorator to
   bypass the dynamic mode detection.

Download links
~~~~~~~~~~~~~~~~~

- NKI baremetal script: :download:`getting_started_baremetal.py <examples/getting_started_baremetal.py>`
- PyTorch script: :download:`getting_started_torch.py <examples/getting_started_torch.py>`
- JAX script: :download:`getting_started_jax.py <examples/getting_started_jax.py>`