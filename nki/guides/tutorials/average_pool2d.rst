.. _nki-averagepool2d:

AveragePool2D
=============

In this tutorial, we examine a case of
dimensionality reduction. We implement a 2D AveragePool operation, which
is used in many vision neural networks.
In doing so, we learn about:

-  NKI syntax and programming model.
-  multi-dimensional memory access patterns in NKI.

The 2D AveragePool operation takes
``C x [H,W]`` matrices and reduces each matrix along the ``H`` and ``W``
axes. To leverage free-dimension flexible indexing, we can map the ``C``
(parallel) axis to the ``P`` dimension and ``H/W`` (contraction)
axes to the ``F`` dimension.
Performing such a 2D pooling operation requires a 4D memory access
pattern in the ``F`` dimension, with reduction along two axes.
:ref:`Figure <nki-fig-avgpool>`
below illustrates the input and output tensor layouts.

.. :

.. figure:: ../../img/pm-index-3.png
   :name: nki-fig-avgpool
   :align: center
   :width: 60%

   2D-Pooling Operation (reducing on axes F2 and F4)

PyTorch
-------

Compute kernel
^^^^^^^^^^^^^^

.. nki_example:: ../../examples/average_pool2d/average_pool2d_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_37


Launching kernel and testing correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute the kernel, we prepare tensors ``in_tensor`` and call ``tensor_avgpool_kernel``:

.. nki_example:: ../../examples/average_pool2d/average_pool2d_torch.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_38

JAX
-------

Compute kernel
^^^^^^^^^^^^^^

Let's reuse the same NKI kernel implementation defined for PyTorch above:

.. nki_example:: ../../examples/average_pool2d/average_pool2d_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_37

In order to pass ``pool_size`` as a compile time constant, we pass ``pool_size`` as kwargs.

.. nki_example:: ../../examples/average_pool2d/average_pool2d_jax.py
   :language: python
   :marker: NKI_EXAMPLE_39

We write a reference JAX implementation of ``AveragePool2D`` as JAX does
not have a primitive for it.

.. nki_example:: ../../examples/average_pool2d/average_pool2d_jax.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_40


Launching kernel and testing correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute the kernel, we prepare array ``in_array`` and invoke the kernel caller function ``tensor_avgpool_kernel``:

.. nki_example:: ../../examples/average_pool2d/average_pool2d_jax.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_41


Download All Source Code
--------------------------

Click the links to download source code of the kernels and the testing code
discussed in this tutorial.

* NKI baremetal implementation: :download:`average_pool2d_nki_kernels.py <../../examples/average_pool2d/average_pool2d_nki_kernels.py>`
* PyTorch implementation: :download:`average_pool2d_torch.py <../../examples/average_pool2d/average_pool2d_torch.py>`
    * You must also download :download:`average_pool2d_nki_kernels.py <../../examples/average_pool2d/average_pool2d_nki_kernels.py>`
      into the same folder to run this PyTorch script.
* JAX implementation: :download:`average_pool2d_jax.py <../../examples/average_pool2d/average_pool2d_jax.py>`
    * You must also download :download:`average_pool2d_nki_kernels.py <../../examples/average_pool2d/average_pool2d_nki_kernels.py>`
      into the same folder to run this JAX script.

You can also view the source code in the GitHub repository `nki_samples <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/tutorials/average_pool2d/>`_

Example usage of the scripts:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run NKI baremetal implementation:

.. code-block::

   python3 average_pool2d_nki_kernels.py

Run PyTorch implementation:

.. code-block::

   python3 average_pool2d_torch.py

Run JAX implementation:

.. code-block::

   python3 average_pool2d_jax.py
