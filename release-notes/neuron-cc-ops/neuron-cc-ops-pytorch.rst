.. _neuron-cc-ops-pytorch:

Supported operators [PyTorch]
=============================

Current operator lists may be generated with these commands inside
python:

.. code:: python

   import torch.neuron
   print(*torch.neuron.get_supported_operations(), sep='\n')


.. _pytorch-neuron-release-15210:

PyTorch Neuron Release [1.5.21.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No change


.. _pytorch-neuron-release-1570:

PyTorch Neuron Release [1.5.7.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::

    aten::erf
    prim::DictConstruct


.. _pytorch-neuron-release-1410:

PyTorch Neuron Release [1.4.1.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No change

.. _pytorch-neuron-release-1350:

PyTorch Neuron Release [1.3.5.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::
   
   aten::numel
   aten::ones_like
   aten::reciprocal
   aten::topk

.. _pytorch-neuron-release-12160:

PyTorch Neuron Release [1.2.16.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No change

.. _pytorch-neuron-release-12150:

PyTorch Neuron Release [1.2.15.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No change

.. _pytorch-neuron-release-1230:

PyTorch Neuron Release [1.2.3.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::

   aten::silu
   aten::zeros_like

.. _pytorch-neuron-release-1170:

PyTorch Neuron Release [1.1.7.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::

   aten::_shape_as_tensor
   aten::chunk
   aten::empty
   aten::masked_fill

.. _pytorch-neuron-release-10240450:

PyTorch Neuron Release [1.0.24045.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::

    aten::__and__
    aten::bmm
    aten::clone
    aten::expand_as
    aten::fill_
    aten::floor_divide
    aten::full
    aten::hardtanh
    aten::hardtanh_
    aten::le
    aten::leaky_relu
    aten::lt
    aten::mean
    aten::ne
    aten::softplus
    aten::unbind
    aten::upsample_bilinear2d


.. _pytorch-neuron-release-10172000:

PyTorch Neuron Release [1.0.1720.00]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::

   aten::constant_pad_nd
   aten::meshgrid

.. _pytorch-neuron-release-1015320:

PyTorch Neuron Release [1.0.1532.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::

   aten::ones

.. _pytorch-neuron-release-1015220:

PyTorch Neuron Release [1.0.1522.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  No change

.. _pytorch-neuron-release-1013860:

PyTorch Neuron Release [1.0.1386.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added the following instructions. Please note, primitives are included
in this list from this release.

::

   aten::ceil
   aten::clamp
   aten::eq
   aten::exp
   aten::expand_as
   aten::flip
   aten::full_like
   aten::ge
   aten::gt
   aten::log2
   aten::log_softmax
   aten::max
   aten::neg
   aten::relu
   aten::rsqrt
   aten::scalarImplicit
   aten::sqrt
   aten::squeeze
   aten::stack
   aten::sub
   aten::sum
   aten::true_divide
   aten::upsample_nearest2d
   prim::Constant
   prim::GetAttr
   prim::ImplicitTensorToNum
   prim::ListConstruct
   prim::ListUnpack
   prim::NumToTensor
   prim::TupleConstruct
   prim::TupleUnpack

.. _pytorch-neuron-release-1011680:

PyTorch Neuron Release [1.0.1168.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added ``aten::ScalarImplicit``

.. _pytorch-neuron-release-1010010:

PyTorch Neuron Release [1.0.1001.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added

::

   aten::detach
   aten::floor
   aten::gelu
   aten::pow
   aten::sigmoid
   aten::split

Removed ( Reasons given alongside )

::

   aten::embedding (does not meet performance criteria)
   aten::erf (error function does not meet accuracy criteria)
   aten::tf_dtype_from_torch (internal support function, not an operator)

.. _pytorch-neuron-release-108250:

PyTorch Neuron Release [1.0.825.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _pytorch-neuron-release-107630:

PyTorch Neuron Release [1.0.763.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Please note. Starting with this release we will not publish
   primitives (prim::).

-  Previous release inaccurately listed these operators as aten ops,
   they are not.

::

   aten::tf_broadcastable_slice
   aten::tf_padding

The following new operators are added in this release.

::

   aten::Int
   aten::arange
   aten::contiguous
   aten::div
   aten::embedding
   aten::erf
   aten::expand
   aten::eye
   aten::index_select
   aten::layer_norm
   aten::matmul
   aten::mm
   aten::permute
   aten::reshape
   aten::rsub
   aten::select
   aten::size
   aten::slice
   aten::softmax
   aten::tf_dtype_from_torch
   aten::to
   aten::transpose
   aten::unsqueeze
   aten::view
   aten::zeros

These operators were already supported previously (removing the two that
were included by mistake)

::

   aten::_convolution
   aten::adaptive_avg_pool2d
   aten::add
   aten::add_
   aten::addmm
   aten::avg_pool2d
   aten::batch_norm
   aten::cat
   aten::dimension_value
   aten::dropout
   aten::flatten
   aten::max_pool2d
   aten::mul
   aten::relu_
   aten::t
   aten::tanh
   aten::values
   prim::Constant
   prim::GetAttr
   prim::ListConstruct
   prim::ListUnpack
   prim::TupleConstruct
   prim::TupleUnpack

.. _pytorch-neuron-release-106720:

PyTorch Neuron Release [1.0.672.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No change

.. _pytorch-neuron-release-105520:

PyTorch Neuron Release [1.0.552.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   aten::_convolution
   aten::adaptive_avg_pool2d
   aten::add
   aten::add_
   aten::addmm
   aten::avg_pool2d
   aten::batch_norm
   aten::cat
   aten::dimension_value
   aten::dropout
   aten::flatten
   aten::max_pool2d
   aten::mul
   aten::relu_
   aten::t
   aten::tanh
   aten::tf_broadcastable_slice
   aten::tf_padding
   aten::values
   prim::Constant
   prim::GetAttr
   prim::ListConstruct
   prim::ListUnpack
   prim::TupleConstruct
   prim::TupleUnpack
