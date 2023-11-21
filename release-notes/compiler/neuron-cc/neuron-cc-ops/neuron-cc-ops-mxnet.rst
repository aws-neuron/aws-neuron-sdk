.. _neuron-cc-ops-mxnet:


Neuron Apache MXNet Supported operators
====================================================

To see a list of supported operators for MXNet, run the following command:

``neuron-cc list-operators --framework MXNET``

.. _neuron-compiler-release-1600:

Neuron Compiler Release [1.6.13.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added

::

  amp_cast
  amp_multicast

.. _neuron-compiler-release-1410:

Neuron Compiler Release [1.4.1.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1400:

Neuron Compiler Release [1.4.0.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1300:

Neuron Compiler Release [1.3.0.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1270:

Neuron Compiler Release [1.2.7.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1220:

Neuron Compiler Release [1.2.2.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1200:

Neuron Compiler Release [1.2.0.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added

::

 Deconvolution
 LayerNorm
 Pad
 SwapAxis
 _contrib_arange_like
 _contrib_interleaved_matmul_encdec_qk
 _contrib_interleaved_matmul_encdec_valatt
 _contrib_interleaved_matmul_selfatt_qk
 _contrib_interleaved_matmul_selfatt_valatt
 arctan
 broadcast_like
 cos
 erf
 pad
 sin
 slice_axis


.. _neuron-compiler-release-10240450:

Neuron Compiler Release [1.0.24045.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added ``_contrib_div_sqrt_dim``, ``broadcast_axis``

.. _neuron-compiler-release-10180010:

Neuron Compiler Release [1.0.18001.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-10179370:

Neuron Compiler Release [1.0.17937.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-10168610:

Neuron Compiler Release [1.0.16861.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Removed ``log`` (Was erroneously reported as added in previous release.
)

.. _neuron-compiler-release-1015275:

Neuron Compiler Release [1.0.15275]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added ``log``

.. _neuron-compiler-release-1012696:

Neuron Compiler Release [1.0.12696]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-109410:

Neuron Compiler Release [1.0.9410]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-107878:

Neuron Compiler Release [1.0.7878]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-106801:

Neuron Compiler Release [1.0.6801]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-105939:

Neuron Compiler Release [1.0.5939]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

no changes

.. _neuron-compiler-release-105301:

Neuron Compiler Release [1.0.5301]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

no changes

.. _neuron-compiler-release-1046800:

Neuron Compiler Release [1.0.4680.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   Activation
   BatchNorm
   Cast
   Concat
   Convolution
   Convolution_v1
   Dropout
   Flatten
   FullyConnected
   LeakyReLU
   Pooling
   Pooling_v1
   RNN
   Reshape
   SequenceMask
   SliceChannel
   Softmax
   UpSampling
   __add_scalar__
   __div_scalar__
   __mul_scalar__
   __pow_scalar__
   __rdiv_scalar__
   __rpow_scalar__
   __rsub_scalar__
   __sub_scalar__
   _arange
   _copy
   _div_scalar
   _equal_scalar
   _full
   _greater_equal_scalar
   _greater_scalar
   _lesser_equal_scalar
   _lesser_scalar
   _maximum
   _maximum_scalar
   _minimum
   _minimum_scalar
   _minus_scalar
   _mul_scalar
   _not_equal_scalar
   _ones
   _plus_scalar
   _power_scalar
   _rdiv_scalar
   _rminus_scalar
   _rnn_param_concat
   _zeros
   batch_dot
   broadcast_add
   broadcast_div
   broadcast_equal
   broadcast_greater
   broadcast_greater_equal
   broadcast_lesser
   broadcast_lesser_equal
   broadcast_maximum
   broadcast_minimum
   broadcast_mod
   broadcast_mul
   broadcast_not_equal
   broadcast_sub
   ceil
   clip
   concat
   elemwise_add
   elemwise_div
   elemwise_mul
   elemwise_sub
   exp
   expand_dims
   flatten
   floor
   gather_nd
   log
   log_softmax
   max
   mean
   min
   negative
   ones_like
   relu
   repeat
   reshape
   reshape_like
   reverse
   rsqrt
   sigmoid
   slice
   slice_like
   softmax
   split
   sqrt
   square
   squeeze
   stack
   sum
   tanh
   tile
   transpose
   where
   zeros_like
