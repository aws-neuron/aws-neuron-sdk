.. _neuron-cc-ops-tensorflow:

Supported operators [TensorFlow]
================================

.. contents:: Table of Contents
   :local:
   :depth: 1

.. _neuron-compiler-release-1270:

Neuron Compiler Release [1.2.7.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1220:

Neuron Compiler Release [1.2.2.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added

::

 AdjustContrastv2
 AdjustSaturation
 BroadcastTo
 Cholesky
 Conv2DBackpropInput
 Conv3D
 CropAndResize
 FloorDiv
 HSVToRGB
 InvertPermutation
 L2Loss
 Log1p
 MatrixBandPart
 MatrixDiag
 MatrixSetDiag
 MatrixTriangularSolve
 MaxPool3D
 MirrorPad
 RGBToHSV
 Range
 SoftmaxCrossEntropyWithLogits
 SquaredDifference
 StopGradient
 Unpack
 UnsortedSegmentSum


To see a list of supported operators for TensorFlow, run the following command:

``neuron-cc list-operators --framework TENSORFLOW``

.. _neuron-compiler-release-10240450:

Neuron Compiler Release [1.0.24045.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added ``FloorDiv``, ``Softplus``, ``Unstack``


.. _neuron-compiler-release-1018001:

Neuron Compiler Release [1.0.18001]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1016764:

Neuron Compiler Release [1.0.16764]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added:

::

   LogSoftmax
   Neg
   ResizeBilinear
   ResizeNearestNeighbor

.. _neuron-compiler-release-1015275:

Neuron Compiler Release [1.0.15275]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added

::

   Neg 

Removed

::

   Log

(was inadvertently advertised as supported)

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

No changes

.. _neuron-compiler-release-105301:

Neuron Compiler Release [1.0.5301]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No changes

.. _neuron-compiler-release-1046800:

Neuron Compiler Release [1.0.4680.0]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   Add
   AddV2
   All
   AvgPool
   BatchMatMul
   BatchMatMulV2
   BatchToSpaceND
   BiasAdd
   Cast
   Ceil
   Concat
   ConcatV2
   Const
   Conv2D
   Equal
   Exp
   ExpandDims
   Fill
   Floor
   FusedBatchNorm
   Greater
   GreaterEqual
   Identity
   LRN
   LeakyRelu
   Less
   LessEqual
   Log
   LogicalAnd
   LogicalNot
   LogicalOr
   MatMul
   Max
   MaxPool
   Maximum
   Mean
   Min
   Minimum
   Mul
   NoOp
   NotEqual
   Pack
   Pad
   PadV2
   Placeholder
   Pow
   Prod
   RandomUniform
   RealDiv
   Reciprocal
   Relu
   Relu6
   Reshape
   ReverseV2
   Round
   Rsqrt
   Select
   Shape
   Sigmoid
   Sign
   Slice
   Softmax
   SpaceToBatchND
   Split
   SplitV
   Sqrt
   Square
   Squeeze
   StridedSlice
   Sub
   Sum
   Tanh
   Tile
   Transpose
   ZerosLike
