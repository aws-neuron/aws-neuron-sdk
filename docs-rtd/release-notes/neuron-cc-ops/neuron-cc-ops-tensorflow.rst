.. _neuron-cc-ops-tensorflow:

Supported operators [TensorFlow]
================================

To see a list of supported operators for TensorFlow, run the following command:

``neuron-cc list-operators --framework TENSORFLOW``


# Supported operators [TensorFlow]

### Neuron Compiler Release 1.12.0 [XXXX]

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


### Neuron Compiler Release [1.0.24045.0]

Added ```FloorDiv```, ```Softplus```, ```Unstack```


### Neuron Compiler Release [1.0.18001]

No changes


### Neuron Compiler Release [1.0.16764]

Added:   
```
LogSoftmax
Neg
ResizeBilinear
ResizeNearestNeighbor
```

### Neuron Compiler Release [1.0.15275]

Added 
``` 
Neg 
```

Removed
```
Log
``` 
(was inadvertently advertised as supported)


### Neuron Compiler Release [1.0.12696]

No changes

### Neuron Compiler Release [1.0.9410]

No changes

### Neuron Compiler Release [1.0.7878]

No changes

### Neuron Compiler Release [1.0.6801]

No changes

### Neuron Compiler Release [1.0.5939]

No changes

### Neuron Compiler Release [1.0.5301]

No changes

### Neuron Compiler Release [1.0.4680.0]

Inital

### The current list

::


   Add
   AddV2
   AdjustContrastv2
   AdjustSaturation
   All
   AvgPool
   BatchMatMul
   BatchMatMulV2
   BatchToSpaceND
   BiasAdd
   BroadcastTo
   Cast
   Ceil
   Cholesky
   Concat
   ConcatV2
   Const
   Conv2D
   Conv2DBackpropInput
   Conv3D
   CropAndResize
   Equal
   Exp
   ExpandDims
   Fill
   Floor
   FloorDiv
   FusedBatchNorm
   Greater
   GreaterEqual
   HSVToRGB
   Identity
   InvertPermutation
   L2Loss
   LeakyRelu
   Less
   LessEqual
   Log
   Log1p
   LogicalAnd
   LogicalNot
   LogicalOr
   LogSoftmax
   LRN
   MatMul
   MatrixBandPart
   MatrixDiag
   MatrixSetDiag
   MatrixTriangularSolve
   Max
   Maximum
   MaxPool
   MaxPool3D
   Mean
   Min
   Minimum
   MirrorPad
   Mul
   Neg
   NoOp
   NotEqual
   Pack
   Pad
   PadV2
   Placeholder
   Pow
   Prod
   RandomUniform
   Range
   RealDiv
   Reciprocal
   Relu
   Relu6
   Reshape
   ResizeBilinear
   ResizeNearestNeighbor
   ReverseV2
   RGBToHSV
   Round
   Rsqrt
   Select
   Shape
   Sigmoid
   Sign
   Slice
   Softmax
   SoftmaxCrossEntropyWithLogits
   Softplus
   SpaceToBatchND
   Split
   SplitV
   Sqrt
   Square
   SquaredDifference
   Squeeze
   StopGradient
   StridedSlice
   Sub
   Sum
   Tanh
   Tile
   Transpose
   Unpack
   UnsortedSegmentSum
   ZerosLike





