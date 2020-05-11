# Supported operators [PyTorch]

Current operator lists may be generated with these commands inside python:

```python
import torch.neuron
print(*torch.neuron.get_supported_operations(), sep='\n')
```
### PyTorch Neuron Release [1.0.1001.0]

Added

```
aten::detach
aten::floor
aten::gelu
aten::pow
aten::sigmoid
aten::split
```

Removed ( Reasons given alongside )

```
aten::embedding (does not meet performance criteria)
aten::erf (error function does not meet accuracy criteria)
aten::tf_dtype_from_torch (internal support function, not an operator)
```
### PyTorch Neuron Release [1.0.825.0]

### PyTorch Neuron Release [1.0.763.0]

* Please note. Starting with this release we will not publish primitives (prim::). 

* Previous release inaccurately listed these operators as aten ops, they are not. 

```
aten::tf_broadcastable_slice
aten::tf_padding
```

The following new operators are added in this release.

```
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
```
These operators were already supported previously (removing the two that were included by mistake)
```
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
```

### PyTorch Neuron Release [1.0.672.0]
No change

### PyTorch Neuron Release [1.0.552.0]

```
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
```


