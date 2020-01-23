# Supported operators [PyTorch]

Current operator lists may be generated with these commands inside python:

```python
import torch.neuron
print(*torch.neuron.get_supported_operations(), sep='\n')
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


