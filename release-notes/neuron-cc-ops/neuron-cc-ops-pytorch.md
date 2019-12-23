# Supported operators [PyTorch]

Current operator lists may be generated with these commands inside python:

```python
import torch.neuron
print(torch.neuron.get_supported_operations())
```

### PyTorch Neuron Release [1.0.552.0]

```
_convolution
adaptive_avg_pool2d
add
add_
addmm
avg_pool2d
batch_norm
cat
dimension_value
dropout
flatten
max_pool2d
mul
relu_
t
tanh
tf_broadcastable_slice
tf_padding
values
```


