# Batching

Batching refers to the process of grouping multiple inference requests together, and processing them as a group. This enables Neuron to better amortize the cost of reading weights from the external memory (i.e. read weights from the memory once, and use them in multiple calculations), and thus improve the overall hardware efficiency. Batching is typically used for optimizing throughput at the expense of latency.

The concept of batched inference is illustrated below, with a NeuronCore performing batched computation of a 3 layer computation graph with a batch-size of 4. The NeuronCore reads weights from the external memory, and then perform the corresponding computations for all 4 inference-requests, thus better amortizing the cost of reading the weights from the memory. 
![Image:](./images/NeuronCoreBatching.png)

To enable batching in Neuron, a model should be explicitly compiled for a target batch-size by invoking the compiler with the  `--batching_en` flag, and by setting the input-tensor batch dimension accordingly. Users are encouraged to evaluate multiple batch sizes, in order to determine the optimal latency/throughput deployment-point (which is application dependent).

For example, the TensorFlow code snippet below enables batching, with a batch-size of N=4:

```
import numpy as np
import tensorflow.neuron as tfn


# to change the batch size, change the first dimension in example_input
example_input = np.zeros([4,224,224,3], dtype='float16') 
tfn.saved_model.compile("rn50_fp16", 
                        "rn50_fp16_compiled/1", 
                        model_feed_dict={'input_1:0' : example_input },
                        compiler_args = ['--batching_en'])
```


