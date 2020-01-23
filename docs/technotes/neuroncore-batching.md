# Tech Note: Batching

Batching refers to the process of grouping multiple inference requests together and processing them as a group. Typically this will be done on a layer-by-layer basis, which thus allows for each set of weights in a given layer to be reused for each inference in the batch before needing to retrieve additional new weights. This enables Neuron to better amortize the cost of reading weights from the external memory (i.e. read weights from the memory once, and use them in multiple calculations), and thus improve the overall hardware efficiency. Batching is typically used as an optimization for throughput at the expense of higher latency.

The concept of batched inference is illustrated below, with a NeuronCore performing batched computation of a 3 layer neural network with a batch-size of 4. The NeuronCore reads weights from the external memory, and then performs the corresponding computations for all 4 inference-requests, thus better amortizing the cost of reading the weights from the memory. 
![Image:](./images/NeuronCoreBatching.png)
To enable batching in Neuron, a model should be explicitly compiled for a target batch-size by setting the input-tensor batch dimension accordingly. Users are encouraged to evaluate multiple batch sizes, in order to determine the optimal latency/throughput deployment-point (which is application dependent). 

During inference, dynamic batching can be used to process a larger client-side inference batch-size, and allow the framework to automatically break up the user-batch into smaller batch sizes, to match compiled batch-size. This technique increases the achievable throughput by hiding the framework-to-runtime overhead, and amortizing it over a larger batch size.

For example, the TensorFlow code snippet below enables batching, with dynamic-batching and a batch-size of N=4:

```
import numpy as np
import tensorflow.neuron as tfn

# To change the batch size, change the first dimension in example_input
batch_sz = 4
example_input = np.zeros([batch_sz,224,224,3], dtype='float16')

# Note: currently the following compilation flags are required for batching 
# (they will automatically enabled by default in the future)
compiler_args = ['--batching_en', '--rematerialization_en', '--spill_dis',
                 '--sb_size', str((batch_sz + 6)*10),
                 '--enable-replication', 'True']

tfn.saved_model.compile("rn50_fp16", 
                        "rn50_fp16_compiled/1", 
                        model_feed_dict={'input_1:0': example_input },
                        dynamic_batch_size=True,
                        compiler_args=compiler_args)
```

At runtime, the following TensorFlow code snippet shows that the model can accept inference requests with arbitrary batch size:

```
import tensorflow as tf
import tensorflow.neuron as tfn

predictor = tf.contrib.predictor.from_saved_model("rn50_fp16_compiled/1")
rt_batch_sz_list = [1, 4, 7, 8, 1024]
for rt_batch_sz in rt_batch_sz_list:
    example_input = np.zeros([rt_batch_sz,224,224,3], dtype='float16')
    model_feed_dict = {'input_1:0': example_input}
    result = predictor(model_feed_dict)
```


**Note1**: Currently, a know Neuron compiler issue may sometime lead to compilation error for large batch-sizes that don’t fit the on-chip memory. This limitation is being addressed and will be fixed in future releases of the compiler.
Additionally, compiler experimental flags are currently required, as shown in the code snippet above. This will be deprecated (set by default) in future releases.

**Note2**: To enable dynamic batching in TensorFlow, user should set an experimental argument `dynamic_batch_size=True` in `tfn.saved_model.compile` as shown in example above. This will be deprecated (set by default) in future releases.


