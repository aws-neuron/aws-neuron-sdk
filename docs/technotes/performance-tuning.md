# Application Note: Performance Tuning

This guide is intended to provide the reader with an in-depth understanding on how to optimize neural network performance on Inferentia for both throughput and latency. For simplicity, the guide uses TensorFlow and ResNet-50 model as a teaching example to learn how choosing between different compile-time optimizations (e.g. Batching and NeuronCore Pipeline), as well as model-serving optimizations (e.g. multi-threading and dynamic-batching) improves inference performance.

The following guides are considered prerequisites for this tutorial:

* [Getting Started with TensorFlow-Neuron (ResNet-50 Tutorial)](https://github.com/aws/aws-neuron-sdk/blob/master/docs/tensorflow-neuron/tutorial-compile-infer.md)
* [Configuring NeuronCore Groups](https://github.com/aws/aws-neuron-sdk/blob/master/docs/tensorflow-neuron/tutorial-NeuronCore-Group.md)
* [Tech Note: Batching](https://github.com/aws/aws-neuron-sdk/blob/master/docs/technotes/neuroncore-batching.md)
* [Tech Note: NeuronCore Pipeline](https://github.com/aws/aws-neuron-sdk/blob/master/docs/technotes/neuroncore-pipeline.md)

## Batching and pipelining (technical background)

Neuron provides developers with various performance optimization knobs. Two of the most widely used ones are batching and pipelining. Both techniques aim to keep the data close to the compute engines, but achieve that in different ways. In batching it is achieved by loading the data into an on-chip cache and reusing it multiple times for multiple different model-inputs, while in pipelining this is achieved by caching all model parameters into the on-chip cache across multiple NeuronCores and streaming the calculation across them.

As a general rule of thumb, batching is preferred for applications that aim to optimize throughput and cost at the expense of latency, while pipelining is preferred for applications with high-throughput requirement under a strict latency budget.

## Compiling for batching optimization

To enable the batching optimization, we first need to compile the model for a target batch-size. This is done by specifying the batch size in the input tensor's batch dimension during compilation. Users are encouraged to evaluate multiple batch sizes in order to determine the optimal latency/throughput deployment-point, which is application dependent.

For example, the code snippet below enables batching on a ResNet50 model, with a batch-size of 5:

```python
import numpy as np
import tensorflow.neuron as tfn

# To change the batch size, change the first dimension in example_input
batch_size = 5
example_input = np.zeros([batch_size,224,224,3], dtype='float16')

# Note: Users should temporarily use the following compilation flags when
# batch size is larger than 1. These flags are only applicable to CNNs
# (ResNet50 and similar models) and will be deprecated in the future.
compiler_args = ['--batching_en', '--rematerialization_en', '--spill_dis',
                 '--sb_size', str((batch_size + 6)*10),
                 '--enable-replication', 'True']

tfn.saved_model.compile("rn50_fp16",
                        "rn50_fp16_compiled/1",
                        model_feed_dict={'input_1:0': example_input },
                        dynamic_batch_size=True,
                        compiler_args=compiler_args)
```

**Note 1:** Users should temporarily use the following compilation flags when batch size is larger than 1: `--batching_en --rematerialization_en --spill_dis --sb_size <(batch_size + 6)*10> --enable-replication=True`. These flags are only applicable to CNNs (ResNet50 and similar models) and will be deprecated in the future.

**Note 2:** Depending on the neural network size, Neuron will have a maximum batch size that works optimally on Inferentia. Currently, FP16 ResNet50 is supported up to batch 5 only. Additionally, ResNet50 with FP32 input is limited to batch 1 only. These limitations are being addressed and will be fixed in a future releases of the compiler.  If a unsupported batch size is used, an internal compiler error message will be displayed (see [Known Issues](#known-issues) section below).

## Compiling for pipeline optimization

With NeuronCore Pipeline mode, Neuron stores the model parameters onto the Inferentias' local caches, and streams the inference requests across the available NeuronCores, as specified by the `--neuroncore-pipeline-cores` compiler argument. For example, to compile the model to fit pipeline size of four Inferentia devices (16 NeuronCores) avaliable in the inf1.6xlarge instance size:

```python
import numpy as np
import tensorflow.neuron as tfn

compiler_args = ['--neuroncore-pipeline-cores', '16']
example_input = np.zeros([1,224,224,3], dtype='float16')
tfn.saved_model.compile("rn50_fp16",
                        "rn50_fp16_compiled/1",
                        model_feed_dict={'input_1:0': example_input },
                        compiler_args=compiler_args)
```


**Note:** If static weights flag is set and there is not enough NeuronCore cache memory to support fully-cached weights, the compiler will emit an internal compiler error message. To address such an error, users could use a larger NeuronCore Group (larger instance size). See [Known Issues](#known-issues) section below for more details.


## Model-serving inference optimizations

In order to fully realize the maximum throughput of the compiled model (for either batching and pipelining), users need to launch multiple host CPU threads to feed inputs into the Neuron pipeline. The number of threads need to be larger than the specified maximum number of NeuronCores.

Additionally, dynamic batching (framework optimization currently supported only by TensorFlow-Neuron) can be used to process a larger client-side inference batch-size and the framework automatically breaks up the user-batch into smaller batch sizes to match the compiled batch-size. This technique increases the achievable throughput by hiding the framework-to-neuron overhead, and amortizing it over a larger batch size. To use dynamic batching, set the argument `--dynamic_batch_size=True` during compilation and send larger inference batch size (user inference batch size) that is equal to a multiple of the compiled batch size.

Both of methods can be applied together if that shows improvement in performance. However, multi-threading is always needed as a first step to achieve high throughput. You may need to experiment in order to find the right optimization settings for your application.

By default, the framework sets the number of outstanding inference requests to the total number of NeuronCores plus three. This can be changed by setting the NEURON_MAX_NUM_INFERS environment variable. For example, if the compiled model includes some CPU partitions (as when Neuron compiler decided some operations are more efficient to execute on CPU), the number of threads should be increased to account for the additional compute performed on the CPU. Note that the available instance host memory size should be taken into consideration to avoid out-of-memory errors. As above, you need to experiment in order to find the right optimization settings for your application.

**Note:** By default the framework allocates NeuronCore Group size to match the size of the compiled model. The size of the model is the number of NeuronCores limit passed to compiler during compilation (`--neuroncore-pipeline-cores` option). For more information see [Tutorial TensorFlow-Neuron NeuronCore Groups](https://github.com/aws/aws-neuron-sdk/blob/master/docs/tensorflow-neuron/tutorial-NeuronCore-Group.md).

## Other considerations

### Mixed Precision

Reduced precision data-types are typically used to improve performance. In the example below, we convert all operations to FP16. Neuron also supports conversion to a mixed-precision graph, wherein only the weights and the data inputs to matrix multiplies and convolutions are converted to FP16, while the rest of the intermediate results are kept at FP32.

The Neuron compiler is able to automatically convert (also referred to as auto-cast) from FP32 model to bfloat16 for execution on Inferentia. While the larger (compared to fp16 model) size of input/output tensors being transferred to/from Inferentia may add some execution overhead, this feature will, in most cases, produce similar accuracy to FP32 and will not require to downcast or retrain models.

To selectively cast only inputs to MatMul and Conv operators, use option “`--fp32-cast=matmult`“.  This option may be required in certain networks such as BERT where additional accuracy is desired. **Note:** this option is experimental and may cause compiler to crash; please file issue to request further support.

For a more efficient data transfer and use of Inferentia, using a pre-trained FP16 model is suggested. If not, it is also possible to use a pre-casting script to convert FP32 model to be used as FP16.


### Operator support

The Neuron Compiler maintains an evolving list of supported operators for each framework:
* [TensorFlow-Neuron](https://github.com/aws/aws-neuron-sdk/blob/master/release-notes/neuron-cc-ops/neuron-cc-ops-tensorflow.md)
* [MXNet-Neuron](https://github.com/aws/aws-neuron-sdk/blob/master/release-notes/neuron-cc-ops/neuron-cc-ops-mxnet.md)
* [PyTorch-Neuron](https://github.com/aws/aws-neuron-sdk/blob/master/release-notes/neuron-cc-ops/neuron-cc-ops-pytorch.md)

AWS Neuron handles unsupported operators by partitioning the graph into subgraph, and executing them on different targets (e.g. NeuronCore partition, CPU partition). If the entire model can run on Inferentia (i.e. all operators are supported), then the model will be compiled into a single subgraph which will be executed by a NeuronCore Group.

### Debug

You can examine both the pre-compiled model to determine what portions of the graph can be compiled to Inferentia and also the post-compiled model to view the compilation results using Tensorboard-Neuron. See [TensorBoard Neuron: How To Check Neuron Compatibility](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-tools/getting-started-tensorboard-neuron.md#how-to-check-neuron-compatibility)


## ResNet-50 optimization example

For an example demonstrating the concepts described here, see [ResNet-50 optimization example](../../src/examples/tensorflow/keras_resnet50/README.md)
