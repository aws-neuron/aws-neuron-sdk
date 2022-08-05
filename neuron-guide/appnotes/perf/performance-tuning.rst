.. _appnote-performance-tuning:

Performance Tuning
==================

.. important ::
  NeuronCore Groups (NCG) is deprecated, please see :ref:`eol-ncg` and :ref:`neuron-migrating-apps-neuron-to-libnrt` for more details.

This guide is intended to provide the reader with an in-depth
understanding on how to optimize neural network performance on
Inferentia for both throughput and latency. For simplicity, the guide
uses TensorFlow and ResNet-50 model as a teaching example to learn how
choosing between different compile-time optimizations (e.g. Batching and
NeuronCore Pipeline), as well as model-serving optimizations (e.g.
multi-threading and dynamic-batching) improves inference performance.

The following guides are considered prerequisites for this tutorial:

-  :ref:`/src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb`
-  :ref:`tensorflow-serving-neurocore-group`
-  :ref:`neuron-batching`
-  :ref:`neuroncore-pipeline`

Batching and pipelining (technical background)
----------------------------------------------

Neuron provides developers with various performance optimization features.

Two of the most widely used features are batching and pipelining. Both
techniques aim to keep the data close to the compute engines, but achieve
this data locality in different ways. In batching it is achieved by loading
the data into an on-chip cache and reusing it multiple times for multiple
different model-inputs, while in pipelining this is achieved by caching all
model parameters into the on-chip cache across multiple NeuronCores and
streaming the calculation across them.

As a general rule of thumb, batching is preferred for applications that
aim to optimize throughput and cost at the expense of latency, while
pipelining is preferred for applications with high-throughput
requirement under a strict latency budget.

Compiling for batching optimization
-----------------------------------

To enable the batching optimization, we first need to compile the model
for a target batch-size. This is done by specifying the batch size in
the input tensor's batch dimension during compilation. Users are
encouraged to evaluate multiple batch sizes in order to determine the
optimal latency/throughput deployment-point, which is application
dependent.

For example, the code snippet below enables batching on a ResNet50
model, with a batch-size of 5:

.. code:: python

   import numpy as np
   import tensorflow.neuron as tfn

   # To change the batch size, change the first dimension in example_input
   batch_size = 5
   example_input = np.zeros([batch_size,224,224,3], dtype='float16')

   tfn.saved_model.compile("rn50_fp16",
                           "rn50_fp16_compiled/1",
                           model_feed_dict={'input_1:0': example_input },
                           dynamic_batch_size=True)

.. note::

   Depending on the neural network size, Neuron will have a maximum
   batch size that works optimally on Inferentia. If
   an unsupported batch size is used, an internal compiler error message
   will be displayed.
   A simple way to explore optimal batch size for your specific model is to
   increment the batch size from 1 upward, one at a time, and test
   application performance.

Compiling for pipeline optimization
-----------------------------------

With NeuronCore Pipeline mode, Neuron stores the model parameters onto
the Inferentias' local caches, and streams the inference requests across
the available NeuronCores, as specified by the
``--neuroncore-pipeline-cores`` compiler argument. For example, to
compile the model to fit pipeline size of four Inferentia devices (16
NeuronCores) avaliable in the inf1.6xlarge instance size:

.. code:: python

   import numpy as np
   import tensorflow.neuron as tfn

   compiler_args = ['--neuroncore-pipeline-cores', '16']
   example_input = np.zeros([1,224,224,3], dtype='float16')
   tfn.saved_model.compile("rn50_fp16",
                           "rn50_fp16_compiled/1",
                           model_feed_dict={'input_1:0': example_input },
                           compiler_args=compiler_args)

The minimum number of NeuronCores needed to run a compiled model can be
found using Neuron Check Model tool. Please see :ref:`neuron_check_model`.

Model-serving inference optimizations
-------------------------------------

In order to fully realize the maximum throughput of the compiled model
(for either batching and pipelining), users need to launch multiple host
CPU threads to feed inputs into the Neuron pipeline. The number of
threads need to be larger than the specified maximum number of
NeuronCores.

Additionally, dynamic batching can be used to process a larger
client-side inference batch-size and the framework automatically breaks
up the user-batch into smaller batch sizes to match the compiled
batch-size. This technique increases the achievable throughput by hiding
the framework-to-neuron overhead, and amortizing it over a larger batch
size. To use dynamic batching, set the argument
``--dynamic_batch_size=True`` during compilation and send larger
inference batch size (user inference batch size) that is equal to a
multiple of the compiled batch size.

Both of methods can be applied together if that shows improvement in
performance. However, multi-threading is always needed as a first step
to achieve high throughput. You may need to experiment in order to find
the right optimization settings for your application.

By default, the framework sets the number of outstanding inference
requests to the total number of NeuronCores plus three. This can be
changed by setting the NEURON_MAX_NUM_INFERS environment variable. For
example, if the compiled model includes some CPU partitions (as when
Neuron compiler decided some operations are more efficient to execute on
CPU), the number of threads should be increased to account for the
additional compute performed on the CPU. Note that the available
instance host memory size should be taken into consideration to avoid
out-of-memory errors. As above, you need to experiment in order to find
the right optimization settings for your application.

.. note::

   By default the framework allocates NeuronCore Group size to
   match the size of the compiled model. The size of the model is the
   number of NeuronCores limit passed to compiler during compilation
   (``--neuroncore-pipeline-cores`` option). For more information see
   :ref:`tensorflow-serving-neurocore-group`.

Other considerations
--------------------

Mixed Precision
~~~~~~~~~~~~~~~

You can find more defails about performance and accuracy trade offs
in :ref:`mixed-precision`.


Operator support
~~~~~~~~~~~~~~~~

The Neuron Compiler maintains an evolving list of supported operators
for each framework: :ref:`neuron-supported-operators`

AWS Neuron handles unsupported operators by partitioning the graph into
subgraph, and executing them on different targets (e.g. NeuronCore
partition, CPU partition). If the entire model can run on Inferentia
(i.e. all operators are supported), then the model will be compiled into
a single subgraph which will be executed by a NeuronCore Group.

Debug
~~~~~

You can examine the post-compiled model to view the compilation results
using the Neuron plugin for TensorBoard.
See :ref:`tensorboard-plugin-visualize-graph`.

ResNet-50 optimization example
------------------------------

For an example demonstrating the concepts described here, see
:ref:`/src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb`
