.. _tensorflow-ref-neuron-tracing-api:

TensorFlow 2.x (``tensorflow-neuron``) Tracing API
============================================

The Neuron tracing API enables tracing TensorFlow 2.x models for deployment
on AWS Machine Learning Accelerators.

Method
------

``tensorflow.neuron.trace``

Description
-----------

Trace a ``keras.Model`` or a Python callable that can be decorated by
``tf.function``, and return an AWS-Neuron-optimized ``keras.Model`` that
can execute on AWS Machine Learning Accelerators. Tracing is ideal for
``keras.Model`` that accepts a list of ``tf.Tensor`` objects and returns
a list of ``tf.Tensor`` objects. It is expected that users will provide
example inputs, and the ``trace`` function will execute ``func``
symbolically and convert it to a ``keras.Model``.

The returned ``keras.Model`` will support inference only. Attributes or
variables held by the original function or ``keras.Model`` will be dropped.

The returned ``keras.Model`` can be exported as SavedModel and served using
TensorFlow Serving. Please see :ref:`tensorflow-serving` for more
information about exporting to saved model and serving using TensorFlow
Serving.

The returned ``keras.Model`` has an ``.on_neuron_ratio`` attribute
which shows the percentage of ops mapped to neuron hardware. This calculation
ignores PlaceholerOp, IdentityOp, ReadVariableOp and NoOp.

Options can be passed to Neuron compiler via the environment variable
``NEURON_CC_FLAGS``. For example, the syntax
``env NEURON_CC_FLAGS="--neuroncore-pipeline-cores=4"`` directs Neuron
compiler to compile each subgraph to fit in the specified number of
NeuronCores. This number can be less than the total available NeuronCores
on an Inf1 instance. See  :ref:`neuron-compiler-cli-reference` for more
information about compiler options.

Arguments
---------

-   **func:** The ``keras.Model`` or function to be traced.
-   **example_inputs:** A ``tf.Tensor`` or a tuple/list/dict of
    ``tf.Tensor`` objects for tracing the function. When ``example_inputs``
    is a ``tf.Tensor`` or a list of ``tf.Tensor`` objects, we expect
    ``func`` to have calling signature ``func(example_inputs)``. Otherwise,
    the expectation is that inference on ``func`` is done by calling
    ``func(*example_inputs)`` when ``example_inputs`` is a ``tuple``,
    or ``func(**example_inputs)`` when ``example_inputs`` is a ``dict``.
    The case where ``func`` accepts mixed positional and keyword arguments
    is currently unsupported.
-   **subgraph_builder_function:** (Optional) A callable with signature

    ``subgraph_builder_function(node : NodeDef) -> bool``
    (``NodeDef`` is defined in tensorflow/core/framework/node_def.proto)

    that is used as a call-back function to determine which part of
    the tensorflow GraphDef given by tracing ``func`` will be placed on
    Machine Learning Accelerators.

    If ``subgraph_builder_function`` is not provided, then ``trace`` will
    automatically place operations on Machine Learning Accelerators or
    on CPU to maximize the execution efficiency.

    If it is provided, and ``subgraph_builder_function(node)`` returns
    ``True``, and placing ``node`` on Machine Learning Accelerators
    will not cause deadlocks during execution, then ``trace`` will place
    ``node`` on Machine Learning Accelerators. If
    ``subgraph_builder_function(node)`` returns ``False``, then ``trace``
    will place ``node`` on CPU.

Special Flags
-------------

These are flags that get passed directly to the Neuron tracing API
(rather than the Neuron Compiler). The flags are still passed
via the environment variable ``NEURON_CC_FLAGS``.

-   **workdir:** example usage - ``NEURON_CC_FLAGS='--workdir ./artifacts'``
    will create a folder named artifacts in the current directory and
    save artifacts that can be used for debug.
-   **dynamic-batch-size:** example usage -
    ``NEURON_CC_FLAGS='--dynamic-batch-size'`` A flag to allow Neuron graphs to
    consume variable sized batches of data. Dynamic sizing is restricted to the
    0th dimension of a tensor.
-   **extract-weights (EXPERIMENTAL):** example usage -
    ``NEURON_CC_FLAGS='--extract-weights'`` will reduce the compiled
    model's protobuf size by taking the weights out of the protobuf.
    Useful for compiling large models that would exceed the 2GB protobuf
    size limit. This feature is experimental. Model performance is not
    guaranteed and the flag does not work in combination with
    ``--neuroncore-pipeline-cores``, ``--dynamic-batch-size``, models with
    multiple NEFFs, and models that are 4GB or greater. 
    Compiles models for the inf1.2xlarge instance.

Returns
-------

-  An AWS-Neuron-optimized ``keras.Model``.


Example Usage
-------------

.. code:: python

    import tensorflow as tf
    import tensorflow.neuron as tfn

    input0 = tf.keras.layers.Input(3)
    dense0 = tf.keras.layers.Dense(3)(input0)
    model = tf.keras.Model(inputs=[input0], outputs=[dense0])
    example_inputs = tf.random.uniform([1, 3])
    model_neuron = tfn.trace(model, example_inputs)  # trace
    # check to see how much of the model was compiled successfully
    print(model_neuron.on_neuron_ratio) 

    model_dir = './model_neuron'
    model_neuron.save(model_dir)
    model_neuron_reloaded = tf.keras.models.load_model(model_dir)


Example Usage with Manual Device Placement Using `subgraph_builder_function`
-------------

.. code:: python

    import tensorflow as tf
    import tensorflow.neuron as tfn

    input0 = tf.keras.layers.Input(3)
    dense0 = tf.keras.layers.Dense(3)(input0)
    reshape0 = tf.keras.layers.Reshape([1, 3])(dense0)
    output0 = tf.keras.layers.Dense(2)(reshape0)
    model = tf.keras.Model(inputs=[input0], outputs=[output0])
    example_inputs = tf.random.uniform([1, 3])

    def subgraph_builder_function(node):
        return node.op == 'MatMul'

    model_neuron = tfn.trace(
        model, example_inputs,
        subgraph_builder_function=subgraph_builder_function,
    )

.. important ::

    Although the old API ``tensorflow.neuron.saved_model.compile`` is still available under tensorflow-neuron 2.x,
    it supports only the limited capabilities of ``tensorflow.neuron.trace`` and will be deprecated in future releases.
