.. _tensorflow-ref-auto-replication-python-api:

TensorFlow Neuron (``tensorflow-neuron``) Auto Multicore Replication (Beta)
===================================================================================

The Neuron auto multicore replication Python API enables modifying TensorFlow 2.x
traced models so that they can be automatically replicated across multiple cores.
For Tensorflow-Serving models and TensorFlow 1.x models, see :ref:`tensorflow-ref-auto-replication-cli-api`

.. contents:: Table of contents
   :local:
   :depth: 1

TensorFlow Neuron TF 2.x (``tensorflow-neuron TF2.x``) Auto Multicore Replication Python API (Beta)
-----------------------------------------------------------------------------------------------------------

Method
^^^^^^

``tensorflow.neuron.auto_multicore``

Description
^^^^^^^^^^^

Converts an existing AWS-Neuron-optimized ``keras.Model`` and returns an auto-replication tagged
AWS-Multicore-Neuron-optimized  ``keras.Model`` that can execute on AWS Machine Learning Accelerators.
Like the traced model, the returned ``keras.Model`` will support inference only. Attributes or
variables held by the original function or ``keras.Model`` will be dropped.

The auto model replication feature in TensorFlow-Neuron enables you to
create a model once and the model parallel replication would happen
automatically. The desired number of cores can be less than the total available NeuronCores
on an Inf1 instance but not less than 1. This reduces framework memory usage as you are not
loading the same model multiple times manually. Calls to the returned model will execute the call
on each core in a round-robin fashion.

The returned ``keras.Model`` can be exported as SavedModel and served using
TensorFlow Serving. Please see :ref:`tensorflow-serving` for more
information about exporting to saved model and serving using TensorFlow
Serving.

Note that the automatic replication will only work on models compiled with pipeline size 1:
via ``--neuroncore-pipeline-cores=1``. If auto replication is not enabled, the model will default to
replicate on up to 4 cores.

See  :ref:`neuron-compiler-cli-reference` for more information about compiler options.

Arguments
^^^^^^^^^

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
-   **num_cores:** The desired number of cores where the model will be automatically
    replicated across

Returns
^^^^^^^

-  An AWS-Multicore-Neuron-optimized ``keras.Model``.


Example Python API Usage for TF2.x traced models:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code :: python

        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, input0_tensor)

        num_cores = 4
        multicore_model = tfn.auto_multicore(model_neuron, input0_tensor, num_cores=num_cores)
        multicore_model(input0_tensor)

Example Python API Usage for TF2.x saved models:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code :: python

        from tensorflow.python import saved_model

        input0_tensor = tf.random.uniform([1, 3])
        num_cores = 4
        reload_model = saved_model.load(model_dir)
        multicore_model = tfn.auto_multicore(reload_model, input0_tensor, num_cores=num_cores)

.. _tensorflow-ref-auto-replication-cli-api:

TensorFlow Neuron TF1.x/TF2.x (``tensorflow-neuron TF1.x/TF2.x``) Auto Multicore Replication CLI (Beta)
---------------------------------------------------------------------------------------------------------------

The Neuron auto multicore replication CLI  enables modifying TensorFlow 1.x and Tensorflow 2.x
traced saved models so that they can be automatically replicated across multiple cores. By performing
this call on Tensorflow Saved Models, we can support both Tensorflow-Serving and Tensorflow 1.x
without significant modifications to the code. Note that the python API does not support Tensorflow 1.x.

Method
^^^^^^

``tf-neuron-auto-multicore MODEL_DIR --num_cores NUM_CORES --new_model_dir NEW_MODEL_DIR``

Arguments
^^^^^^^^^

-   **MODEL_DIR:** The directory of a saved AWS-Neuron-optimized ``keras.Model``.
-   **NUM_CORES:** The desired number of cores where the model will be automatically
    replicated across
-   **NEW_MODEL_DIR:** The directory of where the AWS-Multicore-Neuron-optimized
    ``keras.Model`` will be saved

Example CLI Usage for TF 1.x and Tensorflow-Serving saved models:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code :: python

        tf-neuron-auto-multicore ./resnet --num_cores 8 --new_model_dir ./modified_resnet
