.. _tensorflow-ref-neuron-analyze_model-api:

TensorFlow 2.x (``tensorflow-neuron``) analyze_model API
========================================================

Method
------

``tensorflow.neuron.analyze_model``

Description
-----------

Analyzes a ``keras.Model`` or a Python callable that can be decorated by
``tf.function`` for it's compatability with neuron. It displays supported 
vs. unsupported operators in the model as well as percentages and counts of 
each operator and returns a dictionary with operator statistics.

Arguments
---------

-   **func:** The ``keras.Model`` or function to be analyzed.
-   **example_inputs:** A ``tf.Tensor`` or a tuple/list/dict of
    ``tf.Tensor`` objects for tracing the function. When ``example_inputs``
    is a ``tf.Tensor`` or a list of ``tf.Tensor`` objects, we expect
    ``func`` to have calling signature ``func(example_inputs)``. Otherwise,
    the expectation is that inference on ``func`` is done by calling
    ``func(*example_inputs)`` when ``example_inputs`` is a ``tuple``,
    or ``func(**example_inputs)`` when ``example_inputs`` is a ``dict``.
    The case where ``func`` accepts mixed positional and keyword arguments
    is currently unsupported.

Returns
-------

-  A results ``dict`` with these keys: ``'percent_supported', 'supported_count', 
  'total_count', 'supported_operators', 'unsupported_operators', 'operators', 
  'operator_count'``.

Example Usage
-------------

.. code:: python

    import tensorflow as tf
    import tensorflow.neuron as tfn

    input0 = tf.keras.layers.Input(3)
    dense0 = tf.keras.layers.Dense(3)(input0)
    model = tf.keras.Model(inputs=[input0], outputs=[dense0])
    example_inputs = tf.random.uniform([1, 3])
    results = tfn.analyze_model(model, example_inputs)