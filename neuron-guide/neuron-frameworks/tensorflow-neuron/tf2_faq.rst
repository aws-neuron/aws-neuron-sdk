.. _tf2_faq:

TensorFlow 2.x FAQ
===================

.. contents::
   :local:
   :depth: 1


How do I get started with TensorFlow?
-------------------------------------

The easiest entry point is the tutorials offered by the AWS Neuron team. For beginners, the :ref:`HuggingFace Pipelines distilBERT Tutorial </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>` is a good place to start.

What TensorFlow versions are supported by Neuron?
-------------------------------------------------

The AWS Neuron provide well-tested tensorflow-neuron packages that work with a range of tensorflow official releases, as long as the version of tensorflow-neuron matches that of tensorflow. For example, you may install ``tensorflow-neuron==2.3.3.1.0.9999.0`` on top of ``tensorflow==2.3.3`` and expect them to work together.

Currently, tensorflow-neuron can work with tensorflow versions 2.1.4, 2.2.3, 2.3.3, 2.4.2, 2.5.0.

In a fresh Python environment, ``pip install tensorflow-neuron`` would bring in the highest version (2.5.0 as of 07/13/2021), which then pulls ``tensorflow==2.5.0`` into the current environment.

If you already have a particular version of tensorflow 2.x installed, then it is recommended to pay attention to the precise version of tensorflow-neuron and only install the desired one. For example, in an existing Python environment with ``tensorflow==2.3.3`` installed, you may install tensorflow-neuron by pip install ``tensorflow-neuron==2.3.3``, which will reuse the existing tensorflow installation.

What operators are supported?
-----------------------------

Due to fundamental backend design changes in the TensorFlow 2.x framework, the concept of "supported graph operators" is no longer well-defined. Please refer to :ref:`Accelerated Python APIs and graph operators <tensorflow-ref-neuron-accelerated-ops>` for a guide to the set of TensorFlow 2.x Python APIs and graph operators that can be accelerated by Neuron.

How do I compile my model?
--------------------------

It is achieved by a new public API called tfn.trace, which resembles the compilation API of AWS Neuron PyTorch integration. Programmatically, customers would be able to execute the following code.

.. code::

    import tensorflow as tf
    import tensorflow.neuron as tfn

    ...
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model_neuron = tfn.trace(model, example_inputs)
    model_neuron.save('./model_neuron_dir')
    ...
    model_loaded = tf.saved_model.load('./model_dir')
    predict_func = model_loaded['serving_default']
    model_loaded_neuron = tfn.trace(predict_func, example_inputs2)
    model_loaded_neuron.save('./model_loaded_neuron_dir')
    ...

How do I deploy my model?
-------------------------

Python tensorflow
^^^^^^^^^^^^^^^^^

Pre-compiled models can be saved and reloaded back into a Python environment using regular tensorflow model loading APIs, as long as tensorflow-neuron is installed.

.. code::

    import tensorflow as tf

    model = tf.keras.models.load_model('./model_loaded_neuron_dir')
    example_inputs = ...
    output = model(example_inputs)

tensorflow-serving
^^^^^^^^^^^^^^^^^^

Pre-compiled models can be saved into SavedModel format via tensorflow SavedModel APIs

.. code::

    import tensorflow as tf
    import tensorflow.neuron as tfn

    ...
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model_neuron = tfn.trace(model, example_inputs)
    tf.saved_model.save(model_neuron, './model_neuron_dir/1')

The generated SavedModel './model_neuron_dir' can be loaded into tensorflow-model-server-neuron, which can be installed through apt or yum based on the type of the operating system. For example, on Ubuntu 18.04 LTS the following command installs and launches a tensorflow-model-server-neuron on a pre-compiled SavedModel.

.. code::

    sudo apt install tensorflow-model-server-neuron
    # --model_base_path needs to be an absolute path
    tensorflow_model_server_neuron --model_base_path=$(pwd)/model_neuron_dir

Where can I find tutorials and examples ?
-----------------------------------------

:ref:`HuggingFace Pipelines distilBERT Tutorial </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>` is a good place to start.


How to debug or profile my model?
---------------------------------

:ref:`AWS Neuron TensorBoard integration <neuron-plugin-tensorboard>` provides visibility into what is happening inside of the Neuron runtime, and allows a more fine-grained (but also more hardware-awared) reasoning on where to improve the performance of machine learning applications.

