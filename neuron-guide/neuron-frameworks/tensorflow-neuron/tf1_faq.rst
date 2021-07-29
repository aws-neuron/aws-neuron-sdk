.. _tf1_faq:

TensorFlow 1.x FAQ
===================

.. contents::
   :local:
   :depth: 1

How do I get started with TensorFlow?
-------------------------------------

The easiest entry point is the tutorials offered by the AWS Neuron team. For beginners, the :ref:`ResNet50 tutorial </src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb>` is a good place to start.

What TensorFlow versions are supported by Neuron?
-------------------------------------------------

TensorFlow version 1.15.5

What operators are supported?
-----------------------------
``neuron-cc list-operators --framework TENSORFLOW`` provides a list of supported TensorFlow 1.x operators, and they are the operators that run on the machine learning accelerator. Note that operators not in this list are still expected to work with the supported operators in native TensorFlow together, although not accelerated by the hardware.

How do I compile my model?
--------------------------

tensorflow-neuron includes a public-facing compilation API called tfn.saved_model.compile. More can be found here :ref:`tensorflow-ref-neuron-compile-api`.

How do I deploy my model?
-------------------------

Same way as deploying any tensorflow `SavedModel <https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#save-and-restore-models>`_. In Python TensorFlow, the easiest way is through the `tf.contrib.predictor module <https://docs.w3cub.com/tensorflow~python/tf/contrib/predictor/from_saved_model>`_. If a Python-free deployment is preferred for performance or some other reasons, `tensorflow-serving <https://www.tensorflow.org/tfx/guide/serving>`_ is a great choice and the AWS Neuron team provides pre-built model server apt/yum packages named as ``tensorflow-model-server-neuron``.

Where can I find tutorials and examples ?
----------------------------------------------------------

:ref:`tensorflow-tutorials` is a great place to start with.

Is XLA supported?
-----------------

No, the AWS Neuron TensorFlow 1.x integration project was done without reusing any component from Google’s XLA compiler project, and does not work with mechanisms such as XLA `JIT-clustering <https://www.tensorflow.org/xla/tutorials/autoclustering_xla>`_.

How to debug or profile my model?
-----------------------------

At TensorFlow level, the `v1 profiler <https://www.tensorflow.org/api_docs/python/tf/compat/v1/profiler/Profiler>`_ is a great tool that provides operator-level breakdown of the inference execution time. Additionally, the :ref:`AWS Neuron TensorBoard integration <neuron-plugin-tensorboard>` provides visibility into what is happening inside of the Neuron runtime, and allows a more fine-grained (but also more hardware-awared) reasoning on where to improve the performance of machine learning applications.
