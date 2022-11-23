.. _tensorflow-tutorials:

TensorFlow Tutorials
====================

Before running a tutorial
-------------------------

You will run the tutorials on an inf1.6xlarge instance running Deep Learning AMI (DLAMI) to enable both compilation and deployment (inference) on the same instance. In a production environment we encourage you to try different instance sizes to optimize to your specific deployment needs.

Follow instructions at :ref:`tensorflow-tutorial-setup` before running a TensorFlow tutorial on Inferentia. We recommend new users start with the ResNet-50 tutorial.


.. toctree::
   :hidden:

   /frameworks/tensorflow/tensorflow-neuron/tutorials/tensorflow-tutorial-setup

.. _tensorflow-computervision:


Computer Vision
---------------

*  Tensorflow 1.x - OpenPose tutorial :ref:`[html] </src/examples/tensorflow/openpose_demo/openpose.ipynb>` :tensorflow-neuron-src:`[notebook] <openpose_demo/openpose.ipynb>`
*  Tensorflow 1.x - ResNet-50 tutorial :ref:`[html] </src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb>` :tensorflow-neuron-src:`[notebook] <tensorflow_resnet50/resnet50.ipynb>`
*  Tensorflow 1.x - YOLOv4 tutorial :ref:`[html] <tensorflow-yolo4>` :tensorflow-neuron-src:`[notebook] <yolo_v4_demo/evaluate.ipynb>`
*  Tensorflow 1.x - YOLOv3 tutorial :ref:`[html] </src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb>` :tensorflow-neuron-src:`[notebook] <yolo_v3_demo/yolo_v3.ipynb>`
*  Tensorflow 1.x - SSD300 tutorial :ref:`[html] <tensorflow-ssd300>`
*  Tensorflow 1.x - Keras ResNet-50 optimization tutorial :ref:`[html] </src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb>` :tensorflow-neuron-src:`[notebook] <keras_resnet50/keras_resnet50.ipynb>`

.. toctree::
   :hidden:

   /src/examples/tensorflow/openpose_demo/openpose.ipynb
   /src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb
   /frameworks/tensorflow/tensorflow-neuron/tutorials/yolo_v4_demo/yolo_v4_demo
   /src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb
   /frameworks/tensorflow/tensorflow-neuron/tutorials/ssd300_demo/ssd300_demo
   /src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb


.. _tensorflow-nlp:

Natural Language Processing
---------------------------

*  Tensorflow 1.x - Running TensorFlow BERT-Large with AWS Neuron :ref:`[html] <tensorflow-bert-demo>`
*  Tensorflow 2.x - HuggingFace Pipelines distilBERT with Tensorflow2 Neuron :ref:`[html] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>` :github:`[notebook] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`

.. toctree::
   :hidden:

   /frameworks/tensorflow/tensorflow-neuron/tutorials/bert_demo/bert_demo
   /src/examples/tensorflow/huggingface_bert/huggingface_bert

.. _tensorflow-utilize-neuron:

Utilizing Neuron Capabilities
-----------------------------

*  Tensorflow 1.x & 2.x - Using NEURON_RT_VISIBLE_CORES with TensorFlow Serving :ref:`[html] <tensorflow-serving-neuronrt-visible-cores>`

.. toctree::
   :hidden:

   /frameworks/tensorflow/tensorflow-neuron/tutorials/tutorial-tensorflow-serving
   /frameworks/tensorflow/tensorflow-neuron/tutorials/tutorial-tensorflow-serving-NeuronRT-Visible-Cores
