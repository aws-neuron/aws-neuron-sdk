.. _tensorflow-tutorials:

TensorFlow Tutorials
====================

Before running a tutorial
-------------------------

You will run the tutorials on an inf1.6xlarge instance running Deep Learning AMI (DLAMI) to enable both compilation and deployment (inference) on the same instance. In a production environment we encourage you to try different instance sizes to optimize to your specific deployment needs. 

Follow instructions at :ref:`tensorflow-tutorial-setup` before running a TensorFlow tutorial on Inferentia. We recommend new users start with the ResNet-50 tutorial.


.. toctree::
   :hidden:

   /neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/tensorflow-tutorial-setup

.. _tensorflow-computervision:


Computer Vision
---------------

*  OpenPose tutorial :ref:`[html] </src/examples/tensorflow/openpose_demo/openpose.ipynb>` :tensorflow-neuron-src:`[notebook] <openpose_demo/openpose.ipynb>`
*  ResNet-50 tutorial :ref:`[html] </src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb>` :tensorflow-neuron-src:`[notebook] <tensorflow_resnet50/resnet50.ipynb>`
*  YOLOv4 tutorial :ref:`[html] <tensorflow-yolo4>`
*  YOLOv3 tutorial :ref:`[html] </src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb>` :tensorflow-neuron-src:`[notebook] <yolo_v3_demo/yolo_v3.ipynb>`
*  SSD300 tutorial :ref:`[html] <tensorflow-ssd300>`
*  Tensorflow 1.15 - Keras ResNet-50 optimization tutorial :ref:`[html] </src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb>` :tensorflow-neuron-src:`[notebook] <keras_resnet50/keras_resnet50.ipynb>`

.. toctree::
   :hidden:

   /src/examples/tensorflow/openpose_demo/openpose.ipynb
   /src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb
   /neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/yolo_v4_demo/yolo_v4_demo
   /src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb
   /neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/ssd300_demo/ssd300_demo
   /src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb


.. _tensorflow-nlp:

Natural Language Processing
---------------------------

*  Running TensorFlow BERT-Large with AWS Neuron :ref:`[html] <tensorflow-bert-demo>`

.. toctree::
   :hidden:

   /neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/bert_demo/bert_demo

   

.. _tensorflow-utilize-neuron:

Utilizing Neuron Capabilities
-----------------------------

*  NeuronCore Groups tutorial :ref:`[html] <tensorflow-neurocore-group>`
*  TensorFlow Serving tutorial :ref:`[html] <tensorflow-serving>`
*  NeuronCore Groups with TensorFlow Serving tutorial :ref:`[html] <tensorflow-serving-neurocore-group>`

.. toctree::
   :hidden:

   /neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/tutorial-tensorflow-NeuronCore-Group
   /neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/tutorial-tensorflow-serving
   /neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/tutorial-tensorflow-serving-NeuronCore-Group
