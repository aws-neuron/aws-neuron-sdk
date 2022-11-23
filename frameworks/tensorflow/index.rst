.. _tensorflow-neuron-main:
.. _tensorflow-neuron:

TensorFlow Neuron
=================
TensorFlow Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and Inferentia-based Amazon EC2 instances.

TensorFlow Neuron enables native TensorFlow models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes.

.. tab-set::


   .. tab-item:: Inference
        :name: torch-neuronx-training-main

        .. dropdown::  Setup Guide for Inf1 
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in
                

                .. toctree::
                    :maxdepth: 1


                    Fresh Install </frameworks/tensorflow/tensorflow-neuron/setup/tensorflow-install>
                    Update to Latest Release </frameworks/tensorflow/tensorflow-neuron/setup/tensorflow-update>
                    Install Previous Releases </frameworks/tensorflow/tensorflow-neuron/setup/tensorflow-install-prev>

        .. dropdown::  Tutorials (``tensorflow-neuron``)
                :class-title: sphinx-design-class-title-med
                :animate: fade-in
                
                .. tab-set::

                    .. tab-item:: Computer Vision Tutorials
                                :name:         

                                *  Tensorflow 1.x - OpenPose tutorial :ref:`[html] </src/examples/tensorflow/openpose_demo/openpose.ipynb>` :github:`[notebook] </src/examples/tensorflow/openpose_demo/openpose.ipynb>`
                                *  Tensorflow 1.x - ResNet-50 tutorial :ref:`[html] </src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb>` :github:`[notebook] </src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb>`
                                *  Tensorflow 1.x - YOLOv4 tutorial :ref:`[html] <tensorflow-yolo4>` :github:`[notebook] </src/examples/tensorflow/yolo_v4_demo/evaluate.ipynb>`
                                *  Tensorflow 1.x - YOLOv3 tutorial :ref:`[html] </src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb>` :github:`[notebook] </src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb>`
                                *  Tensorflow 1.x - SSD300 tutorial :ref:`[html] <tensorflow-ssd300>`
                                *  Tensorflow 1.x - Keras ResNet-50 optimization tutorial :ref:`[html] </src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb>` :github:`[notebook] </src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb>`

                                .. toctree::
                                        :hidden:

                                        /src/examples/tensorflow/openpose_demo/openpose.ipynb
                                        /src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb
                                        /frameworks/tensorflow/tensorflow-neuron/tutorials/yolo_v4_demo/yolo_v4_demo
                                        /src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb
                                        /frameworks/tensorflow/tensorflow-neuron/tutorials/ssd300_demo/ssd300_demo
                                        /src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb

                    .. tab-item:: Natural Language Processing (NLP) Tutorials
                                :name:
          

                                *  Tensorflow 1.x - Running TensorFlow BERT-Large with AWS Neuron :ref:`[html] <tensorflow-bert-demo>`
                                *  Tensorflow 2.x - HuggingFace Pipelines distilBERT with Tensorflow2 Neuron :ref:`[html] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>` :github:`[notebook] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`

                                .. toctree::
                                        :hidden:

                                        /frameworks/tensorflow/tensorflow-neuron/tutorials/bert_demo/bert_demo
                                        /src/examples/tensorflow/huggingface_bert/huggingface_bert


                    .. tab-item:: Utilizing Neuron Capabilities Tutorials
                                :name:
            

                                *  Tensorflow 1.x - Using NEURON_RT_VISIBLE_CORES with TensorFlow Serving :ref:`[html] <tensorflow-serving-neuronrt-visible-cores>`

                                .. toctree::
                                        :hidden:

                                        /frameworks/tensorflow/tensorflow-neuron/tutorials/tutorial-tensorflow-serving-NeuronRT-Visible-Cores



                .. note::

                        To use Jupyter Notebook see:

                        * :ref:`setup-jupyter-notebook-steps-troubleshooting`
                        * :ref:`running-jupyter-notebook-as-script` 


        .. dropdown::  Additional Examples (``tensorflow-neuron``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                * `AWS Neuron Samples GitHub Repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/tensorflow-neuron/inference>`_


        .. dropdown::  API Reference Guide (``tensorflow-neuron``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                .. toctree::
                    :maxdepth: 1

                    /frameworks/tensorflow/tensorflow-neuron/api-tracing-python-api
                    /frameworks/tensorflow/tensorflow-neuron/api-compilation-python-api
                    /frameworks/tensorflow/tensorflow-neuron/api-auto-replication-api

        .. dropdown::  
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in
                :open:
                

                .. toctree::
                    :maxdepth: 1


                    /release-notes/tensorflow/tensorflow-neuron/tensorflow-neuron
                    /release-notes/tensorflow/tensorflow-neuron/tensorflow-neuron-v2                  
                    /frameworks/tensorflow/tensorflow-neuron/tensorflow2-accelerated-ops
                    /release-notes/compiler/neuron-cc/neuron-cc-ops/neuron-cc-ops-tensorflow



   .. tab-item:: Training
        :name: torch-neuronx-training-main

        .. note::

           TensorFlow Neuron support is coming soon.






