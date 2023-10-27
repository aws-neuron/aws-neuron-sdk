.. _pytorch-tutorials:

PyTorch Neuron Tutorials
====================

Before running a tutorial
-------------------------

You will run the tutorials on an inf1.6xlarge instance running Deep Learning AMI (DLAMI) to enable both compilation and deployment (inference) on the same instance. In a production environment we encourage you to try different instance sizes to optimize to your specific deployment needs. 

Follow instructions at :ref:`pytorch-tutorial-setup` before running a PyTorch tutorial on Inferentia . We recommend new users start with the ResNet-50 tutorial.


.. toctree::
   :hidden:

   /neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/pytorch-tutorial-setup

.. _pytorch-computervision:

Computer Vision
---------------

* ResNet-50 tutorial :ref:`[html] </src/examples/pytorch/resnet50.ipynb>` :pytorch-neuron-src:`[notebook] <resnet50.ipynb>`
* PyTorch YOLOv4 tutorial :ref:`[html] </src/examples/pytorch/yolo_v4.ipynb>` :pytorch-neuron-src:`[notebook] <yolo_v4.ipynb>`

.. toctree::
   :hidden:
   
   /src/examples/pytorch/resnet50.ipynb
   /src/examples/pytorch/yolo_v4.ipynb

.. _pytorch-nlp:

Natural Language Processing
---------------------------

* HuggingFace pretrained BERT tutorial :ref:`[html] </src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb>` :pytorch-neuron-src:`[notebook] <bert_tutorial/tutorial_pretrained_bert.ipynb>`
* HuggingFace pretrained BERT tutorial with shared weights :ref:`[html] </src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert_shared_weights.ipynb>` :pytorch-neuron-src:`[notebook] <bert_tutorial/tutorial_pretrained_bert_shared_weights.ipynb>`
* Bring your own HuggingFace pretrained BERT container to Sagemaker Tutorial :ref:`[html] </src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>` :pytorch-neuron-src:`[notebook] <byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>`
* LibTorch C++ tutorial :ref:`[html] <pytorch-tutorials-libtorch>`
* TorchServe tutorial :ref:`[html] <pytorch-tutorials-torchserve>`
* HuggingFace MarianMT tutorial :ref:`[html] </src/examples/pytorch/transformers-marianmt.ipynb>` :pytorch-neuron-src:`[notebook] <transformers-marianmt.ipynb>`


.. toctree::
   :hidden:
   
   /src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb
   /src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert_shared_weights.ipynb
   /src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb
   /neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/tutorial-libtorch
   /frameworks/torch/torch-neuron/tutorials/tutorial-torchserve
   /src/examples/pytorch/transformers-marianmt.ipynb



.. _pytorch-utilize-neuron:

Utilizing Neuron Capabilities
-----------------------------


* BERT TorchServe tutorial :ref:`[html] <pytorch-tutorials-torchserve>`
* NeuronCore Pipeline tutorial :ref:`[html] </src/examples/pytorch/pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>` :pytorch-neuron-src:`[notebook] <pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>`


.. toctree::
   :hidden:
   
   /neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/tutorial-torchserve
   /src/examples/pytorch/pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb

