.. _pytorch-neuronx-main:
.. _neuron-pytorch:

PyTorch Neuron
==============

PyTorch Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and Inferentia-based Amazon EC2 instances.

PyTorch Neuron plugin architecture enables native PyTorch models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes. 
 
.. _pytorch-neuronx-training:


.. tab-set::


   .. tab-item:: Training
        :name: torch-neuronx-training-main


        .. dropdown::  Setup Guide for Trn1 
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in
                

                .. toctree::
                    :maxdepth: 1

                    Fresh install <torch-neuronx/setup/pytorch-install>
                    Update to latest release <torch-neuronx/setup/pytorch-update>
                    Install previous releases <torch-neuronx/setup/pytorch-install-prev>

                .. include:: /general/setup/install-templates/trn1-ga-warning.txt
                
        .. dropdown::  Tutorials  (``torch-neuronx``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in
                

                .. toctree::
                    :maxdepth: 1

                    /frameworks/torch/torch-neuronx/tutorials/training/bert
                    /frameworks/torch/torch-neuronx/tutorials/training/mlp
                    /frameworks/torch/torch-neuronx/tutorials/training/finetune_hftrainer
                    /frameworks/torch/torch-neuronx/tutorials/training/megatron_lm_gpt

                .. note::

                        To use Jupyter Notebook see:

                        * :ref:`setup-jupyter-notebook-steps-troubleshooting`
                        * :ref:`running-jupyter-notebook-as-script`


        .. dropdown::  Additional Examples (``torch-neuronx``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                * `AWS Neuron Reference for Megatron-LM GitHub Repository <https://github.com/aws-neuron/aws-neuron-reference-for-megatron-lm>`_
                * `AWS Neuron Samples for AWS ParallelCluster <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples>`_
                * `AWS Neuron Samples GitHub Repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training>`_

        .. dropdown::  API Reference Guide  (``torch-neuronx``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in
                

                .. toctree::
                    :maxdepth: 1

                    PyTorch Neuron neuron_parallel_compile CLI <torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile>
                    PyTorch Neuron Environment Variable <torch-neuronx/api-reference-guide/training/torch-neuron-envvars>
                    PyTorch Neuron Profiling API <torch-neuronx/api-reference-guide/torch-neuronx-profiling-api>

        .. dropdown::  Developer Guide  (``torch-neuronx``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in
                

                .. toctree::
                    :maxdepth: 1

                    torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide
                    torch-neuronx/programming-guide/training/pytorch-neuron-debug
                    torch-neuronx/programming-guide/torch-neuronx-profiling-dev-guide

        .. dropdown::  
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in
                :open:
                

                .. toctree::
                    :maxdepth: 1

                    /frameworks/torch/torch-neuronx/pytorch-neuron-supported-operators
                    /frameworks/torch/torch-neuronx/training-troubleshooting
                    /release-notes/torch/torch-neuronx/index




   .. tab-item:: Inference
        :name: torch-neuron-inference-main

        .. dropdown:: Setup Guide for Inf1
                :class-title: sphinx-design-class-title-med
                :animate: fade-in

                .. toctree::
                    :maxdepth: 1

                .. toctree::
                    :maxdepth: 1

                    Fresh install </frameworks/torch/torch-neuron/setup/pytorch-install>
                    Update to latest release </frameworks/torch/torch-neuron/setup/pytorch-update>
                    Install previous releases </frameworks/torch/torch-neuron/setup/pytorch-install-prev>
                    /frameworks/torch/torch-neuron/setup/pytorch-install-cxx11          



        .. dropdown:: Tutorials  (``torch-neuron``)
                :class-title: sphinx-design-class-title-med
                :animate: fade-in
                :name: torch-neuronx-training-tutorials

                .. tab-set::

                    .. tab-item:: Computer Vision Tutorials
                            :name: 


                            * ResNet-50 tutorial :ref:`[html] </src/examples/pytorch/resnet50.ipynb>` :pytorch-neuron-src:`[notebook] <resnet50.ipynb>`
                            * PyTorch YOLOv4 tutorial :ref:`[html] </src/examples/pytorch/yolo_v4.ipynb>` :pytorch-neuron-src:`[notebook] <yolo_v4.ipynb>`

                            .. toctree:: 
                                :hidden:
                                
                                /src/examples/pytorch/resnet50.ipynb
                                /src/examples/pytorch/yolo_v4.ipynb


                    .. tab-item:: Natural Language Processing (NLP) Tutorials
                            :name: 

 
                            * HuggingFace pretrained BERT tutorial :ref:`[html] </src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb>` :pytorch-neuron-src:`[notebook] <bert_tutorial/tutorial_pretrained_bert.ipynb>`
                            * Bring your own HuggingFace pretrained BERT container to Sagemaker Tutorial :ref:`[html] </src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>` :pytorch-neuron-src:`[notebook] <byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>`
                            * LibTorch C++ tutorial :ref:`[html] <pytorch-tutorials-libtorch>`
                            * HuggingFace MarianMT tutorial :ref:`[html] </src/examples/pytorch/transformers-marianmt.ipynb>` :pytorch-neuron-src:`[notebook] <transformers-marianmt.ipynb>`


                            .. toctree:: 
                                :hidden:
                                
                                /src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb
                                /src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb
                                /neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/tutorial-libtorch
                                /src/examples/pytorch/transformers-marianmt.ipynb

                    .. tab-item:: Utilizing Neuron Capabilities Tutorials
                            :name: 


                            * BERT TorchServe tutorial :ref:`[html] <pytorch-tutorials-torchserve>`
                            * NeuronCore Pipeline tutorial :ref:`[html] </src/examples/pytorch/pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>` :pytorch-neuron-src:`[notebook] <pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>`


                            .. toctree::
                                :hidden:
                                
                                /neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/tutorial-torchserve
                                /src/examples/pytorch/pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb


                .. note::

                        To use Jupyter Notebook see:

                        * :ref:`setup-jupyter-notebook-steps-troubleshooting`
                        * :ref:`running-jupyter-notebook-as-script`                            

        .. dropdown::  Additional Examples (``torch-neuron``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                * `AWS Neuron Samples GitHub Repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuron/inference>`_


        .. dropdown:: API Reference Guide (``torch-neuron``)
                :class-title: sphinx-design-class-title-med
                :animate: fade-in
                

                .. toctree::
                    :maxdepth: 1

                    PyTorch Neuron trace Python API </frameworks/torch/torch-neuron/api-compilation-python-api>
                    torch.neuron.DataParallel API </frameworks/torch/torch-neuron/api-torch-neuron-dataparallel-api>
                    /frameworks/torch/torch-neuron/api-core-placement

        .. dropdown:: Developer Guide (``torch-neuron``)
                :class-title: sphinx-design-class-title-med
                :animate: fade-in
                
                .. toctree::
                    :maxdepth: 1

                    Running Inference on Variable Input Shapes with Bucketing </general/appnotes/torch-neuron/bucketing-app-note>                    
                    Data Parallel Inference on PyTorch Neuron </general/appnotes/torch-neuron/torch-neuron-dataparallel-app-note>
                    /frameworks/torch/torch-neuron/guides/torch-lstm-support
                    /frameworks/torch/torch-neuron/guides/core-placement/torch-core-placement

        .. dropdown:: 
                :class-title: sphinx-design-class-title-med
                :animate: fade-in
                :open:
        

                .. toctree::
                    :maxdepth: 1

                    /release-notes/compiler/neuron-cc/neuron-cc-ops/neuron-cc-ops-pytorch
                    /frameworks/torch/torch-neuron/troubleshooting-guide
                    /release-notes/torch/torch-neuron/torch-neuron
