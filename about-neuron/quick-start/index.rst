.. meta::
   :description: Get started quickly with AWS Neuron SDK for PyTorch, JAX, and TensorFlow on Inferentia and Trainium
   :keywords: neuron, quickstart, getting started, pytorch, jax, tensorflow, inferentia, trainium, training, inference
   :instance-types: inf2, trn1, trn2, trn3
   :content-type: navigation-hub
   :date-modified: 2026-03-03

.. _neuron-quickstart:

Get Started with AWS Neuron
============================

Get up and running with AWS Neuron SDK in minutes. These quickstarts guide you through your first training or inference workload on Inferentia and Trainium instances.

.. note::
   
   **First time using AWS Neuron?** These quickstarts assume you have:
   
   - An active AWS account with EC2 access
   - Basic familiarity with your chosen ML framework (PyTorch, JAX, or TensorFlow)
   - SSH access to launch and connect to EC2 instances
   
   For detailed installation instructions, see the :doc:`Setup Guide </setup/index>`.

Choose Your Path
----------------

Select the quickstart that matches your use case:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🚀 Training Quickstart
      :link: training-quickstart
      :link-type: ref
      :class-card: sd-border-2
      
      Train your first model on Trainium
      
      - Launch a Trn1 instance
      - Run a PyTorch training script
      - Monitor training progress
      
      **Time**: ~15 minutes
      
      :bdg-primary:`Trn1` :bdg-primary:`Trn2` :bdg-primary:`Trn3`

   .. grid-item-card:: 🎯 Inference Quickstart
      :link: inference-quickstart
      :link-type: ref
      :class-card: sd-border-2
      
      Run your first inference on Inferentia
      
      - Launch an Inf2 instance
      - Load a pre-compiled model
      - Run predictions
      
      **Time**: ~10 minutes
      
      :bdg-success:`Inf2` :bdg-success:`Trn1`

Specialized Quickstarts
-----------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 💬 LLM Serving with vLLM
      :class-card: sd-border-1
      
      Deploy large language models for production inference
      
      - :doc:`Online serving </libraries/nxd-inference/vllm/quickstart-vllm-online-serving>` (OpenAI-compatible API)
      - :doc:`Offline batch inference </libraries/nxd-inference/vllm/quickstart-vllm-offline-serving>`
      
      **Time**: ~20 minutes
      
      :bdg-info:`Inf2` :bdg-info:`Trn1`

   .. grid-item-card:: 🤖 Amazon AI helper tools
      :link: amazon-q-dev
      :link-type: ref
      :class-card: sd-border-1
      
      Use AI-powered code assistance for Neuron development
      
      - Get code suggestions
      - Debug Neuron applications
      - Optimize performance
      
      **Time**: ~5 minutes

Framework-Specific Guides
-------------------------

Need framework-specific setup instructions?

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: PyTorch
      :link: /setup/pytorch/index
      :link-type: doc
      :class-card: sd-border-1
      :class-body: sphinx-design-class-title-small
      
      PyTorch 2.9+ setup

   .. grid-item-card:: JAX
      :link: /setup/jax/index
      :link-type: doc
      :class-card: sd-border-1
      :class-body: sphinx-design-class-title-small
      
      JAX 0.7+ setup

   .. grid-item-card:: TensorFlow
      :link: /archive/tensorflow/index
      :link-type: doc
      :class-card: sd-border-1
      :class-body: sphinx-design-class-title-small
      
      TensorFlow 2.x setup

Additional Resources
--------------------

- :doc:`/about-neuron/models/index` - Pre-tested model samples and tutorials
- :doc:`/devflows/ec2-flows` - Detailed EC2 deployment workflows
- :doc:`/containers/index` - Use Deep Learning Containers
- :doc:`docs-quicklinks` - Quick links to all Neuron documentation
- :doc:`github-samples` - GitHub sample repositories

Legacy Quick-Start Pages (Inf1)
--------------------------------

.. warning::
   
   The following pages are for legacy Inf1 instances only. For new projects, use the quickstarts above for Inf2, Trn1, Trn2, or Trn3.

- :doc:`torch-neuron` - PyTorch on Inf1
- :doc:`tensorflow-neuron` - TensorFlow on Inf1
- :doc:`mxnet-neuron` - MXNet on Inf1

.. toctree::
   :hidden:
   :maxdepth: 1
   
   training-quickstart
   inference-quickstart
   /libraries/nxd-inference/vllm/quickstart-vllm-online-serving
   /libraries/nxd-inference/vllm/quickstart-vllm-offline-serving
   /about-neuron/amazonq-getstarted
   docs-quicklinks
   github-samples
   torch-neuron
   tensorflow-neuron
   mxnet-neuron
