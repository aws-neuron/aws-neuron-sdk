.. _pytorch-tutorials-marianmt:

PyTorch HuggingFace MarianMT Tutorial
=====================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

In this tutorial you will compile and deploy the `HuggingFace MarianMT <https://huggingface.co/transformers/v4.0.1/model_doc/marian.html>`_ model for sequence-to-seqeunce language translation on an Inf1 instance.

To enable faster environment setup, you will run the tutorial on an inf1.6xlarge instance to enable both compilation and deployment (inference) on the same instance.

In a production environment we encourage you to try different instance sizes to optimize to your specific deployment needs.

If you have already launched an Inf1 instance and have Neuron pytorch DLAMI environment ready, tutorial is available as a Jupyter notebook at :pytorch-neuron-src:`transformers-marianmt.ipynb <transformers-marianmt.ipynb>` and instructions can be viewed at:

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/transformers-marianmt.ipynb

Instructions of how to setup Neuron pytorch environment and run the tutorial as a Jupyter notebook are available in the next sections.

.. _pytorch-marianmt-env-setup:

Setup The Environment
---------------------

Launch an Inf1 instance by following the below steps, please make sure to choose an inf1.6xlarge instance.

.. include:: /setup/install-templates/inf1/launch-inf1-dlami.rst


.. _pytorch-marianmt-run-tutorial:

Run The Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code::

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/pytorch

The Jupyter notebook is available as a file with the name :pytorch-neuron-src:`transformers-marianmt.ipynb <transformers-marianmt.ipynb>` that you can run from browser:

* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions

You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/transformers-marianmt.ipynb


.. _marianmt-cleanup-instances:

Clean up your instance/s
------------------------

After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.