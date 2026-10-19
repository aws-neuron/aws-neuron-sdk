.. _tensorflow-tutorial-setup:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

TensorFlow Tutorial Setup
=========================

.. warning::

   This document is archived. TensorFlow is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/archive/frameworks/index`.


#. Launch an Inf1.6xlarge Instance:
    .. include:: /setup/install-templates/inf1/launch-inf1-dlami.rst

#. Set up a development environment:
    * Enable or install TensorFlow-Neuron: :ref:`install-neuron-tensorflow`.
    
#. Run tutorial in Jupyter notebook:
    * Follow instruction at :ref:`Setup Jupyter notebook <setup-jupyter-notebook-steps-troubleshooting>` to:
    
      #. Start the Jupyter Notebook on the instance
      #. Run the Jupyter Notebook from your local browser

    * Connect to the instance from the terminal, clone the Neuron Github repository to the Inf1 instance and then change the working directory to the tutorial directory:

      .. code::

        git clone https://github.com/aws/aws-neuron-sdk.git
        cd aws-neuron-sdk/src/examples/tensorflow

    * Locate the tutorial notebook file (.ipynb file) under ``aws-neuron-sdk/src/examples/tensorflow``
    * From your local browser, open the tutorial notebook from the menu and follow the instructions.

    
