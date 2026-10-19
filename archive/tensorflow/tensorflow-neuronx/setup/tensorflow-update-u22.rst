
.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

.. tensorflow-neuronx-u22-update:

Update to latest TensorFlow Neuron  (``tensorflow-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This document is archived. TensorFlow is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/archive/frameworks/index`.


If you already have a previous Neuron release installed, this section provide links that will assist you to update to latest Neuron release.


.. tab-set::

    .. tab-item:: Tensorflow 2.10.1

        .. include:: /archive/frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. code-block:: bash


           # Activate Python venv 
           source aws_neuron_venv_tensorflow/bin/activate 

           # Install Jupyter notebook kernel
           pip install ipykernel 
           python3.12 -m ipykernel install --user --name aws_neuron_venv_tensorflow --display-name "Python (tensorflow-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository 
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli 
           python -m pip install wget 
           python -m pip install awscli 

           # Update Neuron Compiler and Framework
           python -m pip install --upgrade neuronx-cc==2.* tensorflow-neuronx
