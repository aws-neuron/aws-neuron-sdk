To activate TensorFlow 1.x:

.. code::

  source activate aws_neuron_tensorflow_p36

To activate TensorFlow 2.x:

.. code::

  # As of DLAMI v48,  TF2.x Neuron conda environment is still not available
  # To work with TF2.x Neuron , you need to clone DLAMI TF2.x conda environment
  # and install TF2.x Neuron within the cloned environment
  # Those instructions will change when DLAMI creates a TF2.x Neuron conda environment
  conda create --name aws-neuron-tensorflow2_p37 --clone tensorflow2_p37
  source activate aws-neuron-tensorflow2_p37
  pip install --upgrade pip
  pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
  pip install tensorflow-neuron[cc]
