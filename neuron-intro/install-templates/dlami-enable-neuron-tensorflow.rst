To activate TensorFlow 1.x:

.. code::

  source activate aws_neuron_tensorflow_p36

To activate TensorFlow 2.x:

.. code::

  # Setup a new Python virtual environment
  sudo apt-get install python3-venv
  python3 -m venv neuron_tf2_env
  source neuron_tf2_env/bin/activate
  pip install --upgrade pip
  pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
  pip install tensorflow-neuron[cc]
  pip install ipykernel
  python -m ipykernel install --user --name neuron_tf2 --display-name "Python (Neuron TensorFlow 2)"
  pip install jupyter notebook
  pip install environment_kernels
