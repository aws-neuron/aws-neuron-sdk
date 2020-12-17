
.. code:: bash

   pip install neuron-cc
   pip install tensorflow-neuron

Please ignore the following error displayed during installation:

.. code:: bash

   ERROR: tensorflow-serving-api 1.15.0 requires tensorflow~=1.15.0, which is not installed.

**Install TensorFlow Model Serving**

.. code:: bash

   # install TensorFlow Model Serving
   sudo apt-get install tensorflow-model-server-neuron
   pip install tensorflow_serving_api
