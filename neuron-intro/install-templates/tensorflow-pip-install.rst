**To install TensorFlow 1.x**

.. code:: bash

   pip install neuron-cc
   pip install tensorflow-neuron~=1.15.0

Please ignore the following error displayed during installation:

.. code:: bash

   ERROR: tensorflow-serving-api 1.15.0 requires tensorflow~=1.15.0, which is not installed.

**To install TensorFlow 2.x**

.. code:: bash

    pip install tensorflow-neuron[cc]

**Then install TensorFlow Model Serving (valid for both TensorFlow 1.x and TensorFlow 2.x)**

.. code:: bash

   # install TensorFlow Model Serving
   sudo apt-get install tensorflow-model-server-neuron
   pip install tensorflow_serving_api
