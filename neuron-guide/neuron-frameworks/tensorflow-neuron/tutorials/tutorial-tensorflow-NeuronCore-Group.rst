.. _tensorflow-neurocore-group:

Tutorial: Configuring NeuronCore Groups
=======================================

A NeuronCore Group is a set of NeuronCores that are used to load and run
a compiled model. At any point in time, only one model will be running
in a NeuronCore Group. Within a NeuronCore Group, loaded models can be
dynamically started and stopped, allowing for dynamic context switching
from one model to another.

By default, TensorFlow-Neuron creates NeuronCore Group size that fits a
single compiled model being loaded. For multiple models, the
NEURONCORE_GROUP_SIZES environment variable provides user control over
the multiple groups of NeuronCores. By creating several of these
NeuronCore Groups, a user is able to deploy independent models to the
hardware, all running in parallel.

To run model(s) in parallel within one process, use
NEURONCORE_GROUP_SIZES to specify the NeuronCore group sizes and
instantiate the models in sequence. This situation applies whether there
is one model or multiple models. For example, if there are 4 models A,
B, C, D compiled to 2, 4, 6, and 4 NeuronCores respectively, then you
would specify NEURONCORE_GROUP_SIZES=2,4,6,4 for the process and load
the models A, B, C, D in sequence within the process. This example
requires inf1.6xlarge with 16 NeuronCores.

To run models in multiple processes in parallel, use
NEURONCORE_GROUP_SIZES setting per process. For example, to launch two
processes with each process running a two-NeuronCore model, you would
set environment NEURONCORE_GROUP_SIZES=2 before launching each process.

More than one model can be loaded within one process into the same
NeuronCore Group(s). To achieve this, simply load the additional models
after the initial number of models has been loaded to fill the
NeuronCore Groups of the process. The additional models must have been
precompiled to fit into same NeuronCore Group size(s). During inference,
runtime would automatically context switch between models within the
same NeuronCore Group. Note that there's overhead associated with
context switching between models within the same NeuronCore Group. For
example, if there are 2 models A and B compiled to 4 and 3 NeuronCores
respectively, then you would specify NEURONCORE_GROUP_SIZES=4 for the
process and load the models A and B in sequence within the process. A
and B would be loaded into the same NeuronCore Group.

The total NEURONCORE_GROUP_SIZES across all processes should be the
number of NeuronCores visible to this TensorFlow-Neuron (which is bound
to the Neuron Runtime Daemon managing the Inferentias to be used). For
example, on an inf1.xlarge with default configurations where the total
number of NeuronCores visible to TensorFlow-Neuron is 4, you can launch
one process with NEURONCORE_GROUP_SIZES=1,1 and another process with
NEURONCORE_GROUP_SIZES=2.

In this tutorial you will learn how to enable multiple NeuronCore Groups
running the same compiled TensorFlow Resnet-50 model in parallel.

Steps Overview:
---------------

1. Launch a ``c5.4xlarge`` instance for compilation and ``inf1.6xlarge``
   for inference.
2. Install Neuron for compiler and runtime execution as shown in
   :ref:`tensorflow-resnet50`
3. Run the example up to step 5.1 as shown in
   :ref:`tensorflow-resnet50`.
4. Step 5.2 is replaced by one of the following two scenarios
   demonstrating inference using multiple NeuronCore Groups

Scenario 1: Allow one TensorFlow-Neuron or Tensorflow-Model-Server-Neuron process to run model(s) in parallel
-------------------------------------------------------------------------------------------------------------

The following example shows how to run multiple copies of ResNet50 model
in parallel on a single process. The method can be applied to running
different models in parallel on a single process. This is the
recommended method to run models in parallel as it reduces the overhead
of TensorFlow-Neuron and TensorFlow-Model-Server-Neuron resource
handling as well as Python’s global interpreter lock.

On the Inf1, create an inference Python script named
``infer_resnet50.py`` with the following content:

.. code:: python

   import os
   import numpy as np
   import tensorflow as tf
   from concurrent import futures
   from tensorflow.keras.preprocessing import image
   from tensorflow.keras.applications import resnet50

   tf.keras.backend.set_image_data_format('channels_last')

   num_model = 16
   num_image = 100

   # set NEURONCORE_GROUP_SIZES for 1-core models
   os.environ['NEURONCORE_GROUP_SIZES'] = ','.join('1' for _ in range(num_model))
   neuron_model_dir = './resnet50_neuron'

   try:
       predictor_list = [tf.contrib.predictor.from_saved_model(neuron_model_dir)
                         for _ in range(num_model)]
   except Exception as e:
       print(str(e))
       exit(1)

   # assuming model only has one input and one output
   input_name = list(predictor_list[0].feed_tensors.keys())[0]
   output_name = list(predictor_list[0].fetch_tensors.keys())[0]

   # Create input from image
   img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
   img_arr = image.img_to_array(img_sgl)
   img_arr2 = np.expand_dims(img_arr, axis=0)
   img_arr3 = resnet50.preprocess_input(img_arr2)

   model_feed_dict_list = [{input_name: img_arr3} for _ in range(num_image)]

   # submit each image to predictors in a round-robin fashion
   future_list = []
   with futures.ThreadPoolExecutor(max_workers=len(predictor_list)) as executor:
       for idx, model_feed_dict in enumerate(model_feed_dict_list):
           predictor = predictor_list[idx % len(predictor_list)]
           future_list.append(executor.submit(predictor, model_feed_dict))
       result_list = [fut.result() for fut in future_list]

   # print NEURONCORE_GROUP_SIZES setting
   print('NEURONCORE_GROUP_SIZES={}'.format(os.environ['NEURONCORE_GROUP_SIZES']))

   # print first predictions
   first_result = result_list[0]['output']
   print(resnet50.decode_predictions(first_result, top=5)[0])

   # check all remaining results
   for i in range(1, num_image):
       comp = first_result == result_list[i]['output']
       assert(all(comp.flatten()))

Run the inference:

.. code:: bash

   python infer_resnet50

.. code:: bash

   NEURONCORE_GROUP_SIZES=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
   [('n02123045', 'tabby', 0.68817204), ('n02127052', 'lynx', 0.12701613), ('n02123159', 'tiger_cat', 0.08736559), ('n02124075', 'Egyptian_cat', 0.063844085), ('n02128757', 'snow_leopard', 0.009240591)]

Scenario 2: Allowing more concurrent Tensorflow-Neuron or Tensorflow-Model-Server-Neuron processes
--------------------------------------------------------------------------------------------------

To execute concurrent processes in parallel, set environment variable
NEURONCORE_GROUP_SIZES for each process.

On the Inf1, create an inference Python script named
``infer_resnet50.py`` with the following content:

.. code:: python

   import os
   import time
   import numpy as np
   import tensorflow as tf
   from tensorflow.keras.preprocessing import image
   from tensorflow.keras.applications import resnet50

   tf.keras.backend.set_image_data_format('channels_last')

   # Create input from image
   img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
   img_arr = image.img_to_array(img_sgl)
   img_arr2 = np.expand_dims(img_arr, axis=0)
   img_arr3 = resnet50.preprocess_input(img_arr2)

   # Load model
   COMPILED_MODEL_DIR = './resnet50_neuron/'
   try:
       predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)
   except Exception as e:
       print(str(e))
       exit(1)

   # Run inference
   model_feed_dict={'input': img_arr3}
   infa_rslts = predictor_inferentia(model_feed_dict);

   # Display results
   print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])

Run 16 concurrent processes:

.. code:: bash

   # run 16 Python processes with TensorFlow-Neuron
   for i in {1..16}; do NEURONCORE_GROUP_SIZES=1 python infer_resnet50.py & done

.. code:: bash

   [('n02123045', 'tabby', 0.68817204), ('n02127052', 'lynx', 0.12701613), ('n02123159', 'tiger_cat', 0.08736559), ('n02124075', 'Egyptian_cat', 0.063844085), ('n02128757', 'snow_leopard', 0.009240591)]

   (repeats 16 times)

Scenario 3: Allowing context switching between models in the same NeuronCore Groups
-----------------------------------------------------------------------------------

To context switch between models, set environment variable
NEURONCORE_GROUP_SIZES for each process and load the models in sequence
to first fill up the NeuronCore Groups and then load additional models
in sequence into the same NeuronCore Groups. In the example below, the
NeuronCore Group size is 1 for each process and 2 models are loaded into
the same NeuronCore Group within each process.

On the Inf1, create an inference Python script named
``infer_resnet50.py`` with the following content:

.. code:: python

   import os
   import numpy as np
   import tensorflow as tf
   from concurrent import futures
   from tensorflow.keras.preprocessing import image
   from tensorflow.keras.applications import resnet50

   tf.keras.backend.set_image_data_format('channels_last')

   num_model = 2
   num_image = 100

   neuron_model_dir = './resnet50_neuron'

   try:
       predictor_list = [tf.contrib.predictor.from_saved_model(neuron_model_dir)
                         for _ in range(num_model)]
   except Exception as e:
       print(str(e))
       exit(1)

   # assuming model only has one input and one output
   input_name = list(predictor_list[0].feed_tensors.keys())[0]
   output_name = list(predictor_list[0].fetch_tensors.keys())[0]

   # Create input from image
   img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
   img_arr = image.img_to_array(img_sgl)
   img_arr2 = np.expand_dims(img_arr, axis=0)
   img_arr3 = resnet50.preprocess_input(img_arr2)

   model_feed_dict_list = [{input_name: img_arr3} for _ in range(num_image)]

   # submit each image to predictors in a round-robin fashion
   future_list = []
   with futures.ThreadPoolExecutor(max_workers=len(predictor_list)) as executor:
       for idx, model_feed_dict in enumerate(model_feed_dict_list):
           predictor = predictor_list[idx % len(predictor_list)]
           future_list.append(executor.submit(predictor, model_feed_dict))
       result_list = [fut.result() for fut in future_list]

   # print first predictions
   first_result = result_list[0]['output']
   print(resnet50.decode_predictions(first_result, top=5)[0])

   # check all remaining results
   for i in range(1, num_image):
       comp = first_result == result_list[i]['output']
       assert(all(comp.flatten()))

Run 16 concurrent processes, each loading 2 models:

.. code:: bash

   # run 16 Python processes with TensorFlow-Neuron, each process context switches between 2 models
   for i in {1..16}; do NEURONCORE_GROUP_SIZES=1 python infer_resnet50.py & done

.. code:: bash

   [('n02123045', 'tabby', 0.68817204), ('n02127052', 'lynx', 0.12701613), ('n02123159', 'tiger_cat', 0.08736559), ('n02124075', 'Egyptian_cat', 0.063844085), ('n02128757', 'snow_leopard', 0.009240591)]

   (repeats 16 times)

Troubleshooting
---------------

If you see the following message during inference:

.. code:: bash

   tensorflow.python.framework.errors_impl.ResourceExhaustedError: All machine learning accelerators are currently being consumed. Please check if there are other processes running on the accelerator. If no other processes are consuming machine learning accelerator resource, please manually free up hardware resource by `sudo systemctl restart neuron-rtd`. If you have package `aws-neuron-tools` installed, you may also free up resource by `/opt/aws/neuron/bin/neuron-cli reset`. IMPORTANT: MANUALLY FREEING UP HARDWARE RESOURCE CAN DESTROY YOUR OTHER PROCESSES RUNNING ON MACHINE LEARNING ACCELERATORS!

Please try running ``sudo systemctl restart neuron-rtd`` or
``/opt/aws/neuron/bin/neuron-cli reset`` to clean up resources. Please
note that this can destroy processing currently running on
Inferentia(s).

Also, please check the setting of NEURONCORE_GROUP_SIZES enviroment
variable.
