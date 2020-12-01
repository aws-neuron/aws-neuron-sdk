.. _tensorflow-resnet50:

Getting Started with TensorFlow (ResNet50)
==========================================

Steps Overview:
---------------

1. Launch an EC2 Compilation Instance (recommended instance: c5.4xlarge)
2. Install TensorFlow-Neuron and Neuron-Compiler on the Compilation
   Instance
3. Compile the compute-graph on the compilation-instance, and copy the
   artifacts into the deployment-instance
4. Install TensorFlow-Neuron and Neuron-Runtime on Deployment Instance
5. Deploy inferences inference on the Deployment Instance (Inf1)

Step 1: Launch EC2 Instance(s)
------------------------------

A typical workflow with the Neuron SDK will be to compile trained ML
models on a compilation instance and then distribute the artifacts to a
fleet of deployment instances, for execution. Neuron enables TensorFlow
to be used for all of these steps.

1.1. Select an AMI of your choice. Refer to the
:ref:`neuron-install-guide` for details.

1.2. Select and launch an EC2 instance of your choice to compile. Launch
an instance by following `EC2 instance launch
instructions <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`__.

-  It is recommended to use c5.4xlarge or larger. For this example we
   will use a c5.4xlarge.
-  Users may choose to compile and deploy on the same instance, in which
   case it is recommend to use an inf1.6xlarge instance or larger.

1.3. Select and launch a deployment (Inf1) instance of your choice.

Step 2: Compilation Instance Installations
------------------------------------------

If using Conda DLAMI version 26 and up, activate pre-installed
TensorFlow-Neuron environment (using
``source activate aws_neuron_tensorflow_p36`` command). Please update
Neuron by following update steps in :ref:`dlami-neuron-rn`.

To install in your own AMI, please see :ref:`neuron-install-guide` to
setup virtual environment and install TensorFlow-Neuron
(tensorflow-neuron) and Neuron Compiler (neuron-cc) packages.

.. _tensorflow-resnet50-compile-on-compilation-instance:

Step 3: Compile on Compilation Instance
---------------------------------------

A trained model must be compiled to Inferentia target before it can be
deployed on Inferentia instances. In this step we compile the Keras
ResNet50 model and export it as a SavedModel which is an interchange
format for TensorFlow models.

3.1. Create a python script named ``compile_resnet50.py`` with the
following content:

.. code:: python

   import os
   import time
   import shutil
   import tensorflow as tf
   import tensorflow.neuron as tfn
   import tensorflow.compat.v1.keras as keras
   from tensorflow.keras.applications.resnet50 import ResNet50
   from tensorflow.keras.applications.resnet50 import preprocess_input

   # Create a workspace
   WORKSPACE = './ws_resnet50'
   os.makedirs(WORKSPACE, exist_ok=True)

   # Prepare export directory (old one removed)
   model_dir = os.path.join(WORKSPACE, 'resnet50')
   compiled_model_dir = os.path.join(WORKSPACE, 'resnet50_neuron')
   shutil.rmtree(model_dir, ignore_errors=True)
   shutil.rmtree(compiled_model_dir, ignore_errors=True)

   # Instantiate Keras ResNet50 model
   keras.backend.set_learning_phase(0)
   keras.backend.set_image_data_format('channels_last')

   model = ResNet50(weights='imagenet')

   # Export SavedModel
   tf.saved_model.simple_save(
       session            = keras.backend.get_session(),
       export_dir         = model_dir,
       inputs             = {'input': model.inputs[0]},
       outputs            = {'output': model.outputs[0]})

   # Compile using Neuron
   tfn.saved_model.compile(model_dir, compiled_model_dir)    

   # Prepare SavedModel for uploading to Inf1 instance
   shutil.make_archive('./resnet50_neuron', 'zip', WORKSPACE, 'resnet50_neuron')

3.2. Run the compilation script, which will take a few minutes on
c5.4xlarge. At the end of script execution, the compiled SavedModel is
zipped as ``resnet50_neuron.zip`` in local directory:

.. code:: bash

   python compile_resnet50.py

::

   ...
   INFO:tensorflow:fusing subgraph neuron_op_d6f098c01c780733 with neuron-cc
   INFO:tensorflow:Number of operations in TensorFlow session: 4638
   INFO:tensorflow:Number of operations after tf.neuron optimizations: 556
   INFO:tensorflow:Number of operations placed on Neuron runtime: 554
   INFO:tensorflow:Successfully converted ./ws_resnet50/resnet50 to ./ws_resnet50/
   ...

3.3. If not compiling and inferring on the same instance, copy the
artifact to the inference server:

.. code:: bash

   scp -i <PEM key file>  ./resnet50_neuron.zip ubuntu@<instance DNS>:~/ # if Ubuntu-based AMI
   scp -i <PEM key file>  ./resnet50_neuron.zip ec2-user@<instance DNS>:~/  # if using AML2-based AMI

Step 4: Deployment Instance Installations
-----------------------------------------

**If using DLAMI, activate pre-installed TensorFlow-Neuron environment
(using \``source activate aws_neuron_tensorflow_p36`\` command) and skip
this step.**

On the instance you are going to use for inference, install
TensorFlow-Neuron and Neuron Runtime

4.1. Follow Step 2 above to install TensorFlow-Neuron.

-  Install neuron-cc if compilation on inference instance is desired
   (see notes above on recommended Inf1 sizes for compilation)
-  Skip neuron-cc if compilation is not done on inference instance

4.2. To install Neuron Runtime, see :ref:`rtd-getting-started`.

Step 5: Deploy
--------------

In this step we run inference on Inf1 using the model compiled in Step
3.

5.1. Unzip the compiled model package from Step 3, download the example
image, and install pillow module for inference:

.. code:: bash

   unzip -o resnet50_neuron.zip
   curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
   pip install pillow "h5py<3.0.0" # Necessary for loading images

5.2. On the Inf1, create a inference Python script named
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
   predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)

   # Run inference
   model_feed_dict={'input': img_arr3}
   infa_rslts = predictor_inferentia(model_feed_dict);

   # Display results
   print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])

5.3. Run the inference:

.. code:: bash

   python infer_resnet50.py

::

   [('n02123045', 'tabby', 0.6956522), ('n02127052', 'lynx', 0.120923914), ('n02123159', 'tiger_cat', 0.08831522), ('n02124075', 'Egyptian_cat', 0.06453805), ('n02128757', 'snow_leopard', 0.0087466035)]
