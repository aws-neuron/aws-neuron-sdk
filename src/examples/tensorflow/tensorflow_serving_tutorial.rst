.. _tensorflow-serving-neuronrt-visible-cores:

Using NEURON_RT_VISIBLE_CORES with TensorFlow Serving
=====================================================

TensorFlow serving allows customers to scale-up inference workloads
across a network. TensorFlow Neuron Serving uses the same API as normal
TensorFlow Serving with two differences: (a) the saved model must be
compiled for Inferentia and (b) the entry point is a different binary
named ``tensorflow_model_server_neuron``. Follow the steps below 
to install the package using apt-get or yum. This will be pre-installed in a future relase.

Install TensorFlow Model Server and Serving API
-----------------------------------------------

Follow the steps in the :ref:`install-neuron-tensorflow`.

Then ensure you install using either apt-get or yum.
If using TF 1.x, install the appropriate version (see above).:

.. code:: bash

   sudo apt-get install tensorflow-model-server-neuron

or

.. code:: bash

   sudo yum install tensorflow-model-server-neuron

Also, you would need TensorFlow Serving API (use --no-deps to prevent
installation of regular tensorflow). Depending on the version of Tensorflow
you wish to use:

For Tensorflow 1.x:

.. code:: bash

   pip install --no-deps tensorflow_serving_api==1.15

For Tensorflow 2.x:

.. code:: bash

   pip install --no-deps tensorflow_serving_api

For the example image preprocessing using Keras preprocessing, the
Python Imaging Library Pillow is required:

.. code:: bash

   pip install pillow

To workaround h5py issue https://github.com/aws/aws-neuron-sdk/issues/220:

.. code:: bash

   pip install "h5py<3.0.0"


Export and Compile Saved Model
------------------------------

The following example shows graph construction followed by the addition
of Neuron compilation step before exporting to saved model.

For Tensorflow 1.x:

.. code:: python

   import tensorflow as tf
   import tensorflow.neuron

   tf.keras.backend.set_learning_phase(0)
   tf.keras.backend.set_image_data_format('channels_last')
   model = tf.keras.applications.ResNet50(weights='imagenet')
   sess = tf.keras.backend.get_session()
   inputs = {'input': model.inputs[0]}
   outputs = {'output': model.outputs[0]}

   # save the model using tf.saved_model.simple_save
   modeldir = "./resnet50/1"
   tf.saved_model.simple_save(sess, modeldir, inputs, outputs)

   # compile the model for Inferentia
   neuron_modeldir = "./resnet50_inf1/1"
   tf.neuron.saved_model.compile(modeldir, neuron_modeldir, batch_size=1)

For Tensorflow 2.x:

.. code:: python

    import tensorflow as tf
    import tensorflow.neuron as tfn
    import numpy as np

    tf.keras.backend.set_learning_phase(0)
    tf.keras.backend.set_image_data_format('channels_last')
    image_sizes = [224, 224]
    model = tf.keras.applications.ResNet50(weights='imagenet')
    example_inputs = tf.random.uniform([1, *image_sizes, 3], dtype=tf.float32)

    # run the model once to define the forward pass and allow for saving
    model_neuron(example_inputs)
    model_neuron = tfn.trace(model, example_inputs)
    tf.keras.models.save_model(model_neuron, './resnet50_inf1/1')



Serving Saved Model
-------------------

User can now serve the saved model with the
tensorflow_model_server_neuron binary. To utilize multiple NeuronCores,
it is recommended to launch multiple tensorflow model servers that
listen to the same gRPC port:

.. code:: bash

   export NEURON_RT_VISIBLE_CORES=0  # important to set this environment variable before launching model servers
   tensorflow_model_server_neuron --model_name=resnet50_inf1 \
        --model_base_path=$(pwd)/resnet50_inf1/ --port=8500

   #then to run another server on a different neuron core open another
   #window and run this, except this time set NEURON_RT_VISIBLE_CORES=1
   #you can keep doing this up to the number of Neuron Cores on your machine

   export NEURON_RT_VISIBLE_CORES=1
   tensorflow_model_server_neuron --model_name=resnet50_inf1 \
        --model_base_path=$(pwd)/resnet50_inf1/ --port=8500

The compiled model is staged in Inferentia DRAM by the server to prepare
for inference.

Generate inference requests to the model server
-----------------------------------------------

Now run inferences via GRPC as shown in the following sample client
code:

For Tensorflow 1.x:

.. code:: python

  import numpy as np
  import grpc
  import tensorflow as tf
  from tensorflow.keras.preprocessing import image
  from tensorflow.keras.applications.resnet50 import preprocess_input
  from tensorflow.keras.applications.resnet50 import decode_predictions
  from tensorflow_serving.apis import predict_pb2
  from tensorflow_serving.apis import prediction_service_pb2_grpc

  if __name__ == '__main__':
      channel = grpc.insecure_channel('localhost:8500')
      stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
      img_file = tf.keras.utils.get_file(
          "./kitten_small.jpg",
          "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg")
      img = image.load_img(img_file, target_size=(224, 224))
      img_array = preprocess_input(image.img_to_array(img)[None, ...])
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'resnet50_inf1'
      request.inputs['input'].CopyFrom(
          tf.contrib.util.make_tensor_proto(img_array, shape=img_array.shape))
      result = stub.Predict(request)
      prediction = tf.make_ndarray(result.outputs['output'])
      print(decode_predictions(prediction))

For Tensorflow 2.x:

.. code:: python

    import numpy as np
    import grpc
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2_grpc
    from tensorflow.keras.applications.resnet50 import decode_predictions

    tf.keras.backend.set_image_data_format('channels_last')

    if __name__ == '__main__':
        channel = grpc.insecure_channel('localhost:8500')
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        img_file = tf.keras.utils.get_file(
            "./kitten_small.jpg",
            "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg")
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = preprocess_input(image.img_to_array(img)[None, ...])
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'resnet50_inf1'
        request.inputs['input_1'].CopyFrom(
            tf.make_tensor_proto(img_array, shape=img_array.shape))
        result = stub.Predict(request)
        prediction = tf.make_ndarray(result.outputs['output_1'])
        print(decode_predictions(prediction))
