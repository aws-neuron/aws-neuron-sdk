# Neuron TensorFlow Serving

TensorFlow Serving is a serving system that allows customers to scale-up inference across a network. Neuron TensorFlow Serving uses the same API as normal TensorFlow Serving. The only differences are that the saved model must be compiled for Inferentia and the entry point is a different binary named `tensorflow_model_server_neuron`. The binary is found at `/usr/local/bin/tensorflow_model_server_neuron` and is pre-installed in the DLAMI or installed with APT/YUM tensorflow-model-server-neuron package.

If using DLAMI and aws_neuron_tensorflow_p36 environment, you can skip the installation step below.

## Install TensorFlow Model Server and Serving API
Follow the steps in the [Neuron installation guide](./neuron-install-guide.md).

Then ensure you install using either apt-get or yum:
```bash
sudo apt-get install tensorflow-model-server-neuron
```
or
```bash
sudo yum install tensorflow-model-server-neuron
```

Also, you would need TensorFlow Serving API (use --no-deps to prevent installation of regular tensorflow):
```bash
pip install --no-deps tensorflow_serving_api==1.15
```

## Export and Compile Saved Model

The following example shows graph construction followed by the addition of Neuron compilation step before exporting to saved model.

```python
import tensorflow as tf

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
```

After model is exported to saved model, user can serve the model by following instructions below.

## Serving Saved Model

User can now serve the saved model with the tensorflow_model_server_neuron binary:

```bash
tensorflow_model_server_neuron --model_name=resnet50_inf1 \
    --model_base_path=$(pwd)/resnet50_inf1/ --port=8500
```

The compiled model is staged in Inferentia DRAM by the server to prepare for inference.

## Generate inference requests to the model server
Now run inferences via GRPC as shown in the following sample client code:

```python
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
```
