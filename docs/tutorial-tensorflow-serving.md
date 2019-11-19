# Neuron TensorFlow Serving

TensorFlow Serving is a serving system that allows customers to scale-up inference across a network. Neuron TensorFlow Serving uses the same API as normal TensorFlow Serving. The only differences are that the saved model must be compiled for Inferentia and the entry point is a different binary named `tensorflow_model_server_neuron`. The binary is found at `/usr/local/bin/tensorflow_model_server_neuron` and is pre-installed in the DL AMI or installed with APT/YUM tensorflow-model-server-neuron package (see installation instruction here [link]).

## Export and Compile Saved Model

The following example shows graph construction followed by the addition of Neuron compilation step before exporting to saved model.

```python
import tensorflow as tf

tf.keras.backend.set_learning_phase(0)
model = tf.keras.applications.ResNet50(weights='imagenet')
sess = tf.keras.backend.get_session()
inputs = {'input': model.inputs[0]}
outputs = {'output': model.outputs[0]}

# save the model using tf.saved_model.simple_save
modeldir = "./resnet50/1"
tf.saved_model.simple_save(sess, modeldir, inputs, outputs)

# compile the model for Inferentia
neuron_modeldir = "/home/ubuntu/resnet50_inf1/1"
tf.neuron.saved_model.compile(modeldir, neuron_modeldir, batch_size=1)
```

After model is exported to saved model, user can serve the model by following instructions below.

## Serving Saved Model

User can now serve the saved model with the tensorflow_model_server_neuron binary:

```bash
tensorflow_model_server_neuron --model_name=resnet50_inf1 \
    --model_base_path=/home/ubuntu/resnet50_inf1/ --port=8500
```

The compiled model is staged in Inferentia DRAM by the server to prepare for inference. If the saved model is uncompiled (export first flow), the server would call Neuron compiler to do the compilation before staging.

If the saved model was precompiled contains a graph with compiled NeuronOps and therefore must be served using the model server that is compatible with the TensorFlow generating the Saved Model.

Now, user can run inferences via GRPC from the following client code:

```python
import numpy as np
import grpc
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
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
        tf.contrib.util.make_tensor_proto(img_array, img_array=data.shape))
    result = stub.Predict(request)
    prediction = tf.make_ndarray(result.outputs['output'])
    print(decode_predictions(prediction))
```