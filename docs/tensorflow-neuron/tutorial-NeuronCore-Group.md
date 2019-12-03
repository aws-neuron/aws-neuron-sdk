# Tutorial: Configuring NeuronCore Groups

A NeuronCore Group is a set of NeuronCores that are used to load and run compiled models. At any time, one model will be running in a NeuronCore Group. By changing to a different sized NeuronCore Group and then creating several of these NeuronCore Groups, a user may create independent and parallel models running in the Inferentia. Additionally, within a NeuronCore Group, loaded models can be dynamically started and stopped, allowing for dynamic context switching from one model to another. By default, a single NeuronCoreGroup is created by Neuron Runtime that contains all four NeuronCores in an Inferentia. In this default case, when models are loaded to that default NeuronCore Group, only one will be running at any time. By configuring multiple NeuronCore Groups as shown in this tutorial, multiple models may be made to run simultaneously.

The NEURONCORE_GROUP_SIZES environment variable provides user control over the grouping of NeuronCores in Neuron-integrated TensorFlow. By default, TensorFlow-Neuron will choose the optimal utilization mode based on model metadata, but in some cases manually setting NEURONCORE_GROUP_SIZES can provide additional performance benefits.

In this tutorial you will learn how to enable a NeuronCore Group running TensorFlow Resnet-50 model.

## Steps Overview:

1. Launch an EC2 instance for compilation and/or Inference
2. Install Neuron for Compiler and Runtime execution as shown in  [tensorflow-neuron tutorial](./tutorial-compile-infer.md)
3. Run examples

## Example 1
These steps are the same as described in [tensorflow-neuron tutorial](./tutorial-compile-infer.md). The final step to create the inference script - step 3.4 is replaced with this:

3.4. On the Inf1, create an inference Python script named `infer_resnet50.py` with the following content:
```python
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(img_arr2)

# Load model
COMPILED_MODEL_DIR = './resnet50_neuron/'
num_parallel = 4
predictor_list = [tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR) for _ in range(num_parallel)]

# Run inference
model_feed_dict={'input': img_arr3}
with ThreadPoolExecutor(max_workers=num_parallel) as executor:
    future_list = [executor.submit(pred, {'input': img_arr3}) for pred in predictor_list]
    infa_rslts_list = [future.result() for future in future_list]

# Display results
for infa_rslts in infa_rslts_list:
    print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])
```

3.5. Unzip the mode, download the example image and run the inference:
```bash
unzip resnet50_neuron.zip
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
pip install pillow # Necessary for loading images
python infer_resnet50.py
```
```
[('n02123045', 'tabby', 0.6956522), ('n02127052', 'lynx', 0.120923914), ('n02123159', 'tiger_cat', 0.08831522), ('n02124075', 'Egyptian_cat', 0.06453805), ('n02128757', 'snow_leopard', 0.0087466035)]
```

## Example 2:

Scenario 1: allow tensorflow-neuron to utilize more than one Inferentia on inf1.6xlarge and inf1.24xlarge instance sizes.

By default, one Python process with tensorflow-neuron or one tensorflow_model_server_neuron process tries to allocate all NeuronCores in an Inferentia from the Neuron Runtime Daemon. To utilize multiple Inferentias, the recommended parallelization mode is process-level parallelization, as it bypasses the overhead of Python and tensorflow_model_server_neuron resource handling as well as Python’s global interpreter lock (GIL). Note that TensorFlow’s session.run function actually does not hold the GIL.

When there is a need to allocate more Inferentia compute into a single process, the following example shows the usage:

```python
# Run 16 ResNet50 models on inf1.6xlarge
import os
from concurrent import futures
import numpy as np
import tensorflow as tf

num_model = 16
num_image = 5000
os.environ['NEURONCORE_GROUP_SIZES'] = ','.join('1' for _ in range(num_model))
neuron_model_dir = './resnet50_neuron'
predictor_list = [tf.contrib.predictor.from_saved_model(neuron_model_dir)
                  for _ in range(num_model)]

# assuming model only has one input and one output
input_name = list(predictor_list[0].feed_tensors.keys())[0]
output_name = list(predictor_list[0].fetch_tensors.keys())[0]

# using random numbers as inputs... please plugin your dataset accordingly
model_feed_dict_list = [{input_name: np.random.rand(1, 224, 224, 3)}
                        for _ in range(num_image)]

# submit each image to predictors in a round-robin fashion
future_list = []
with futures.ThreadPoolExecutor(max_workers=len(predictor_list)) as executor:
    for idx, model_feed_dict in enumerate(model_feed_dict_list):
        predictor = predictor_list[idx // len(predictor_list)]
        future_list.append(executor.submit(predictor, model_feed_dict))
    result_list = [fut.result() for fut in future_list]

# inference results should be in result_list already

```
## Example 3
Scenario 2: allowing more concurrent tensorflow-neuron or tensorflow_model_server_neuron processes

As we mentioned in Scenario 1, each tensorflow-neuron/tensorflow_model_server_neuron process tries to allocate the equivalence compute power of one full Inferentia from the Neuron Runtime Daemon by default. This puts a limitation on the number of tensorflow-neuron/tensorflow_model_server_neuron processes we can run on an Inf1 instance. For example, on inf1.6xlarge, the default setting allows at most 4 tensorflow-neuron/tensorflow_model_server_neuron processes to run concurrently.

To circumvent this, we can set NEURONCORE_GROUP_SIZES=1, and execute 16 concurrent tensorflow-neuron processes, as demonstrated in the following example

```bash
# infer16.sh -- run 16 Python processes with TensorFlow-Neuron
export NEURONCORE_GROUP_SIZES=1
for i in {1..16}; do python infer.py & done
```

```python
# infer.py -- one Python process with TensorFlow-Neuron
import os
import numpy as np
import tensorflow as tf

num_image = 5000
neuron_model_dir = './resnet50_neuron'
predictor = tf.contrib.predictor.from_saved_model(neuron_model_dir)

# assuming model only has one input and one output
input_name = list(predictor_list[0].feed_tensors.keys())[0]
output_name = list(predictor_list[0].fetch_tensors.keys())[0]

# using random numbers as inputs... please plugin your dataset accordingly
model_feed_dict_list = [{input_name: np.random.rand(1, 224, 224, 3)}
                        for _ in range(num_image)]

result_list = [predictor(feed) for feed in model_feed_dict_list]

# inference results can be found in result_list
```
