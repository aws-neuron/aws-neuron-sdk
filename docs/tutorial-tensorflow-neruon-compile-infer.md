# Tutorial: Using TensorFlow-Neuron and the Neuron Compiler with Resnet50 

## Steps Overview:

1. Launch an EC2 instance for compilation  and/or Infernence
2. Install Neuron for Compiler and Runtime execution
3. Run example 

## Step 1: Launch EC2 Instance(s)

A typical workflow with the Neuron SDK will be for a previously trained ML model to be compiled on a compilation server and then the artifacts distributed to the (fleet of) inf1 instances for execution. Neuron enables Tensorflow to be used for all of these steps.

Steps Overview: 

1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based. To use a pre-built Deep Learning AMI, which includes all of the needed packages, see these instructions: https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html
2. Select and start an EC2 instance of your choice (see https://aws.amazon.com/ec2/instance-types/) to compile
    1. It is recommended to use C5.4xlarge or larger. For this example we will use a C5.4xlarge
    2. If you would like to compile and infer on the same machine, please select Inf1.6xlarge
3. Select and start an INF1 instance of your choice if not compiling and inferencing on  the same instance;
    1. see [LINK]

## Step 2: Install Neuron

### Compiler Instance: Install Neuron Compiler and TensorFlow-Neuron 

On the instance you are going to use for compilation, you must have both the Neuron Compiler and the Tensorflow-Neuron installed. (The inference instance must have the Tensorflow-Neuron and the Neuron Runtime installed.)

Steps Overview:

1. Modify pip repository configurations to point to the Neuron repository.
```
    sudo tee /etc/pip.conf > /dev/null <<EOF
        [global]
        extra-index-url = https://pip.repos.neuron.amazonaws.com
        EOF
 ```
2. Setup a new Python 3.6 environment with either Virtualenv or Conda:
 ``` 
#Example setup for virtualenv:
 sudo apt-get -y install virtualenv
 virtualenv --python=python3.6 test_env_p36
 source test_env_p36/bin/activate
        
 #Example setup for conda environment:
 conda create -q -y -n test_env_p36 python=3.6
 source activate test_env_p36
```
3. Install TensorFlow-Neuron and Neuron Compiler
```
 pip install neuron-cc
 pip install tensorflow-neuron
```

### Inference Instance: Install Tensorflow-Neuron and Neuron-Runtime

1. same as above to install Tensorflow Neuron
2. to install Runtime, see [This link](./getting-started-neuron-rtd.md)

## Step 3: Example

In this example, we compile the Keras ResNet50 model and export it as a SavedModel which is an interchange format for TensorFlow models. Then we run inference on Inf1 with an example input.


1. Create a python script named compile_resnet50.py with the following content:

```
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
modeldir = os.path.join(WORKSPACE, 'resnet50_neuron')
shutil.rmtree(modeldir, ignore_errors=True)

# Instantiate Keras ResNet50 model 
keras.backend.set_learning_phase(0)
model = ResNet50(weights='imagenet')

# Compile and export SavedModel
tfn.saved_model.simple_save(
    sess               = keras.backend.get_session(),
    export_dir         = modeldir,
    inputs             = {'input': model.inputs[0]},
    outputs            = {'output': model.outputs[0]},
    batch_size         = 1)

# Prepare SavedModel for uploading to Inf1 instance
shutil.make_archive(modeldir, 'zip', WORKSPACE, 'resnet50_neuron')
```

2. Run the compilation. The SavedModel is zipped at `ws_resnet50/resnet50_neuron.zip`:

```
> python compile_resnet50.py

...
INFO:tensorflow:fusing subgraph neuron_op_d6f098c01c780733 with neuron-cc; log file is at /home/ubuntu/ws_resnet50/workdir/neuron_op_d6f098c01c780733/graph_def.neuron-cc.log
INFO:tensorflow:Number of operations in TensorFlow session: 3978
INFO:tensorflow:Number of operations after tf.neuron optimizations: 555
INFO:tensorflow:Number of operations placed on Neuron runtime: 554
...
```

3. If not compiling and inferring on the same instance, copy the artifact to the inference server:

```
scp -i <PEM key file>  ws_resnet50/resnet50_neuron.zip ubuntu@<instance DNS>:~/  # Ubuntu
scp -i <PEM key file>  ws_resnet50/resnet50_neuron.zip ec2-user@<instance DNS>:~/  # AML2
```

4. On the Inf1, create a inference Python script named infer_resnet50.py with the following content:

```
import os
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
predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)

# Run inference
model_feed_dict={'input': img_arr3}
infa_rslts = predictor_inferentia(model_feed_dict);

# Display results
print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])
```

5. Unzip the mode, download the example image and run the inference:

```
`>`` unzip resnet50_neuron.zip
> curl ``-``O https``:``//raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg`
`>`` python run_inference``.``py``

[('n02123045', 'tabby', 0.6956522), ('n02127052', 'lynx', 0.120923914), ('n02123159', 'tiger_cat', 0.08831522), ('n02124075', 'Egyptian_cat', 0.06453805), ('n02128757', 'snow_leopard', 0.0087466035)]`
```


