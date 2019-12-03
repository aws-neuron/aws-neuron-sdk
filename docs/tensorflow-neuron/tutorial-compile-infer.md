# Tutorial: Getting Started with TensorFlow-Neuron (ResNet-50 Tutorial)

## Steps Overview:

1. Launch an EC2 compilation-instance (recommended instance: c5.4xl), and an deployment instance
2. Install Neuron compiler on the compilation-instace
3. Compile the compute-graph on the compilation-instance, and copy the compilation artifacts to the deployment-instance
4. Deploy: run inferences on the deployment-instance

## Step 1: Launch EC2 Instance(s)

A typical workflow with the Neuron SDK will be to compile a trained ML model on a compilation-instance and then to distribute the artifacts to Inf1 deployment-instances, for execution.

1.1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, or Amazon Linux 2 based. A pre-built Deep Learning AMI is also available, which includes all of the needed packages (see https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html).

**Note:** If you choose to use Deep Learning AMI (recommended for getting started), you may skip to Step 3 below. Make sure to activate the environment of choice, e.g.
```bash
ubuntu:~$ source activate aws_neuron_tensorflow_p36
(aws_neuron_tensorflow_p36) ubuntu:~$ 
```

1.2. Select and start an EC2 compilation-instance
    1. For this example, we will use c5.4xlarge
    2. Users may choose the compile and deploy on the same instance, in which case we recommend using Inf1.6xlarge or larger

1.3. Select and start a deployment-instance of your choice (if not compiling and inferencing on the same instance).
    1. For this example, we will use Inf1.xl

## Step 2: Installations

### Compilation-Instance: Install Neuron Compiler and TensorFlow-Neuron

Note: this step is only required if you are not using Deep Learning AMI.

On the compilation-instance, install Neuron-compiler and Tensorflow-Neuron.
On the deployment-instance, install Neuron-runtime and Tensorflow-Neuron.

#### Using Virtualenv:

1. Install virtualenv if needed:
```bash
sudo apt-get update
sudo apt-get -y install virtualenv
```
2. Setup a new Python 3.6 environment:
```bash
virtualenv --python=python3.6 test_env_p36
source test_env_p36/bin/activate
```
3. Modify Pip repository configurations to point to the Neuron repository.
```bash
tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF
```
4. Install TensorFlow-Neuron and Neuron Compiler
```bash
pip install neuron-cc[tensorflow]
pip install tensorflow-neuron
```

### Deployment-Instance: Install Tensorflow-Neuron and Neuron-Runtime

1. Same as above to install Tensorflow Neuron.
2. To install Runtime, refer to the [getting started](./../neuron-runtime/nrt_start.md) runtime guide.

## Step 3: Compile

3.1. Create a python script named `compile_resnet50.py` to compile ResNet-50. Example given below:
```python
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
```

3.2. Run the compilation script (can take a few minutes). At the end of script execution, the compiled SavedModel is zipped as `resnet50_neuron.zip` in local directory:
```bash
python compile_resnet50.py
```
```
...
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
INFO:tensorflow:fusing subgraph neuron_op_d6f098c01c780733 with neuron-cc
INFO:tensorflow:Number of operations in TensorFlow session: 4638
INFO:tensorflow:Number of operations after tf.neuron optimizations: 556
INFO:tensorflow:Number of operations placed on Neuron runtime: 554
INFO:tensorflow:Successfully converted ./ws_resnet50/resnet50 to ./ws_resnet50/resnet50_neuron
...
```

3.3. If not compiling and inferring on the same instance, copy the artifact to the inference server:
```bash
scp -i <PEM key file>  ./resnet50_neuron.zip   ubuntu@<instance DNS>:~/ # is using Ubuntu-based AMI
scp -i <PEM key file>  ./resnet50_neuron.zip   ec2-user@<instance DNS>:~/  # if using AML2-based AMI
```

## Step 4: Deploy

4.1. On the deployment-instance (Inf1), create an inference Python script named `infer_resnet50.py` with the following content:
```python
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

4.2. Unzip the model, download the example image and run the inference:
```bash
unzip resnet50_neuron.zip
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
pip install pillow # Necessary for loading images
python infer_resnet50.py
```
```
[('n02123045', 'tabby', 0.6956522), ('n02127052', 'lynx', 0.120923914), ('n02123159', 'tiger_cat', 0.08831522), ('n02124075', 'Egyptian_cat', 0.06453805), ('n02128757', 'snow_leopard', 0.0087466035)]
```
