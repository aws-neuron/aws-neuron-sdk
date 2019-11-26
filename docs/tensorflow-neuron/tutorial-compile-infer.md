# Tutorial: Using Neuron to run Resnet50 inference

## Steps Overview:

1. Launch an EC2 instance for compilation and/or Infernence
2. Install Neuron for Compiler and Runtime execution
3. Run example

## Step 1: Launch EC2 Instance(s)

A typical workflow with the Neuron SDK will be to compile a trained ML model on a compilation server and then the distribute the artifacts to Inf1 instances for execution.

1.1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based. To use a pre-built Deep Learning AMI, which includes all of the needed packages, see these instructions: https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html. If you use the pre-built Deep Learning AMI, you can skip to Step 3 below.

1.2. Select and start an EC2 instance of your choice to compile
    1. It is recommended to use c5.4xlarge or larger. For this example we will use a c5.4xlarge
    2. If you would like to compile and run inference on the same machine, please select inf1.6xlarge or larger

1.3. Select and start an Inf1 instance of your choice if not compiling and inferencing on the same instance.

## Step 2: Install Neuron

### Compiler Instance: Install Neuron Compiler and TensorFlow-Neuron

On the instance you are going to use for compilation, you must have both the Neuron compiler and the Tensorflow-Neuron installed. (The inference instance must have the Tensorflow-Neuron and the Neuron Runtime installed.)

#### Using Virtualenv:

1. Install virtualenv:
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

#### Using Conda:
1. Install Conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/):
```bash
cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
echo "bfe34e1fa28d6d75a7ad05fd02fa5472275673d5f5621b77380898dee1be15d2 Miniconda3-4.7.12.1-Linux-x86_64.sh" | sha256sum --check
bash Miniconda3-4.7.12.1-Linux-x86_64.sh
...
source ~/.bashrc
```
2. Setup a new Python3.6 environment:
```bash
conda create -q -y -n test_env_p36 python=3.6
source activate test_env_p36
```

3. Modify Pip repository configurations to point to the Neuron repository.
```bash
tee $CONDA_PREFIX/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF
```

4. Install TensorFlow-Neuron and Neuron Compiler
```bash
pip install neuron-cc[tensorflow]
pip install tensorflow-neuron
```

### Inference Instance: Install Tensorflow-Neuron and Neuron-Runtime

1. Same as above to install Tensorflow Neuron.
2. To install Runtime, refer to the [getting started](./../neuron-runtime/readme.md) runtime guide.

## Step 3: Run Inference

Steps Overview:
1. Compile the Keras ResNet50 model and export it as a SavedModel which is an interchange format for TensorFlow models.
2. Run inference on Inf1 with an example input.

3.1. Create a python script named `compile_resnet50.py` with the following content:
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
3.2. Run the compilation script, which will take a few minutes on c5.4xlarge. At the end of script execution, the compiled SavedModel is zipped as `resnet50_neuron.zip` in local directory:
```bash
python compile_resnet50.py
```
```
 ...
INFO:tensorflow:fusing subgraph neuron_op_d6f098c01c780733 with neuron-cc; log file is at /home/ubuntu/ws_resnet50/workdir/neuron_op_d6f098c01c780733/graph_def.neuron-cc.log
INFO:tensorflow:Number of operations in TensorFlow session: 3978
INFO:tensorflow:Number of operations after tf.neuron optimizations: 555
INFO:tensorflow:Number of operations placed on Neuron runtime: 554
...
```

3.3. If not compiling and inferring on the same instance, copy the artifact to the inference server:
```bash
scp -i <PEM key file>  ./resnet50_neuron.zip ubuntu@<instance DNS>:~/ # Ubuntu
scp -i <PEM key file>  ./resnet50_neuron.zip ec2-user@<instance DNS>:~/  # AML2
```
3.4. On the Inf1, create a inference Python script named `infer_resnet50.py` with the following content:
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
