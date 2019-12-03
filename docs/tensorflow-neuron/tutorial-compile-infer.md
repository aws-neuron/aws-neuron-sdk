# Tutorial: Getting Started with TensorFlow-Neuron (ResNet-50 Tutorial)

## Steps Overview:

1. Launch an EC2 compilation-instance (recommended instance c5.4xlarge), and a deployment instance (recommended inf1.xlarge)
2. Install neuron-sdk on the launched instances
3. Compile the compute-graph on the compilation-instance, and copy the compilation artifacts into the deployment-instance
4. Deploy: run inferences on the deployment-instance

## Step 1: Launch EC2 Instance(s)

A typical workflow with the Neuron SDK will be to compile a trained ML model on a compilation-instance and then to distribute the artifacts to Inf1 deployment-instances, for execution. Neuron enables TensorFlow to be used for all of these steps.

1.1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based.  A pre-built Deep Learning AMI is also available, which includes all of the needed packages, see [Launching and Configuring a DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html)

1.2. Select and launch an EC2 compilation-instance. Launch an instance by following [EC2 instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance).
  1.2.1. It is recommended to use c5.4xlarge or larger. For this example we will use a c5.4xlarge.
  1.2.2. Users may choose the compile and deploy on the same instance, in which case it is recommended to use inf1.6xlarge.
  
1.3. Select and launch a deployment-instance (if not compiling and deploying on the same instance). Launch an instance by following [EC2 instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance).


**Note:** If you choose to use Deep Learning AMI (recommended for getting started), you may skip to Step 3 below. Make sure to activate the environment of choice, e.g.



## Step 2: Compilation-Instance Installations

**If using DLAMI, activate aws_neuron_tensorflow_p36 environment and skip this step.**

On the compilation-instance, install both Neuron Compiler and TensorFlow-Neuron.
2.1. Install virtualenv if needed:

```bash
# Ubuntu
sudo apt-get update
sudo apt-get -y install virtualenv
```

```bash
# Amazon Linux 2
sudo yum update
sudo yum install -y python3
pip3 install --user virtualenv
```

2.2. Setup a new Python 3.6 environment:


```bash
virtualenv --python=python3.6 test_env_p36source test_env_p36/bin/activate
```

2.3. Modify Pip repository configurations to point to the Neuron repository.


```bash
tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF[global]extra-index-url = https://pip.repos.neuron.amazonaws.comEOF
```

2.4. Install TensorFlow-Neuron and Neuron Compiler


```bash
pip install tensorflow-neuron
```



```bash
# can be skipped on inference-only instance
pip install neuron-cc[tensorflow]
```

## Step 3: Compile


3.1. Create a python script named `compile_resnet50.py` with the following content:


```python
import osimport timeimport shutilimport tensorflow as tfimport tensorflow.neuron as tfnimport tensorflow.compat.v1.keras as kerasfrom tensorflow.keras.applications.resnet50 import ResNet50from tensorflow.keras.applications.resnet50 import preprocess_input

 # Create a workspaceWORKSPACE = './ws_resnet50'
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


```python
python compile_resnet50.py
`...
INFO:tensorflow:fusing subgraph neuron_op_d6f098c01c780733 with neuron-cc
`INFO:tensorflow:Number of operations in TensorFlow session: 4638
INFO:tensorflow:Number of operations after tf.neuron optimizations: 556
INFO:tensorflow:Number of operations placed on Neuron runtime: 554
INFO:tensorflow:Successfully converted ./ws_resnet50/resnet50 to ./ws_resnet50/resnet50_neuron`
...`
```

3.3. If not compiling and deploying on the same instance, copy the artifact to the inference server:


```bash
scp -i <PEM key file> ./resnet50_neuron.zip ubuntu@<instance DNS>:~/ # if using Ubuntu-based AMIscp -i <PEM key file> ./resnet50_neuron
.zip ec2-user@<instance DNS>:~/ # if using AML2-based AMI
```

## Step 4: Deployment-Instance Installations

**If using DLAMI, activate aws_neuron_tensorflow_p36 environment and skip this step.**

4.1. Follow Step 2 above to install TensorFlow-Neuron.

* Install neuron-cc if compilation on inference instance is desired (see notes above on recommended Inf1 sizes for compilation)
* Skip neuron-cc if compilation is not done on inference instance

4.2. To install Runtime, see [Getting started: Installing and Configuring Neuron-RTD](https://github.com/aws/aws-neuron-sdk/blob/bec33c18a05f794620c59d51991ec5a9bf99a7ab/docs/neuron-runtime/nrt_start.md).


## Step 5: Execute inference on Inf1

5.1. On the deployment-instance, create a inference Python script named `infer_resnet50.py` with the following content:


```python
import osimport timeimport numpy as npimport tensorflow as tffrom tensorflow.keras.preprocessing import imagefrom tensorflow.keras.applications import resnet50
# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(img_arr2)
# Load modelCOMPILED_MODEL_DIR = './resnet50_neuron/'
predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)
# Run inference
model_feed_dict={'input': img_arr3}
infa_rslts = predictor_inferentia(model_feed_dict);# Display resultsprint(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])
```

5.2. Unzip the mode, download the example image and run the inference:


```bash
unzip resnet50_neuron.zip
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
pip install pillow # Necessary for loading images
python infer_resnet50.py
`[('n02123045', 'tabby', 0.6956522), ('n02127052', 'lynx', 0.120923914), ('n02123159', 'tiger_cat', 0.08831522), ('n021240`
```

