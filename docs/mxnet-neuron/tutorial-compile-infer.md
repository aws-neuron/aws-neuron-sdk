# Tutorial: Using MXNet-Neuron

Neuron supports both Python module and Symbol APIs and the C predict API. The following quick start example uses the Symbol API.

## Steps Overview:

1. Launch an EC2 instance for compilation and/or inference
2. Install Neuron for Compiler and Runtime execution
3. Run Example
    1. Compile
    2. Execute inference on Inf1

## Step 1: Launch EC2 Instances

A typical workflow with the Neuron SDK will be for a trained ML model to be compiled on a compilation server and then the artifacts distributed to the (fleet of) Inf1 instances for execution. Neuron enables MXNet to be used for all of these steps.


1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based. To use a pre-built Deep Learning AMI, which includes all of the needed packages, see these instructions: https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html
2. Select and start an EC2 instance of your choice to compile
    1. It is recommended to use C5.4xlarge or larger. For this example we will use a C5.4xlarge
    2. If you would like to compile and infer on the same machine, please select Inf1.6xlarge
3. Select and start an Inf1 instance of your choice to run the compiled model you started in step 2.2 to run the compiled model.

## Step 2: Install Neuron

If using DLAMI and aws_neuron_mxnet_p36 environment, you can skip to Step 3.

### Compiler Instance: Install Neuron Compiler and MXnet-Neuron

On the instance you are going to use for compilation, you must have both the Neuron Compiler and the MXNet-Neuron installed. (The inference instance must have the MXNet-Neuron and the Neuron Runtime installed.)
Steps Overview:

#### Using Virtualenv:

1. Install virtualenv if needed:
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
4. Install MxNet-Neuron and Neuron Compiler
```bash
pip install neuron-cc[mxnet]
pip install mxnet-neuron
```

### Inference Instance: Install MXNet-Neuron and Neuron-Runtime

1. Same as above to install MXNet-Neuron
2. To install Runtime, see [Getting started: Installing and Configuring Neuron-RTD](./../neuron-runtime/nrt_start.md).

## Step 3: Run Example

1. Create a file `compile_resnet50.py` with the content below and run it using `python compile_resnet50.py`. Compilation will take a few minutes on c5.4xlarge. At the end of compilation, the files `resnet-50_compiled-0000.params` and `resnet-50_compiled-symbol.json` will be created in local directory.

```python
import mxnet as mx
import numpy as np

path='http://data.mxnet.io/models/imagenet/'
mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params')
mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json')
sym, args, aux = mx.model.load_checkpoint('resnet-50', 0)

# Compile for Inferentia using Neuron
inputs = { "data" : mx.nd.ones([1,3,224,224], name='data', dtype='float32') }
sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs)

#save compiled model
mx.model.save_checkpoint("resnet-50_compiled", 0, sym, args, aux)
```

2. If not compiling and inferring on the same instance, copy the artifact to the inference server (use ec2-user as user for AML2):
```bash
scp -i <PEM key file>  resnet-50_compiled-0000.params ubuntu@<instance DNS>:~/  # Ubuntu
scp -i <PEM key file>  resnet-50_compiled-symbol.json ubuntu@<instance DNS>:~/  # Ubuntu
```

3. On the Inf1, create a inference Python script named `infer_resnet50.py` with the following content:
```python
import mxnet as mx
import numpy as np

path='http://data.mxnet.io/models/imagenet/'
mx.test_utils.download(path+'synset.txt')

fname = mx.test_utils.download('https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg?raw=true')
img = mx.image.imread(fname)# convert into format (batch, RGB, width, height)
img = mx.image.imresize(img, 224, 224) # resize
img = img.transpose((2, 0, 1)) # Channel first
img = img.expand_dims(axis=0) # batchify
img = img.astype(dtype='float32')

sym, args, aux = mx.model.load_checkpoint('resnet-50_compiled', 0)
softmax = mx.nd.random_normal(shape=(1,))
args['softmax_label'] = softmax
args['data'] = img

# Inferentia context
ctx = mx.neuron()

exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null')

with open('synset.txt', 'r') as f:
     labels = [l.rstrip() for l in f]

exe.forward(data=img)
prob = exe.outputs[0].asnumpy()# print the top-5
prob = np.squeeze(prob)
a = np.argsort(prob)[::-1]
for i in a[0:5]:
     print('probability=%f, class=%s' %(prob[i], labels[i]))
```

4. Run the script to see inference results:
```bash
python infer_resnet50.py
```
```bash
probability=0.642454, class=n02123045 tabby, tabby cat
probability=0.189407, class=n02123159 tiger cat
probability=0.100798, class=n02124075 Egyptian cat
probability=0.030649, class=n02127052 lynx, catamount
probability=0.016278, class=n02129604 tiger, Panthera tigris
```
