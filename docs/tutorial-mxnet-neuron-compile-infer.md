# Tutorial: Using MXnet-Neuron and the Neuron Compiler

Neuron supports both Python Module and Symbol APIs and the C predict API. Using the Symbol API, users would load checkpoint as usual, then target Inferentia by setting context to mx.Neuron(). Users can optionally compile the graph before the normal binding and inference steps.

The following quick start example uses the Module API.

## Steps Overview:

1. Launch an EC2 instance for compilation and/or Inference
2. Install Neuron for Compiler and Runtime execution
3. Run Example
    1. Compile 
    2. Execute Inference on Inf1

## Step 1: Launch EC2 Instances

A typical workflow with the Neuron SDK will be for a previously trained ML model to be compiled on a compilation server and then the artifacts distributed to the (fleet of) inf1 instances for execution. Neuron enables Tensorflow to be used for all of these steps.
Steps Overview:

1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based. To use a pre-built Deep Learning AMI, which includes all of the needed packages, see these instructions: https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html
2. Select and start an EC2 instance of your choice (see https://aws.amazon.com/ec2/instance-types/) to compile
    1. It is recommended to use C5.4xlarge or larger. For this example we will use a C5.4xlarge
    2. If you would like to compile and infer on the same machine, please select Inf1.6xlarge
3. Select and start an INF1 instance of your choice if not compiling and inferencing on the same instance;

## Step 2: Install Neuron

### Compiler Instance: Install Neuron Compiler and MXnet-Neuron

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
pip install mxnet-neuron
```

### Inference Instance: Install Tensorflow-Neuron and Neuron-Runtime

1. same as above to install Mxnet-Neuron
2. to install Runtime, see [This link](https://github.com/aws/aws-neuron-sdk/blob/master/docs/getting-started-neuron-rtd.md)

## Step 3: Run Example

### Compile on a compute server with Neuron MXNet installed:

```
import mxnet as mx
#load imageimg = mx.image.imread(fname)

#-------Standard MXNet checkpoint load -----------
sym, args, aux = mx.model.load_checkpoint('checkpoint', 0)

#------- Compile and set Inferentia context -----------
sym, args, aux, cnt = mx.contrib.inferentia.compile(sym, args, aux, data=img)
ctx = mx.infa()

#-------Standard MXNet Module instantiation code-----------
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
#save compiled model
mod.save_checkpoint('compiled_checkpoint',0)`
```

Now save the compiled file to your desired distribution repository, in this case, S3:

```
aws s3 cp --recursive compiled_check_point s3://my_bucket/compiled_check_point
```

Note: this step can be combined via the Neuron_mx_compile wrapper script:

```
Neuron_mx_compile --in_model_uri_prefix "s3://my_bucket/checkpoint" --epoch_number 0
                  --out_model_uri_prefix "s3://my_bucket/compiled_checkpoint" --input_shapes {"x":[1,224,224,3]} # can also be numpy file instead of shape tuple`
```

### Download from S3 and execute inference on Inf1:

```
aws s3 cp --recursive s3://my_bucket/compiled_checkpoint_1 compiled_checkpoint
```

Within Python session with Neuron MXNet, import compiled model and execute inference as usual:

```
import mxnet as mx
#load imageimg = mx.image.imread(fname)
#-------Standard MXNet checkpoint load -----------
sym, args, aux = mx.model.load_checkpoint('compiled_checkpoint', 0)
#------- Compile and set Inferentia context -----------ctx = mx.infa()
#-------Standard MXNet Module instantiation code-----------mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)#-------Standard MXNet inference code-----------
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
     label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

mod.forward(Batch([img]))prob = mod.get_outputs()[0].asnumpy()
```

### Alternate : Model Server

An MXNet Model Server can be used to serve the compile model after the check point is converted to MXNet archive. Users should use the MMS Management API documented here (https://github.com/awslabs/mxnet-model-server/blob/master/docs/management_api.md#register-a-model). Invoking MMS without the Management API will result in the creation of 1 MMS worker per vCPU or GPU which is undesirable for most Inferentia use cases.

```
# Add files needed for MXNet archive (model signature and assets)# Here we use SqueezeNet example
aws s3 cp s3://model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt compiled_checkpoint
aws s3 cp s3://model-server/model_archive_1.0/examples/squeezenet_v1.1/signature.json compiled_checkpoint# Create the compiled MXNet archive

model-archiver --model-name resnet --model-path compiled_checkpoint --handler mxnet_vision_service:handle

# Serve the model using MMS Management API# This example creates 1 initial MMS worker synchronously
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=https%3A%2F%2Fs3.amazonaws.com%2Fmodel-server%2Fmodel_archive_1.0%2Fsqueezenet_v1.1.mar"
```

To dive deeper, consult the [Neuron MXNet API Getting Started guide]()

