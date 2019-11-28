# AWS Neuron: Frequently Asked Questions WIP

[Getting Started FAQs](#getting-started)

[General Neuron FAQs](#general)

[Inf1 Instance FAQs](#instance)

[Neuron Runtime FAQs](#runtime)

[Troubleshooting FAQs](#troubleshooting)


<a name="getting-started"></a>
## Getting started with Neuron FAQs

**Q: How can I get started?**

You can start your workflow by building and training your model in one of the popular ML frameworks using GPU compute instances such as P3 or P3dn. Once the model is trained to your required accuracy, you can use the ML framework’s API to invoke Neuron, a software development kit for Inferentia, to compile the model for execution on Inferentia, load it in to Inferentia’s memory and then execute inference calls. In order to get started quickly, you can use AWS Deep Learning AMIs (https://aws.amazon.com/machine-learning/amis/) that come pre-installed with ML frameworks and the Neuron SDK. For a fully managed experience you will be able to use Amazon SageMaker which will enable you to seamlessly deploy your trained models on Inf1 instances. 

For customers who use popular frameworks like Tensorflow, MXNet and PyTorch a guide to help you get started with frameworks 
is available at [TODO](). I you wish to deploy without a framework, you can install Neuron using pip 
install and use Neuron to compile and deploy your models to Inf1 instances to run inference. More details on using Neuron without a framework are here [TODO]().


<a name="general"></a>
## General Neuron FAQs

**Q: Why is a compiler needed and how do I use it?**

The Neuron compiler converts from a neural net graph consisting of operators like convolution and pool into a specific set of 
instructions that can be executed using the unique instruction-set of Inferentia.  The formats of these input graphs may be 
from TensorFlow or MXNet or ONNX.  The output from the compiler is the specific instructions for Inferentia encapsulated 
into a binary form, referred to as a Neuron Executable File Format (NEFF). NEFF contains a combination of these instructions, 
the weights and parameters of the pre-trained model and additional meta-data needed by the runtime. 

**Q: In what environments can I use the compiler in?** 
The compilation step may be performed on any EC2 instance or on-premises. 
We recommend using a high-performance compute server of choice (C/M/R/Z instance types), for the fastest compile times and 
ease-of-use with a prebuilt DLAMI. Developers can also install Neuron in their own environments; this approach may work well 
for example when building a large fleet for inference, allowing the model creation, training and compilation to be done in the 
training fleet, with the NEFF files being distributed by a configuration management application to the inference fleet.


The compiler command line options are as follows:

**kcc compile —framework <value> —model <value> —io-config <value> [—num-tpbs <value>] [—output <value>] [--debug]**

An example of this CLI usage showing how to compile a TensorFlow frozen graph in protobuf format is :

>> kcc compile —framework TENSORFLOW —io-config file://test_graph_tfmatmul.config.pbtxt —model file://test_graph_tfmatmul.pb

The compiler compiles the input graph for 1 Inferentia (4 TPBs) by default. This allows any compiled model to work on any N1 instance without change. Using the num-tpbs compiler option, user can direct the compiler to compile the graph for more or fewer TPBs. By increase the number of TPBs, user can create an execution plan with even more performance (throughput) when the plan is run on an N1 instance sized to fit the compiled model's total number of TPBs. Changing this number of TPBs will require a recompile step, which may be desirable for those customers desiring the higher performance available with N1 instance types with more available Inferentia devices.

I am using TensorFlow today – what will change for me to use this?

The likely (but not only) workflow for customers is to build and train models in their training fleet, then partition and compile on a compute server, then distribute and execute on their inference (N1) fleet. The distribution of the artifact may be via services like S3 for example. The additional step of compiling their neural net is new, but all other steps should be consistent with current practices.

KAENA supports both the Kaena Predictor and TensorFlow Serving interfaces. The following example shows how you can use Kaena python library to load a saved model, compile it and perform inference using instance-local Inferentia. The only changes here from the standard usage are the use of KaenaPredictor and a call to its compile_graph method:

STEP 1: on a compute server with Kaena Tensorflow installed:


import tensorflow.contrib.kaena as kaena

#------- Compile SavedModel to Kaena target -----------
kaena.compile(in_saved_model="savedmodel_path", 
              out_saved_model="s3://my_bucket/compiled_savedmodel" 
              input_data_shapes={"x:0":[4,224,224,3]})

Note: STEP1 can be combined via the kaena_tf_compile wrapper script from command line.

>kaena_tf_compile --in_saved_model "savedmodel_path" 
                  --out_saved_model "s3://my_bucket/compiled_savedmodel" 
                  --input_data_shapes {"x:0":[4,224,224,3]} # can also be numpy file instead of shape tuple

STEP2 : Download from S3 and execute inference on N1:

aws s3 cp --recursive s3://my_bucket/compiled_savedmodel compiled_savedmodel/0001

Within Python session with Kaena Tensorflow, import and execute inference:

import tensorflow as tf

#-------Standard predictor object creation from compiled SavedModel -----------
predictor = tf.contrib.predictor.from_saved_model("compiled_savedmodel/0001") 
#-------Standard predictor inference (execute supported ops on Inferentia) ------
results = predictor(feed_dict)


For users making use of a Tensorflow model server:

aws_tensorflow_model_server --model_name=mysaved_model \
    --model_base_path=/home/ubuntu/compiled_savedmodel/ --port=9000

Once the model is loaded to the N1 instances, customers will  send inference requests to it as they normally would.

For full details on TensorFlow interface support, please refer:  http://github.com/aws/aws-mla/kaena/docs/tensor_flow_interfaces.md.

I am using MXnet today - what will change for me to use this?

The likely (but not only) workflow for customers is to build and train in their training fleet, then partition and compile on a  compute server, then distribute and execute on their inference (N1) fleet. The distribution of the artifact may be via services like S3 for example.

KAENA supports both Python Module and Symbol APIs and C predict API. Using the Symbol API, users would load checkpoint as usual, then target Inferentia by setting context to mx.kaena(). Users can optionally compile the graph before the normal binding and inference steps.

The following quick start example uses Module API.

STEP 1: on a compute server with Kaena MXNet installed:

import mxnet as mx

#load image
img = mx.image.imread(fname)

#-------Standard MXNet checkpoint load -----------
sym, args, aux = mx.model.load_checkpoint('checkpoint', 0)

#------- Compile and set Inferentia context -----------
sym, args, aux, cnt = mx.contrib.inferentia.compile(sym, args, aux, data=img)
ctx = mx.infa()

#-------Standard MXNet Module instantiation code-----------
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

#save compiled model
mod.save_checkpoint('compiled_checkpoint',0)

>aws s3 cp --recursive compiled_check_point s3://my_bucket/compiled_check_point

Note: STEP1 can be combined via the kaena_mx_compile wrapper script.

>kaena_mx_compile --in_model_uri_prefix "s3://my_bucket/checkpoint"
                  --epoch_number 0
                  --out_model_uri_prefix "s3://my_bucket/compiled_checkpoint"
                  --input_shapes {"x":[1,224,224,3]} # can also be numpy file instead of shape tuple

STEP2 : Download from S3 and execute inference on N1:

aws s3 cp --recursive s3://my_bucket/compiled_checkpoint_1 compiled_checkpoint

Within Python session with Kaena MXNet, import compiled model and execute inference as usual:

import mxnet as mx

#load image
img = mx.image.imread(fname)

#-------Standard MXNet checkpoint load -----------
sym, args, aux = mx.model.load_checkpoint('compiled_checkpoint', 0)

#------- Compile and set Inferentia context -----------
ctx = mx.infa()

#-------Standard MXNet Module instantiation code-----------
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

#-------Standard MXNet inference code-----------
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
     label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

mod.forward(Batch([img]))
prob = mod.get_outputs()[0].asnumpy()

As usual, MXNet Model Server can be used to serve the compile model after the check point is converted to MXNet archive. Users should use the MMS Management API documented here (https://github.com/awslabs/mxnet-model-server/blob/master/docs/management_api.md#register-a-model). Invoking MMS without the Management API will result in the creation of 1 MMS worker per vCPU or GPU which is undesirable for most Inferentia use cases.

# Add files needed for MXNet archive (model signature and assets)
# Here we use SqueezeNet example
aws s3 cp s3://model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt compiled_checkpoint
aws s3 cp s3://model-server/model_archive_1.0/examples/squeezenet_v1.1/signature.json compiled_checkpoint
# Create the compiled MXNet archive
model-archiver --model-name resnet --model-path compiled_checkpoint --handler mxnet_vision_service:handle

# Serve the model using MMS Management API 
# This example creates 1 initial MMS worker synchronously
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=https%3A%2F%2Fs3.amazonaws.com%2Fmodel-server%2Fmodel_archive_1.0%2Fsqueezenet_v1.1.mar"

For full details on MXnet interface support, please refer:  http://github.com/aws/aws-mla/kaena/docs/mxnet_flow_interfaces.md.


How does KAENA connect to all the Inferentia chips in an N1 instance?

The N1 instance family supports several instance sizes, scaling from 1 to 16 Inferentias assigned to the instance. By default, a single runtime process will manage all assigned Inferentias, including running the Tensor Processing Pipeline (TPP) mode. In some cases, user can configure multiple KRT processes each managing a fraction of assigned Inferentias. Staying with single KRT, and managing multiple Inferentias as an execution pipeline (TPP), a KRT process enables execution of larger models that benefit from model parallel execution mode. 

Is there an  AMI for this?

The KAENA software packages are installed in the KAENA- DL AMI version, and include the KRT, KCC, KDB and integrated ML framework packages. Following the initial release, the KAENA packages will be integrated into the AWS DL- AMI (Specific version to be determined).  Versions for both Amazon Linux 2 and Ubuntu are available. Additionally, Kaena also supports Docker containers and Kubernetes.

My current Neural Network is based on FP32,  How can I use Inferentia?

Inferentia supports FP16 and BFloat16 mixed-precision data-types. It is common for Neural Networks to be trained in FP32, in which case the trained graph needs to be converted to one of these data types for execution on Inferentia. There are 2 ways to perform this:

1) The neural net may be retrained to natively work in one of the supported data types. 
2) Kaena can compile and execute FP32 neural nets by internally converting (to BFloat16). Given an input using FP32, the compiler output will ensure that the executed graph can accept input inference requests in FP32. This is an option that can be selected [COMMENT: desired to be automatic — late binding decision]

Documented examples of accuracy  for several well-known neural net models are available at https://github.com/aws/aws-mla/kaena/docs/fp32_conversion_examples/

[INTERNAL NOTE: The neural net may be converted prior to use with Kaena, using AMP support in Tensorflow and MXnet. THIS IS WORK UNDERWAY AND NOT COMMITTED YET]

Which operators doES kaena support?

KAENA defines a set of supported operators that may be executed on the Inferentia (see http://github.com/aws/aws-mla/kaena/docs/kaena-supported-operators/, also available via *kcc-cc list-operators --framework <value>*). Any operators not supported by KAENA will be automatically partitioned out of the ML Model and executed by the framework on the host. The partitioning tools are provided for the supported ML frameworks.  

Will Elastic Inference support Inferentia?

Customers will use the same methods that they currently do to create an EI-capable instance (choosing an Inferentia-based EI appliance instance rather than a GPU as today). 

Will SageMaker support Inferentia?

SageMaker-based use of Inferentia and KAENA will be available via Elastic Inference. 

[INTERNAL NOTE: Direct Sagemaker support for N1 is being planned but not committed for re:Invent 2019]  

How can I  debug / profile my inference request?

Tensorboard will enable offline debug and profiling of ML Models from the framework operator level, which is the most convenient for many customers, by processing the output of profile traces to show the execution time of all neural net operators or processing debug logs to display the tensor values at each layer.  For the most advanced builders, KAENA will provide an expert debugger following the MVP. This cli-driven debugger enables single stepping and breakpointing of neural nets executing on Inferentia as well as the examination of memory and the disassembly of instructions into a convenient form. 

Where can I get logging and other telemetry information?

KRT and KCC have been built to generate log information to several destinations: local file, syslog and CloudWatch. Customers can supply their own CloudWatch credentials and destinations.

What about RedHat or other versions of Linux? 

The KAENA software packages are pre-compiled for Linux variants and are available in the ML Acceleration repository in the AWS area on Github http://github.com/aws/aws-mla/kaena/release-packages/. These packages may be installed on Linux distributions directly, or may be installed with apt-get

What about Windows?

Windows is not supported at this time.

How can I use Kaena in a container based environment? Does Kaena work with ECS and EKS?

Taking advantage of Inferentia and Kaena runtime from within a container requires three specific actions: (1) associate the container to specific Inferentias via an environment variable, (2) use the oci-add-hooks as the runtime, and (3) pass the --privileged flag to the container.

export KAENA_VISIBLE_DEVICES="00:1d.5, 00:1d.7"
docker run  -it  <img> runtime=*oci-add-hooks*  --privileged bash

EKS Customers using Inferentia will benefit from Kaena's device plugin found here: https://github.com/aws/aws-mla/kaena/plugins/, which will advertise their Inferentias to the kubelet.  Kaena's device plugin can be deployed manually or as part of a Deamonset. [INTERNAL NOTE: EKS and ECS TEAM SUPPORT IS NOT YET COMMITTED FOR THIS].


Will I need to recompile again if I updated runtime/driver version?

The compiler and runtime are committed to maintaining compatibility for major version releases with each other. The versoning is defined as major.minor, with compatibility for all versions with the same major number. If the versions mismatch, an error notification is logged and the load will fail. This will then require the model to be recompiled.

How can I take advantage of multiple TPBs and run multiple inferences in parallel ?

Running inferences on a set of Inferentias has 2 considerations: how many models are to be executing in parallel and how many TPBs does each model require. The aggregate number of TPBs will determine the number of Inferentias (with each Inferentia supplying 4 TPBS). Once the number of Inferentia devices is determined, a choice can be made as to how to parallelize: via separate N1 instances or via a single N1 instance. If a single N1 instance is desired (as for example where the application needs several models to execute and may wish to be tightly coupled to them all), then  another choice can be made as to whether to aggregate all Inferentias in the N1 instance under a single KRT runtime interface (KRTD), or via separated KRTs, each managing a separate set of Inferentia(s). Separation into more KRTD may be a better choice, if reducing the “blast radius” of software issues is a concern.

Once this is decided, the mapping of each Inferentia to each KRTD is set up with the systemd configuration file, and the framework(s) will then connect to the appropriate set of KRTD service(s).

This is an advanced topic, since in the general case, the default configuration of each N1 instance will be sufficient to allow all available Inferentias to be used via a single KRTD and all TPBs are managed transparently by KRTD.

Examples and further documentation for this can be found at https://github.com/aws/aws-mla/kaena/examples

I have tonga binary, how can I tell which compiler version generated it ? and which config it is targeting ?

The* kaena-info* cli utility  provides this information, as well as other data about the compiled files.

what common neural networks does kaEna support ?

Our supported sample models include representative neural nets in several areas, including Image recognition and object detection, NLP and Translation:  Resnet50, Inception v3, BERT, LSTM-based PTB word model, Parallel Wavenet, Wave RNN

The full list is found at https://github.com/aws/aws-mla/examples 

can i use tensorFlow networks from tfhub.dev as-is ? if not, what should i do?

Yes. Models format can  be imported into Tensorflow, either as a standard model-server, in which case it appears as a simple command line utility, or via the Python based Tensorflow environment.  The primary additional step needed is to compile it. 

do I need to worry about size of model and size of inferentia memory ? what problems can i expect to have?

For best performance, KAENA stages the models and parameters into Inferentia memory. When this is full (because several models are already simultaneously stored), then the load() command will fail. In such a case, calling unload() for previously loaded models will free space. If a single model is unable to be loaded, even with no other models loaded, then it may be necessary to compile this model for additional TPBs, requiring more Inferentia resources. The load() api provides this guidance on failure.

how do i upgrade compiler and runtime components?  (will it have a repo  so customer just do yum update or apt-get?)

The installation process uses apt to install packages form the kaena apt and pip wheel repos:

sudo apt-get update
sudo apt-get install "aws-kaena*"
pip3 install --user infa-cc[tensorflow]
# or 
pip3 install infa-cc[mxnet] for the MxNet "extras"

How WOULD I select which N1 instance to use?

EC2 N1 instances are available in various sizes ranging from 1 to 16 Inferentia accelerators per instance. The decision as to which N1 instance to use is based upon the application and the performance/cost targets and will differ for each customer workload. To help this process, the compiler provides guidance on expected latency and throughput for various N1 instance sizes, and Tensorboard profiling will show actual results when executed on a given instance. The KAENA software compiler is able to target varying amounts of Inferentia resources, which will increase throughput and reduce latency in many cases. A guide to this process and how to supply configuration input to the compiler is available here: http://github.com/aws/aws-mla/kaena/docs/model-parallel-compiling.md (http://github.com/aws/aws-mla/docs/model-parallel-compiling.md)
