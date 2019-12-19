# BERT Demo for AWS Neuron

This demo shows how to compile BERT Large for AWS Neuron, run it on an inf1.2xlarge instance. It uses Tensorflow-Neuron and also shows the performance available with an Inf1 instance.

## Table of Contents

1. Launch EC2 instanceas and update Neuron software
2. Compile BERT Large for Inferemtia
   a. Download BERT Large from github and create a saved model
   b. Compile saved model for Inferentia
3. Run the inference demo
   a. Launch the BERT demo server
   b. Send async requests to server

## Launch EC2 instances

For this demo, a C5.4xlarge EC2 instance for compiling the BERT Model and an inf1.2xlarge instance for running the demo itself. For both of these instances choose the latest an Ubuntu 18 Deep Learning AMI. After starting the instance please make sure you update the Neuron software to the latest version before continuing with this demo.

Instructions to launch and update Neuron Software can be found here :
* [Getting Started with TensorFlow-Neuron (ResNet-50 Tutorial)](../../../../docs/tensorflow-neuron/tutorial-compile-infer.md)
* [Getting Started with Neuron-Runtime](../../../../docs/neuron-runtime/nrt_start.md)


## Compiling BERT Large for Inf1
NOTE : Please make sure you update the Neuron software to the latest version before continuing with this demo.

Connect to you C5.4xlarge instance and run the following commands to activate the tensorflow neuron environment.

```
conda activate aws_neuron_tensorflow_p36

```


### Download BERT Large and create a saved model

This demo uses the public version of BERT located [here](https://github.com/google-research/bert). On your c5 machine, download and convert it into a saved model using steps outlined in public BERT documentation [here](https://github.com/google-research/bert/issues/146) into a directory named bert-saved-model.

### Compile saved model for Inferentia

In the same conda environment and directory containing your bert-saved-model, run the following script :

```
bert_model.py --input_saved_model ./bert-saved-model/ --output_saved_model ./bert_saved_model_neuron --crude_gelu
```

This compiles BERT large for an input size of 128 and batch size of 4. The compilation output is stored in bert_saved_model_neuron. Move this to your Inf1 instance for inferencing.

## Running the inference demo
NOTE : Please make sure you update the Neuron software to the latest version before continuing with this demo.

### Launching the BERT demo server
On your inf1.2xlarge, activate the updated conda environment for tensorflow-neuron :

```
conda activate aws_neuron_tensorflow_p36

```

The launch the BERT demo server :
```
bert_server.py --dir <directory_path_of_bert_saved_model_neuron> --parallel 4
```
This loads 4 BERT Large models, one into each of the 4 NeuronCores in a single Inferentia device. For each of the 4 models, the BERT demo server opportunistically stictch togther asynchronous requests into batch 4 requests when possible. When there are insufficient requests from the clients the server creates dummy requests for batching.

### Sending async requests to server
On the same Inf1.2xlarge instance, launch a separate linux terminal. From there execute the following commands :

```
conda activate aws_neuron_tensorflow_p36
for i in {1..192}; do bert_client.py --cycle 128 & done
```

This spins up 48 clients, each of which sends 64 asynchronous requests. The expected performance is about 200 inferences/second for a single instance of Inf1.2xlarge.

