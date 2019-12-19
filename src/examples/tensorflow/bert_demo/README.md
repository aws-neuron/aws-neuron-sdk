# BERT demo for AWS Neuron

To enable a performant BERT model on Inferentia, we must use a Neuron compatible BERT implementation. This demo shows a Neuron compatible BERT-Large implementation that is functionally equivalent to open source BERT-Large. We then show how to compile and run it on an inf1.2xlarge instance. This demo uses Tensorflow-Neuron and also shows the performance achieved by the Inf1 instance. 

## Table of Contents

1. Launch EC2 instanceas and update Neuron software
2. Compiling Neuron compatible BERT-Large for Inferentia
   a. Create Neuron compatible saved model 
   b. Compile saved model for Inferentia
3. Run the inference demo
   a. Launch the BERT demo server
   b. Send async requests to server

## Launch EC2 instances and update Neuron software

For this demo, we will use a c5.4xlarge EC2 instance for compiling the BERT Model and an inf1.2xlarge instance for running the demo itself. For both of these instances choose the latest an Ubuntu 18 Deep Learning AMI. After starting the instance please make sure you update the Neuron software to the latest version before continuing with this demo.

Instructions to launch and update Neuron Software can be found here :
* [Getting Started with TensorFlow-Neuron (ResNet-50 Tutorial)](../../../../docs/tensorflow-neuron/tutorial-compile-infer.md)
* [Getting Started with Neuron-Runtime](../../../../docs/neuron-runtime/nrt_start.md)


## Compiling Neuron compatible BERT-Large for Inferentia
NOTE : Please make sure you update the Neuron software to the latest version before continuing with this demo.

Connect to your c5.4xlarge instance and run the following commands to activate the tensorflow neuron environment. Also note: refer to the release notes for more information [Release Notes](../../../../release-notes/conda/conda-tensorflow-neuron.md#known-issues-and-limitations-1)

```bash
conda activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```

### Create Neuron compatible saved model
The Neuron compatible implementation of BERT-Large is defined in [bert_model.py](./bert_model.py). Download the weights from the public version of BERT located [here](https://github.com/google-research/bert). Use the Neuron compatible BERT-Large implementation and public BERT-Large weights to generate a saved model using steps outlined in public BERT documentation [here](https://github.com/google-research/bert/issues/146). Save this into a directory named bert-saved-model.


### Compile saved model for Inferentia
In the same conda environment and directory containing your bert-saved-model, run the following script :

```
bert_model.py --input_saved_model ./bert-saved-model/ --output_saved_model ./bert_saved_model_neuron --crude_gelu
```

This compiles BERT-Large for an input size of 128 and batch size of 4. The compilation output is stored in bert_saved_model_neuron. Copy this to your Inf1 instance for inferencing.

## Running the inference demo
NOTE: please make sure you update the Neuron software to the latest version before continuing with this demo.

### Launching the BERT demo server
On your inf1.2xlarge, activate the updated conda environment for tensorflow-neuron :

```
conda activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron

```

Then launch the BERT demo server :
```
bert_server.py --dir <directory_path_of_bert_saved_model_neuron> --parallel 4
```
This loads 4 BERT-Large models, one into each of the 4 NeuronCores in a single Inferentia device. For each of the 4 models, the BERT demo server opportunistically stiches together asynchronous requests into batch 4 requests. When there are insufficient pending requests, the server creates dummy requests for batching.

Wait for the bert_server to finish loading the BERT models to Inferentia memory. When it is ready to accept requests it will print the inferences per second once every second. This reflects the number of real inferences only. Dummy requests created for batching are not credited to inferentia performance.

### Sending async requests to server
On the same inf1.2xlarge instance, launch a separate linux terminal. From there execute the following commands :

```
conda activate aws_neuron_tensorflow_p36
for i in {1..48}; do bert_client.py --cycle 128 & done
```

This spins up 48 clients, each of which sends 128 inference requests. The expected performance is about 200 inferences/second for a single instance of inf1.2xlarge.

