# BERT Demo for AWS Neuron

To enable a performant BERT model on Inferentia, we must use a Neuron compatible BERT implementation. This demo shows a Neuron compatible BERT Large implementation that is functionally equivalent to open source BERT Large. We then show how to compile and run it on an Inf1.2xlarge instance. This demo uses Tensorflow-Neuron, BERT Large weights fine tuned for MRPC and also shows the performance achieved by the Inf1 instance. 

## Table of Contents

1. Launch EC2 instanceas and update Neuron software
2. Compiling Neuron compatible BERT Large for Inferentia
   a. Create saved model from open source BERT Large
   b. Compile model using Neuron compatible BERT Large
3. Running the inference demo
   a. Launching the BERT demo server
   b. Sending async requests to server

## Launch EC2 instances and update Neuron software

For this demo, we will use a c5.4xlarge EC2 instance for compiling the BERT Model and an Inf1.2xlarge instance for running the demo itself. For both of these instances choose the latest an Ubuntu 18 Deep Learning AMI. After starting the instance please make sure you update the Neuron software to the latest version before continuing with this demo.

Instructions to launch and update Neuron Software can be found here :
* [Getting Started with TensorFlow-Neuron (ResNet-50 Tutorial)](../../../docs/tensorflow-neuron/tutorial-compile-infer.md)
* [Getting Started with Neuron-Runtime](../../../docs/neuron-runtime/nrt_start.md)


## Compiling Neuron compatible BERT Large for Inferentia
NOTE : Please make sure you update the Neuron software to the latest version before continuing with this demo.

Connect to you c5.4xlarge instance and run the following commands to activate the tensorflow neuron environment.

```
conda activate aws_neuron_tensorflow_p36

```

### Create saved model from open source BERT Large
For this step, we use publicly available models and instructions that the user can repeat. The BERT Large impelementation and MRPC weights are found [here](https://github.com/google-research/bert). The instructions for creating a saved model for this is found in [this](https://github.com/google-research/bert/issues/146) public document. Place the saved model in a directory named "bert-saved-model".

### Compile model using Neuron compatible BERT Large 
In the same conda environment and directory containing your bert-saved-model, run the following script :

```
bert_model.py --input_saved_model ./bert-saved-model/ --output_saved_model ./bert_saved_model_neuron --crude_gelu
```

This compiles BERT large for an input size of 128 and batch size of 4. The compilation output is stored in bert_saved_model_neuron. Move this to your Inf1 instances for inferencing. For details on what is done by bert_model.py please refer to Appendix 1.

The bert_model.py script does a few things :
1. Define a Neuron compatible implementation of BERT Large. For inference, this is functionally equivalent to the open source BERT Large
2. Extract BERT Large weights from the saved model pointed to by --input_saved_model and associates it with the Neuron compatible implementation
3. Invoke Tensorflow-Neuron compile to compile this model for Inferentia using the newly associated weights
4. Finally, the compiled model is saved into the location given by --output_saved_model


## Running the inference demo
NOTE : Please make sure you update the Neuron software to the latest version before continuing with this demo.

### Launching the BERT demo server
On your Inf1.2xlarge, activate the updated conda environment for tensorflow-neuron :

```
conda activate aws_neuron_tensorflow_p36

```

Then launch the BERT demo server :
```
bert_server.py --dir <directory_path_of_bert_saved_model_neuron> --parallel 4
```
This loads 4 BERT Large models, one into each of the 4 NeuronCores in a single Inferentia device. For each of the 4 models, the BERT demo server opportunistically stiches together asynchronous requests into batch 4 requests. When there are insufficient pending, requests the server creates dummy requests for batching.

Wait for the bert_server to finish loading the BERT models to Inferentia memory. When it is ready to accept requests it will print the inferences per second once every second. This reflects the number of real inferences only. Dummy requests created for batching are not credited to inferentia performance.

### Sending async requests to server
On the same Inf1.2xlarge instance, launch a separate linux terminal. From there execute the following commands :

```
conda activate aws_neuron_tensorflow_p36
for i in {1..48}; do bert_client.py --cycle 128 & done
```

This spins up 48 clients, each of which sends 128 inference requests. The expected performance is about 200 inferences/second for a single instance of Inf1.2xlarge.


## Appendix 1 :
The bert_model.py script does a few things :
1. Define a Neuron compatible implementation of BERT Large. For inference, this is functionally equivalent to the open source BERT Large
2. Extract BERT Large weights from the saved model pointed to by --input_saved_model and associates it with the Neuron compatible implementation
3. Invoke Tensorflow-Neuron compile to compile this model for Inferentia using the newly associated weights
4. Finally, the compiled model is saved into the location given by --output_saved_model
The changes needed for a Neuron compatible BERT implementation is given in Appendix 2.


## Appendix 2 :
The Neuron compatible implementation of BERT Large is functionally equivalent to the open source version when used for inference. However, the detailed implementation does differ and here are the list of changes :

1. Auto BF16/FP16 precision
2. Training only ops were removed
3. Some ops were reimplemented ops
4. Embedding ops were manually partitioned to CPU



