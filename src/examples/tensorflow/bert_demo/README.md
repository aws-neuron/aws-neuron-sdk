# BERT demo for AWS Neuron

To enable a performant BERT model on Inferentia, we must use a Neuron compatible BERT implementation. This demo shows a Neuron compatible BERT Large implementation that is functionally equivalent to open source BERT Large. We then show how to compile and run it on an inf1.2xlarge instance. This demo uses Tensorflow-Neuron, BERT Large weights fine tuned for MRPC and also shows the performance achieved by the inf1 instance. 

This demo assumes users are familiar with AWS in general.

## Table of Contents

1. Launch EC2 instances 
2. Compiling Neuron compatible BERT Large for Inferentia
   * Update compilation EC2 instance
   * Compile open source BERT Large saved model using Neuron compatible BERT Large implementation
3. Running the inference demo
   * Update inference EC2 instance
   * Launching the BERT demo server
   * Sending requests to server from multiple clients

## Launch EC2 instances

For this demo, launch two EC2 instances :
   * a c5.4xlarge EC2 instance for compiling the BERT Model and 
   * an inf1.2xlarge instance for running inference 

For both of these instances choose the latest an Ubuntu 18 Deep Learning AMI (DLAMI).

## Compiling Neuron compatible BERT Large for Inferentia
First connect to your c5.4xlarge instance and update tensorflow-neuron and neuron-cc

### Update compilation EC2 instance
Update to the latest neuron software by executing the following commands :

```bash
source activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```

### Compile open source BERT Large saved model using Neuron compatible BERT Large implementation
Neuron software works with tensorflow saved models. Users should bring their own BERT Large saved model for this section. This demo will run inference for the MRPC task and the saved model should be finetuned for MRPC. Users who need additional help to finetune their model for MRPC or to create a saved model can refer to [Appendix 1](https://github.com/HahTK/aws-neuron-sdk/tree/master/src/examples/tensorflow/bert_demo#appendix-1-). 

In the same conda environment and directory bert_demo scripts, run the following :

```bash
export BERT_LARGE_SAVED_MODEL="/path/to/user/bert-large/savedmodel"
python bert_model.py --input_saved_model $BERT_LARGE_SAVED_MODEL --output_saved_model ./bert-saved-model-neuron --crude_gelu
```

This compiles BERT Large pointed to by $BERT_LARGE_SAVED_MODEL for an input size of 128 and batch size of 4. The compilation output is stored in bert-saved-model-neuron. Copy this to your inf1 instance for inferencing. 

The bert_model.py script encapsulates all the steps necessary for this process. For details on what is done by bert_model.py please refer to [Appendix 2](https://github.com/HahTK/aws-neuron-sdk/tree/master/src/examples/tensorflow/bert_demo#appendix-2-).

## Running the inference demo
Connect to your inf1.2xlarge instance and update tensorflow-neuron, aws-neuron-runtime and aws-neuron-tools.

### Update inference EC2 instance
Update to the latest neuron software by executing the following commands :

```bash
source activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```

### Launching the BERT demo server
Copy the compiled model (bert-saved-model-neuron) from your c5.4xlarge to your Inf1.2xlarge instance. Place the model in the same directory as the bert_demo scripts. Then from the same conda environment launch the BERT demo server :

```bash
python bert_server.py --dir bert-saved-model-neuron --parallel 4
```

This loads 4 BERT Large models, one into each of the 4 NeuronCores found in a single Inferentia device. For each of the 4 models, the BERT demo server opportunistically stiches together asynchronous requests into batch 4 requests. When there are insufficient pending requests, the server creates dummy requests for batching.

Wait for the bert_server to finish loading the BERT models to Inferentia memory. When it is ready to accept requests it will print the inferences per second once every second. This reflects the number of real inferences only. Dummy requests created for batching are not credited to inferentia performance.

### Sending requests to server from multiple clients
Wait until the bert demo server is ready to accept requests. Then on the same inf1.2xlarge instance, launch a separate linux terminal. From the bert_demo directory execute the following commands :

```bash
source activate aws_neuron_tensorflow_p36
for i in {1..48}; do python bert_client.py --cycle 128 & done
```

This spins up 48 clients, each of which sends 128 inference requests. The expected performance is about 200 inferences/second for a single instance of inf1.2xlarge.

## Appendix 1 :
Users who need help finetuning BERT Large for MRPC and creating a saved model may follow the instructions here.

Connect to the c5.4xlarge compilation EC2 instance you started above and download these three items : 
1. clone [this](https://github.com/google-research/bert) github repo. Then edit run_classifier.py as described [here](https://github.com/google-research/bert/issues/146#issuecomment-569138476). We may ignore the changes described for run_squad.py.  
2. download GLUE data as described [here](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks). Do not run the finetuning command.
3. download a desired pre-trained BERT Large checkpoint from [here](https://github.com/google-research/bert#pre-trained-models). This is the model we will fine tune. 

Then from the bert_demo directory run the following :

```bash
source activate aws_neuron_tensorflow_p36
export BERT_REPO_DIR="/path/to/cloned/bert/repo/directory"
export GLUE_DIR="/path/to/glue/data/directory"
export BERT_BASE_DIR="/path/to/pre-trained/bert-large/checkpoint/directory"
./tune_save.sh
```

The a saved model will be created in $BERT_REPO_DIR/bert-saved-model/_random_number_/. Where, _random_number_ is a random number generated for every run. Use this saved model to continue with the rest of the demo. 

## Appendix 2 :
For BERT, we currently augment the standard Neuron compilation process for performance tuning. In the future, we intend to automate this tuning process. This would allow users to use the standard Neuron compilation process, which requires only a one line change in user source code. This is as described [here](https://github.com/aws/aws-neuron-sdk/blob/master/docs/tensorflow-neuron/tutorial-compile-infer.md#step-3-compile-on-compilation-instance).

The augmented Neuron compilation process is encapsulated by the bert_model.py script, which performs the following things :
1. Define a Neuron compatible implementation of BERT Large. For inference, this is functionally equivalent to the open source BERT Large. The changes needed to create a Neuron compatible BERT implementation is described in [Appendix 3](https://github.com/HahTK/aws-neuron-sdk/tree/master/src/examples/tensorflow/bert_demo#appendix-3-).
2. Extract BERT Large weights from the open source saved model pointed to by --input_saved_model and associates it with the Neuron compatible model
3. Invoke Tensorflow-Neuron to compile the Neuron compatible model for Inferentia using the newly associated weights
4. Finally, the compiled model is saved into the location given by --output_saved_model

## Appendix 3 :
The Neuron compatible implementation of BERT Large is functionally equivalent to the open source version when used for inference. However, the detailed implementation does differ and here are the list of changes :

1. Data Type Casting : If the original BERT an FP32 model, bert_model.py contains manually defined cast operators to enable mixed-precision. FP16 is used for multi-head attention and fully-connected layers, and fp32 everywhere else. This will be automated in a future release.
2. Remove Unused Operators: A model typically contains training operators that are not used in inference, including a subset of the reshape operators. Those operators do not affect inference functionality and have been removed.
3. Reimplementation of Selected Operators : A number of operators (mainly mask operators), has been reimplemented to bypass a known compiler issue. This will be fixed in a planned future release. 
4. Manually Partition Embedding Ops to CPU : The embedding portion of BERT has been partitioned manually to a subgraph that is executed on the host CPU, without noticable performance impact. In near future, we plan to implement this through compiler auto-partitioning without the need for user intervention.
