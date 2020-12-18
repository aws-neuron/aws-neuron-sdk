</br>
</br>

Please view our documentation at **[https://awsdocs-neuron.readthedocs-hosted.com/](https://awsdocs-neuron.readthedocs-hosted.com/)** 

**Please note this file will be deprecated.**

</br>
</br>



# Running TensorFlow BERT-Large with AWS Neuron

This example shows a Neuron compatible BERT-Large implementation that is functionally equivalent to open source BERT-Large model. This demo uses TensorFlow-Neuron, BERT-Large weights fine tuned for MRPC and also shows the performance achieved by the Inf1 instance. For users who want to use public BERT SavedModels please also follow the steps described [below](#publicbert).

## Table of Contents

1. Launch EC2 instances 
2. Compiling Neuron compatible BERT-Large
   * Update compilation EC2 instance
   * Compile open source BERT-Large saved model using Neuron compatible BERT-Large implementation
3. Running the inference demo
   * Update inference EC2 instance
   * Launching the BERT-Large demo server
   * Sending requests to server from multiple clients

## Launch EC2 instances

For this demo, launch two EC2 instances :
   * a c5.4xlarge instance for compiling the BERT-Large Model and 
   * an inf1.xlarge instance for running inference

For both of these instances choose the latest Ubuntu 18 Deep Learning AMI (DLAMI).

## Compiling Neuron compatible BERT-Large
First connect to a c5.4xlarge instance and update tensorflow-neuron and neuron-cc

### Update compilation EC2 instance
Update to the latest neuron software by executing the following commands :

```bash
source activate aws_neuron_tensorflow_p36
conda update tensorflow-neuron
conda update numpy
```
Note: if your tensorflow-neuron version on the inference instance is lower than 1.15.0.1.0.1333.0, you will need to run this demo on inf1.2xlarge instead of inf1.xlarge.

### Compile open source BERT-Large saved model using Neuron compatible BERT-Large implementation
Neuron software works with tensorflow saved models. Users should bring their own BERT-Large saved model for this section. This demo will run inference for the MRPC task and the saved model should be fine tuned for MRPC. Users who need additional help to fine-tune the model for MRPC or to create a saved model can refer to [Appendix 1](#appendix1). 

In the same conda environment and directory bert_demo scripts, run the following :

```bash
export BERT_LARGE_SAVED_MODEL="/path/to/user/bert-large/savedmodel"
python bert_model.py --input_saved_model $BERT_LARGE_SAVED_MODEL --output_saved_model ./bert-saved-model-neuron --batch_size=6 --aggressive_optimizations
```

This compiles BERT-Large pointed to by $BERT_LARGE_SAVED_MODEL for an input size of 128 and batch size of 6. The compilation output is stored in bert-saved-model-neuron. Copy this to your Inf1 instance for inferencing.

The bert_model.py script encapsulates all the steps necessary for this process. For details on what is done by bert_model.py please refer to [Appendix 2](#appendix2).

## Running the inference demo
Connect to your inf1.xlarge instance and update tensorflow-neuron, aws-neuron-runtime and aws-neuron-tools.

### Update inference EC2 instance
Update to the latest neuron software by executing the following commands :

```bash
source activate aws_neuron_tensorflow_p36
conda update tensorflow-neuron
conda update numpy
```

### Launching the BERT-Large demo server
Copy the compiled model (bert-saved-model-neuron) from your c5.4xlarge to your inf1.xlarge instance. Place the model in the same directory as the bert_demo scripts. Then from the same conda environment launch the BERT-Large demo server :

```bash
sudo systemctl restart neuron-rtd
python bert_server.py --dir bert-saved-model-neuron --batch 6 --parallel 4
```

This loads 4 BERT-Large models, one into each of the 4 NeuronCores found in an inf1.xlarge instance. For each of the 4 models, the BERT-Large demo server opportunistically stitches together asynchronous requests into batch 6 requests. When there are insufficient pending requests, the server creates dummy requests for batching.

Wait for the bert_server to finish loading the BERT-Large models to Inferentia memory. When it is ready to accept requests it will print the inferences per second once every second. This reflects the number of real inferences only. Dummy requests created for batching are not credited to inferentia performance.

### Sending requests to server from multiple clients
Wait until the bert demo server is ready to accept requests. Then on the same inf1.xlarge instance, launch a separate linux terminal. From the bert_demo directory execute the following commands :

```bash
source activate aws_neuron_tensorflow_p36
for i in {1..96}; do python bert_client.py --cycle 128 & done
```

This spins up 96 clients, each of which sends 128 inference requests. The expected performance is about 360 inferences/second for a single instance of inf1.xlarge.

<a name="publicbert"></a>
## Using public BERT SavedModels
We are now providing a compilation script that has better compatibility with various flavors of BERT SavedModels generated from https://github.com/google-research/bert. Here are the current limitations:

1. You did not change [modeling.py](https://github.com/google-research/bert/blob/master/modeling.py)
2. BERT SavedModel is generated using `estimator.export_saved_model`
3. BERT SavedModel uses fixed sequence length 128 (you may check by `saved_model_cli show --dir /path/to/user/bert/savedmodel --all`)
4. `neuron-cc` version is at least 1.0.12000.0
5. `aws-neuron-runtime` version is at least 1.0.7000.0
6. The `--batch_size` argument specified in this script is at most 4

Example usage is shown below:
```bash
export BERT_LARGE_SAVED_MODEL="/path/to/user/bert-large/savedmodel"
python bert_no_model.py --input_saved_model $BERT_LARGE_SAVED_MODEL --output_saved_model ./bert-saved-model-neuron --batch_size=1
```
<a name="appendix1"></a>
## Appendix 1
Users who need help finetuning BERT-Large for MRPC and creating a saved model may follow the instructions here.

Connect to the c5.4xlarge compilation EC2 instance you started above and download these three items : 
1. clone [this](https://github.com/google-research/bert) github repo. 
2. download GLUE data by running the following command:
``` python download_mrpc_data.py ```
3. download a desired pre-trained BERT-Large checkpoint from [here](https://github.com/google-research/bert#pre-trained-models). This is the model we will fine tune. 

Next edit run_classifier.py in the cloned bert repo to apply the patch described in the following git diff. 

```
diff --git a/run_classifier.py b/run_classifier.py
index 817b147..c9426bc 100644
--- a/run_classifier.py
+++ b/run_classifier.py
@@ -955,6 +955,18 @@ def main(_):
         drop_remainder=predict_drop_remainder)
 
     result = estimator.predict(input_fn=predict_input_fn)
+    features = {
+        "input_ids": tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name='input_ids'),
+        "input_mask": tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name='input_mask'),
+        "segment_ids": tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name='segment_ids'),
+        "label_ids": tf.placeholder(shape=[None], dtype=tf.int32, name='label_ids'),
+        "is_real_example": tf.placeholder(shape=[None], dtype=tf.int32, name='is_real_example'),
+    }
+    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)
+    estimator._export_to_tpu = False  ## !!important to add this
+    estimator.export_saved_model(
+        export_dir_base='./bert_classifier_saved_model',
+        serving_input_receiver_fn=serving_input_fn)
 
     output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
     with tf.gfile.GFile(output_predict_file, "w") as writer:
```

NOTE : Users who are interested may refer to this [link](https://github.com/google-research/bert/issues/146#issuecomment-569138476) for additional background information on the patch but it is not necessary for running this demo.


Then from the bert_demo directory run the following :

```bash
source activate aws_neuron_tensorflow_p36
export BERT_REPO_DIR="/path/to/cloned/bert/repo/directory"
export GLUE_DIR="/path/to/glue/data/directory"
export BERT_BASE_DIR="/path/to/pre-trained/bert-large/checkpoint/directory"
./tune_save.sh
```

The a saved model will be created in $BERT_REPO_DIR/bert-saved-model/_random_number_/. Where, _random_number_ is a random number generated for every run. Use this saved model to continue with the rest of the demo. 
<a name="appendix2"></a>
## Appendix 2
For all BERT variants, we currently need to augment the standard Neuron compilation process for performance tuning. In the future, we intend to automate this tuning process. This would allow users to use the standard Neuron compilation process, which requires only a one line change in user source code. The standard compilation process is described [here](https://github.com/aws/aws-neuron-sdk/blob/master/docs/tensorflow-neuron/tutorial-compile-infer.md#step-3-compile-on-compilation-instance).

The augmented Neuron compilation process is encapsulated by the bert_model.py script, which performs the following things :
1. Define a Neuron compatible implementation of BERT-Large. For inference, this is functionally equivalent to the open source BERT-Large. The changes needed to create a Neuron compatible BERT-Large implementation is described in [Appendix 3](#appendix3).
2. Extract BERT-Large weights from the open source saved model pointed to by --input_saved_model and associates it with the Neuron compatible model
3. Invoke TensorFlow-Neuron to compile the Neuron compatible model for Inferentia using the newly associated weights
4. Finally, the compiled model is saved into the location given by --output_saved_model
<a name="appendix3"></a>
## Appendix 3
The Neuron compatible implementation of BERT-Large is functionally equivalent to the open source version when used for inference. However, the detailed implementation does differ and here are the list of changes :

1. Data Type Casting : If the original BERT-Large an FP32 model, bert_model.py contains manually defined cast operators to enable mixed-precision. FP16 is used for multi-head attention and fully-connected layers, and fp32 everywhere else. This will be automated in a future release.
2. Remove Unused Operators: A model typically contains training operators that are not used in inference, including a subset of the reshape operators. Those operators do not affect inference functionality and have been removed.
3. Reimplementation of Selected Operators : A number of operators (mainly mask operators), has been reimplemented to bypass a known compiler issue. This will be fixed in a planned future release. 
4. Manually Partition Embedding Ops to CPU : The embedding portion of BERT-Large has been partitioned manually to a subgraph that is executed on the host CPU, without noticable performance impact. In near future, we plan to implement this through compiler auto-partitioning without the need for user intervention.
