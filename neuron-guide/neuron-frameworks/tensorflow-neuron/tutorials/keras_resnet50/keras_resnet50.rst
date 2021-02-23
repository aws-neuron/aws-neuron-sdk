.. _tensorflow-keras-resnet50:

ResNet-50 optimization example
==============================

The following example shows how to compile a FP16 ResNet50 network using
various batching parameters to find the optimal solution. For this
tutorial, please use an ``inf1.6xlarge`` instance, which has 16
NeuronCores.

First step is to setup ``c5.4xlarge`` for compilation as described in
steps 1 and 2 of :ref:`tensorflow-resnet50`. Repeat the setup steps for
``inf1.6xlarge`` instance. In addition, install Neuron Runtime on Inf1
following the steps here: :ref:`rtd-getting-started` Also, download the
example source on both ``c5.4xlarge`` and ``inf1.6xlarge``:

.. code:: bash

   git clone https://github.com/aws/aws-neuron-sdk
   cd ~/aws-neuron-sdk/src/examples/tensorflow/keras_resnet50/

.. _compilation-on-c54xlarge:

Compilation on ``c5.4xlarge``
-----------------------------

On ``c5.4xlarge``, run compilation by following the steps below.

Extract Keras ResNet50 FP32 (resnet50_fp32_keras.pb will be generated):

.. code:: bash

   python gen_resnet50_keras.py

Optimize the extracted Keras ResNet50 FP32 graph for inference before
casting (resnet50_fp32_keras_opt.pb will be generated) with the
following transformations to the graph:

::

   * Remove Identity and CheckNumerics nodes
   * Fold FusedBatchNorm constants into previous Conv2D weights
   * Fold other constants
   * Strip unused nodes
   * Sort by execution order

.. code:: bash

   python optimize_for_inference.py --graph resnet50_fp32_keras.pb --out_graph resnet50_fp32_keras_opt.pb

Convert full graph to FP16 (resnet50_fp16_keras_opt.pb will be
generated):

.. code:: bash

   python fp32tofp16.py  --graph resnet50_fp32_keras_opt.pb --out_graph resnet50_fp16_keras_opt.pb

Run the compilation script sweep_all to sweep through various batch
sizes up to 5 and several NeuronCore Group sizes up to 16. The script
calls the compilation script pb2sm_compile.py which tries to perform
compilation. Some error messages are expected due to known issues (see
Known Issues section below). This step takes about 80 minutes on
``c5.4xlarge``.

.. code:: bash

   rm -f *.zip
   time ./full_sweep

The sweep takes about 1 hour 20 minutes. The sweep results are logged in
full_sweep.log:

.. code:: bash

   *** Batch size 1, num NeuronCores 1 (input shape: (1, 224, 224, 3), saved model dir: rn50_fp16_compiled_b1_nc1) ***

   INFO: Compilation finished in 95 seconds with 99.5% operations placed on Inferentia

   1

   *** Batch size 1, num NeuronCores 2 (input shape: (1, 224, 224, 3), saved model dir: rn50_fp16_compiled_b1_nc2) ***

   INFO: Compilation finished in 95 seconds with 99.5% operations placed on Inferentia

   1

   *** Batch size 1, num NeuronCores 4 (input shape: (1, 224, 224, 3), saved model dir: rn50_fp16_compiled_b1_nc4) ***

   INFO: Compilation finished in 95 seconds with 99.5% operations placed on Inferentia

   1

   ... (outputs removed)

   *** Batch size 5, num NeuronCores 16 (input shape: (5, 224, 224, 3), saved model dir: rn50_fp16_compiled_b5_nc16) ***

   ERROR: Compilation finished in 120 seconds with less than 50% operations placed on Inferentia (0.0%)

   INFO: Retry compilation without static weights

   ERROR: Retry compilation finished in 137 seconds with less than 50% operations placed on Inferentia (0.0%)

   0

The file full_sweep_results.txt shows a summary of the sweep results
with latest Neuron 1/27/20 release (0 means compilation unsuccessful and
0 ops mapped to Inferentia, 1 means most ops mapped to Inferentia and
non-static weights, 2 means most ops mapped to Inferentia and using
static weights):

.. code:: bash

   batch, nc1, nc2, nc4, nc8, nc12, nc16
   1, 1, 1, 1, 2, 2, 2
   2, 1, 1, 0, 1, 2, 2
   3, 1, 1, 1, 1, 1, 1
   4, 1, 1, 0, 1, 1, 1
   5, 1, 1, 0, 0, 0, 0

The compiled saved models are zipped as
``rn50_fp16_compiled_bB_ncN.zip``\ where B marks the compiled batch size
and N marks the number of NeuronCores to target. Copy them to the Inf1
instance that was setup previously and unzip them in the
``~/aws-neuron-sdk/src/examples/tensorflow/keras_resnet50/`` directory.

.. _inference-on-inf16xlarge:

Inference on ``inf1.6xlarge``
-----------------------------

Run inference over different batch sizes to obtain throughput and
latency results for ResNet50 replicated on four NeuronCores. To apply
dynamic batching, the user batch size is set to 10x the compiled batch
size, in order to keep input queue full and to amortize
framework-to-Neuron overhead.

.. code:: bash

   pip install pillow # Necessary for loading images
   cd ~/aws-neuron-sdk/src/examples/tensorflow/keras_resnet50/
   echo "" > batch.log
   for i in $(seq 1 5); do python infer_resnet50_keras_loadtest.py --batch_size=$i | tee -a batch.log; done

The file batch.log now contains the results for each batch size.

.. note::

   the results are based on Neuron 1/27/20 release. These will continue
   improve as we increase Neuron performance.

.. code:: bash

   *** Compiled batch size 1, user batch size 10, num NeuronCores 1 (input shape: (10, 224, 224, 3), saved model dir: ./rn50_fp16_compiled_b1_nc1/1) ***

   Instance type inf1.6xlarge with 16 NeuronCores
   NEURON_MAX_NUM_INFERS (env): 2
   NEURONCORE_GROUP_SIZES (env): 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
   NUM THREADS:  32
   NUM_LOOPS_PER_THREAD:  100
   USER_BATCH_SIZE:  10
   Throughput values collected:
   [3110, 3120, 3100, 3080, 3140, 3120, 3130, 3110]

   Compiled batch size 1, user batch size 10, throughput stats (images/sec): max=3140 p99=3139 p50=3115, avg latency 105.3192 sec/user-batch

   *** Compiled batch size 2, user batch size 20, num NeuronCores 1 (input shape: (20, 224, 224, 3), saved model dir: ./rn50_fp16_compiled_b2_nc1/1) ***

   Instance type inf1.6xlarge with 16 NeuronCores
   NEURON_MAX_NUM_INFERS (env): 2
   NEURONCORE_GROUP_SIZES (env): 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
   NUM THREADS:  32
   NUM_LOOPS_PER_THREAD:  100
   USER_BATCH_SIZE:  20
   Throughput values collected:
   [5160, 5200, 5140, 5080, 5120, 5180, 5120, 5120, 5160, 5240]

   Compiled batch size 2, user batch size 20, throughput stats (images/sec): max=5240 p99=5236 p50=5150, avg latency 127.9041 sec/user-batch

   *** Compiled batch size 3, user batch size 30, num NeuronCores 1 (input shape: (30, 224, 224, 3), saved model dir: ./rn50_fp16_compiled_b3_nc1/1) ***

   Instance type inf1.6xlarge with 16 NeuronCores
   NEURON_MAX_NUM_INFERS (env): 2
   NEURONCORE_GROUP_SIZES (env): 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
   NUM THREADS:  32
   NUM_LOOPS_PER_THREAD:  100
   USER_BATCH_SIZE:  30
   Throughput values collected:
   [6030, 5670, 5940, 5820, 5850, 6090, 6000, 6120, 5820, 6180, 5790, 5820, 5790, 5760, 5790]

   Compiled batch size 3, user batch size 30, throughput stats (images/sec): max=6180 p99=6171 p50=5820, avg latency 164.8427 sec/user-batch

   *** Compiled batch size 4, user batch size 40, num NeuronCores 1 (input shape: (40, 224, 224, 3), saved model dir: ./rn50_fp16_compiled_b4_nc1/1) ***

   Instance type inf1.6xlarge with 16 NeuronCores
   NEURON_MAX_NUM_INFERS (env): 2
   NEURONCORE_GROUP_SIZES (env): 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
   NUM THREADS:  32
   NUM_LOOPS_PER_THREAD:  100
   USER_BATCH_SIZE:  40
   Throughput values collected:
   [6080, 6280, 6320, 6040, 6200, 6360, 6440, 6120, 6280, 6360, 6200, 5880, 6240, 5960, 6160, 6040, 6120, 6240, 6320]

   Compiled batch size 4, user batch size 40, throughput stats (images/sec): max=6440 p99=6425 p50=6200, avg latency 209.3087 sec/user-batch

   *** Compiled batch size 5, user batch size 50, num NeuronCores 1 (input shape: (50, 224, 224, 3), saved model dir: ./rn50_fp16_compiled_b5_nc1/1) ***

   Instance type inf1.6xlarge with 16 NeuronCores
   NEURON_MAX_NUM_INFERS (env): 2
   NEURONCORE_GROUP_SIZES (env): 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
   NUM THREADS:  32
   NUM_LOOPS_PER_THREAD:  100
   USER_BATCH_SIZE:  50
   Throughput values collected:
   [6350, 6300, 6400, 6450, 6400, 6350, 6450, 6350, 6450, 6150, 6200, 6550, 6550, 6450, 6550, 6400, 6550, 6400, 6350, 6350, 6500, 6550, 6300]

   Compiled batch size 5, user batch size 50, throughput stats (images/sec): max=6550 p99=6550 p50=6400, avg latency 251.6603 sec/user-batch

Known Issues
~~~~~~~~~~~~

Unable to compile with batch and num NeuronCores combination
------------------------------------------------------------

For some combination of batch and number of NeuronCores setting, you may
see an internal compiler error as below. Please see the sweep result
above for Neuron 1/27/20 release. Furthermore, if using auto-casting to
bfloat16 from FP32 network and batch size is larger than 1 would result
in the same error.

.. code:: bash

   INFO:tensorflow:fusing subgraph neuron_op_a73aed4b95ca5d5b with neuron-cc; log file is at /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.neuron-cc.log
   WARNING:tensorflow:Failed to fuse subgraph neuron_op_a73aed4b95ca5d5b with '/home/ubuntu/test_venv/bin/neuron-cc compile /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.pb --framework TENSORFLOW --pipeline compile SaveTemps --output /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.neff --io-config "{\"inputs\": {\"input_10/_0:0\": [[6, 224, 224, 3], \"float16\"]}, \"outputs\": [\"probs/Softmax:0\"]}" --batching_en --rematerialization_en --sb_size 120 --spill_dis --enable-replication True'
   WARNING:tensorflow:neuron-cc error message:
   WARNING:tensorflow:01/23/2020 01:15:40 AM ERROR [neuron-cc]: ***************************************************************
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:  An Internal Compiler Error has occurred
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: ***************************************************************
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: Please contact Customer Support and provide the following details.
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: Error message:  Non-zero exit status (134) for command: /home/ubuntu/test_venv/lib/python3.6/site-packages/neuroncc/starfish/bin/list_sch --hhir hh-tr-external-move.json --verbose 0 --sb_size 120 --arith_intensity_target 2300 --sb_watermark_low 0.250000 --sb_watermark_high 0.750000 --sb_size_tol 1 --alloc simple1 --alloc_opt --depth_diff 0.100000 --verbose_start_cycle 0 --tt_dist --mm_meet_cnt 1 --load_speed_factor 0.300000 --schir sch_tmp.json --spill_depth_limit 5 --spill_dis --true_dep --mm_order --batching_en --rematerialization_en
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: Error class:    CompilerInternalError
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: Error location: job.Scheduler.3
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: Command line:   /home/ubuntu/test_venv/bin/neuron-cc compile /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.pb --framework TENSORFLOW --pipeline compile SaveTemps --output /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.neff --io-config '{"inputs": {"input_10/_0:0": [[6, 224, 224, 3], "float16"]}, "outputs": ["probs/Softmax:0"]}' --batching_en --rematerialization_en --sb_size 120 --spill_dis --enable-replication True
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: Internal details:
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:   File "neuroncc/driver/Job.py", line 207, in neuroncc.driver.Job.runSingleInputFn
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:   File "neuroncc/driver/jobs/Scheduler.py", line 58, in neuroncc.driver.jobs.Scheduler.Scheduler.runSingleInput
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:   File "neuroncc/driver/Job.py", line 145, in neuroncc.driver.Job.Job.shellCommand
   01/23/2020 01:15:40 AM ERROR [neuron-cc]:
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: Version information:
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   Neuron Compiler version 1.0.6632.0+6001610955
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   HWM version 1.0.839.0-6001300654
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   NEFF version 0.6
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   TVM version 1.0.1589.0+6001610955
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   NumPy version 1.16.5
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   MXNet not available
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:   TF version 1.15.0
   01/23/2020 01:15:41 AM ERROR [neuron-cc]:
