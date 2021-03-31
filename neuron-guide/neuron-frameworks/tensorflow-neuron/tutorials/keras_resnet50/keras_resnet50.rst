.. _tensorflow-keras-resnet50::

Tensorflow-Neuron 1.15 - Keras ResNet-50 Optimization Tutorial
======================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

In this tutorial we will compile and deploy keras_resnet50 model on an Inf1 instance.
To enable faster environment setup, you will run the tutorial on an inf1.6xlarge instance
to enable both compilation and deployment (inference) on the same instance.
The following example shows how to compile a ResNet-50 network using
various batching parameters to find the optimal solution.

.. note::

  Model compilation can be executed on a non-inf1 instance for later deployment. Follow the same EC2 Developer Flow Setup using other instance families and leverage Amazon Simple Storage Service (S3) to share the compiled models between different instances.

.. _tensorflow-keras_resnet50-env-setup:

Setup The Environment
---------------------

Launch Inf1 instance by following the below steps, please make sure to choose an inf1.6xlarge instance.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst

.. _tensorflow-keras_resnet50-run-tutorial:

Run The Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code:: bash

   git clone https://github.com/aws/aws-neuron-sdk
   cd ~/aws-neuron-sdk/src/examples/tensorflow/keras_resnet50/



The Jupyter notebook is available as a file with the name :tensorflow-neuron-src:`keras_resnet50.ipynb <keras_resnet50/keras_resnet50.ipynb>`, you can either run the Jupyter notebook from a browser or run it as a script from terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions


You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb

.. _keras_resnet50-cleanup-instances:

Clean up your instance/s
------------------------

After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.

Known Issues
------------

Unable to compile with batch and num NeuronCores combination


For some combination of batch and number of NeuronCores setting, you may
see an internal compiler error as below. Please see the sweep result
above for Neuron 1/27/20 release. Furthermore, if using auto-casting to
bfloat16 from FP32 network and batch size is larger than 1 would result
in the same error.

.. code:: bash

   INFO:tensorflow:fusing subgraph neuron_op_a73aed4b95ca5d5b with neuron-cc; log file is at /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.neuron-cc.log
   WARNING:tensorflow:Failed to fuse subgraph neuron_op_a73aed4b95ca5d5b with '/home/ubuntu/test_venv/bin/neuron-cc compile /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.pb --framework TENSORFLOW --pipeline compile SaveTemps --output /home/ubuntu/keras_fp16_benchmarking_db/compiler_workdir/neuron_op_a73aed4b95ca5d5b/graph_def.neff --io-config "{\"inputs\": {\"input_10/_0:0\": [[6, 224, 224, 3], \"float16\"]}, \"outputs\": [\"probs/Softmax:0\"]}" --batching_en --rematerialization_en --sb_size 120 --spill_dis --enable-replication True'
   WARNING:tensorflow:neuron-cc error message:
   WARNING:tensorflow:01/23/2020 01:15:40 AM ERROR [neuron-cc]:
   01/23/2020 01:15:40 AM ERROR [neuron-cc]: ***************************************************************
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
