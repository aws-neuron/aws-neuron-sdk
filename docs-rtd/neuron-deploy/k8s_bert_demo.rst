.. _example-deploy-bert-as-k8s-service:

Example: Deploy BERT as a k8s service
=====================================

Introduction
------------

This tutorial uses BERT model as a teaching example on how to deploy an
inference application using Kubernetes on the Inf1 instances.

Prerequisite:
^^^^^^^^^^^^^

-  tutorial-k8s.md: to setup k8s support on your cluster.
-  Inf1 instances as worker nodes with attached roles allowing:

   -  ECR read access policy to retrieve container images from ECR:
      **arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly**
   -  S3 access to retrieve saved_model from within tensorflow serving
      container.

Step 1: Build an example tensorflow serving container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the following dockerfile: :download:`tensorflow-model-server-neuron Dockerfile <docker-example/Dockerfile.tf-serving>`

.. code:: bash

   docker build . -f Dockerfile.tf-serving  -t tf-serving-ctr

Step 2: Compile and place your saved model in an S3 bucket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow this step in BERT example below to compile BERT into a saved
model:

:ref:`compiling-neuron-compatible-bert-large` Section: Compile open source BERT-Large saved model using Neuron
compatible BERT-Large implementation*

The following instructions assume that the saved model is in s3 bucket
*s3:///* as following: *s3:///bert/1/saved_model.pb*

.. _step-3-deploy-bert_serviceyml:

Step 3: Deploy *bert_service.yml*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get a local copy of [./bert_service.yml]; inspect, modify; then, apply
to your cluster.

The example service described in the manifest has two containers in a
pod: The inference serving container and neuron-rtd container. The two
containers talk over a unix domain socket placed in shared mounted
volume. The *neuron-rtd* container requires elevated privileges to
access Inferentia device, hence the following capability must be
provided in the manifest: CAP_SYS_ADMIN, CAP_IPC_LOCK. Neuron-rtd will
drop those capabilities at init time, before opening a GRPC socket. By
default, neuron-rtd will attempt to preallocate 128 2MB hugepages per
Inferentia on start up. The example application uses one Inferentia
device per container, if more are required, the amount of required
hugepages needs to be adjusted in the manifest.

Modify the manifest to point at your own S3 bucket instead of: s3:://
Apply manifest to your cluster.

.. code:: bash

   kubectl apply -f bert_service.yml

Step 4: Run some inferences!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forward gRPC port to the *inf-k8s-test* service:

.. code:: bash

   kubectl port-forward svc/inf-k8s-test 9000:9000 & 

Run the provided [./bert_client.py]

.. code:: bash

   python3 bert_client.py

Result:

::

   WARNING:tensorflow:
   The TensorFlow contrib module will not be included in TensorFlow 2.0.
   For more information, please see:
     * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
     * https://github.com/tensorflow/addons
     * https://github.com/tensorflow/io (for I/O related ops)
   If you depend on functionality not listed there, please file an issue.

   Handling connection for 9000
   Inference successful: 0
   Inference successful: 1
   Inference successful: 2
   Inference successful: 3
   Inference successful: 4
   Inference successful: 5
   Inference successful: 6
   Inference successful: 7
   Inference successful: 8
   Inference successful: 9
   Inference successful: 10
   Inference successful: 11
   ...
