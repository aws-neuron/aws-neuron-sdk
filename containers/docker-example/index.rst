Example: Run containerized neuron application
=============================================

Introduction:
-------------

With this example you will learn how to run a Neuron application using
docker containers.

Prerequisites:
--------------

-  Please ensure the steps from the guide on :ref:`tensorflow-serving`
   were completed successfully before continuing.

Steps:
------

Step 1: Start neuron-rtd container:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may choose to use the following neuron-rtd image:
[790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest], or
build your own image as shown in :ref:`neuron-runtime-dockerfile`.

Run neuron-rtd container as shown below. A volume must be mounted to
:/sock where neuron-rtd will open a UDS socket. The application can
interact with runtime using this socket.

.. code:: bash

   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 790709498068.dkr.ecr.us-east-1.amazonaws.com
 
   docker pull 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:1.1.1402.0
   docker tag 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:1.1.1402.0 neuron-rtd
   mkdir /tmp/neuron_rtd_sock
   chmod o+rwx /tmp/neuron_rtd_sock
   docker run --device=/dev/neuron0 --cap-add IPC_LOCK -v /tmp/neuron_rtd_sock/:/sock -it neuron-rtd
   
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   If using older version of neuorn(below 1.1):
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   docker pull 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:1.0.9592.0
   docker tag 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:1.0.9592.0 neuron-rtd
   mkdir /tmp/neuron_rtd_sock
   chmod o+rwx /tmp/neuron_rtd_sock
   docker run --env AWS_NEURON_VISIBLE_DEVICES="0" --cap-add SYS_ADMIN --cap-add IPC_LOCK -v /tmp/neuron_rtd_sock/:/sock -it neuron-rtd

Step 2: Start application (tensorflow serving) container:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build tensorflow-model-server-neuron image using provided example
dockerfile :ref:`tensorflow-model-server-neuron-dockerfile`.

Run assuming a compiled saved model was stored in s3:///my_model/

.. code:: bash


   # Note: the neuron-rtd socket directory must be mounted and pointed at using environment variable.
   #       TensorFlow serving will use that socket to talk to Neuron-rtd
   docker run --env NEURON_RTD_ADDRESS=unix:/sock/neuron.sock \
              -v /tmp/neuron_rtd_sock/:/sock \
              -p 8501:8501 \
              -p 8500:8500 \
              --env MODEL_BASE_PATH=s3://<my-bucket>/my_model/ \
              --env MODEL_NAME=my_model
              tensorflow-model-server-neuron

Step 3: Verify by running an inference!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As shown in :ref:`tensorflow-serving`
