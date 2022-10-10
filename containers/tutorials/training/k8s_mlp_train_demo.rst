.. _example-deploy-mlp-train-pod:

Deploy a simple mlp training script as a Kubernetes job
----------------------------------------------------------

This tutorial uses mlp train as a teaching example on how to deploy an
training application using Kubernetes on the Trn1 instances.

Prerequisite:
^^^^^^^^^^^^^

-  :ref:`tutorial-k8s-env-setup-for-neuron`: to setup k8s support on your cluster.
-  Trn1 instances as worker nodes with attached roles allowing:

   -  ECR read access policy to retrieve container images from ECR:
      **arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly**
- Have a container image that is build using :ref:`tutorial-training`

Deploy a mlp training image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a file named `mlp_train.yaml` with the contents below\. 

.. note::
   In the image:  add the appropriate location of the image


::

  apiVersion: v1
  kind: Pod
  metadata:
    name: trn1-mlp
  spec:
    restartPolicy: Never
    schedulerName: default-scheduler
    hostNetwork: true
    nodeSelector:
      beta.kubernetes.io/instance-type: trn1.32xlarge
      beta.kubernetes.io/instance-type: trn1.2xlarge
    containers:
      - name: trn1-mlp
        command: ["/usr/local/bin/python3"]
        args:  ["/opt/ml/mlp_train.py"]
        image: 647554078242.dkr.ecr.us-east-1.amazonaws.com/sunda-pt:k8s_mlp_0907
        imagePullPolicy: IfNotPresent
        env:
        - name: NEURON_RT_LOG_LEVEL
          value: "INFO"
        resources:
          limits: 
            aws.amazon.com/neuron: 2
          requests:
            aws.amazon.com/neuron: 2

2. Deploy the pod.

::

   kubectl apply -f mlp_train.yaml

3. Check the logs to make sure training completed
::

   kubectl logs <pod name>

   Your log should have the following

::

  Final loss is 0.1977
  ----------End Training ---------------
