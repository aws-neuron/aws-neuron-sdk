.. _dlc-then-eks-devflow:


Deploy DLC on Elastic Kubernetes Service (EKS)
=================

.. contents:: Table of Contents
   :local:
   :depth: 2

   
Description
-----------

|image|
 
.. |image| image:: /images/dlc-on-eks-dev-flow.png
   :width: 750
   :alt: Neuron developer flow for DLC on ECS
   :align: middle

You can use the Neuron version of the `AWS Deep Learning Containers <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-tutorials-inference.html>`_ to run inference on Amazon Elastic Kubernetes Service (EKS). In this developer flow, you set up an EKS cluster with Inf1 instances, create a Kubernetes manifest for your inference service and deploy it to your cluster. This developer flow assumes:

1. The model has already been compiled through :ref:`Compilation with Framework API on EC2 instance <ec2-then-ec2-devflow>` or through :ref:`Compilation with Sagemaker Neo <neo-then-hosting-devflow>`. 

2. You already set up your container to retrieve it from storage.

.. _dlc-then-eks-setenv:

Setup Environment
-----------------

1. Install pre-requisits:
	Follow `these instruction <https://docs.aws.amazon.com/eks/latest/userguide/eksctl.html>`_ to install or upgrade the *eksctl* command line utility on your local computer.

	Follow `these instruction <https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html>`_ to install *kubectl* in the same computer. *kubectl* is a command line tool for working with Kubernetes clusters.

2. Create a Amazon EKS cluster:
	Create a cluster with Inf1 Amazon EC2 instance nodes. You can replace *inf1.2xlarge* with any `Inf1 instance type <http://aws.amazon.com/ec2/instance-types/inf1/)>`_. *eksctl* detects that you are launching a node group with an Inf1 instance type and will start your nodes using one of the `Amazon EKS optimized accelerated Amazon Linux AMI <eks-linux-ami-versions.md#eks-gpu-ami-versions>`_.

	.. code::

		eksctl create cluster \
	       --name <inferentia> \
	       --version <1.16> \
	       --region <region-code> \
	       --nodegroup-name <ng-inf1> \
	       --node-type <inf1.2xlarge> \
	       --nodes <2> \
	       --nodes-min <1> \
	       --nodes-max <4>
	
	.. note::

		You cannot use `IAM roles for the Kubernetes service account <https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html>`_ with TensorFlow Serving. Note the value of the following Instance role output so you can set up *AmazonS3ReadOnlyAccess* IAM policies for your application.  

		.. code::

			[ℹ]  adding identity "arn:aws:iam::<111122223333>:role/eksctl-<inferentia>-<nodegroup-ng-in>-NodeInstanceRole-<FI7HIYS3BS09>" to auth ConfigMap
   
	When launching a node group with Inf1 instances, *eksctl* automatically installs the AWS Neuron Kubernetes device plugin\. This plugin publishes Neuron devices as a system resource to the Kubernetes scheduler, which can be requested by a container\. In addition to the default Amazon EKS node IAM policies, the Amazon S3 read only access policy is added so that the sample applications can load a trained model artifacts from Amazon S3\.

	Make sure that all pods have started correctly\.

	.. code::

		kubectl get pods -n kube-system

	Output
	
	.. code::

		NAME                                   READY   STATUS    RESTARTS   AGE
		aws-node-kx2m8                         1/1     Running   0          5m
		aws-node-q57pf                         1/1     Running   0          5m
		coredns-86d5cbb4bd-56dz2               1/1     Running   0          5m
		coredns-86d5cbb4bd-d6n4z               1/1     Running   0          5m
		kube-proxy-75zx6                       1/1     Running   0          5m
		kube-proxy-plkfq                       1/1     Running   0          5m
		neuron-device-plugin-daemonset-6djhp   1/1     Running   0          5m
		neuron-device-plugin-daemonset-hwjsj   1/1     Running   0          5m

3. (Optional) Set up Amazon S3 Read Only Access for pods.
	Add the *AmazonS3ReadOnlyAccess* IAM policy to the node instance role that was created. This is necessary so that the sample application can load a trained model from Amazon S3.

	.. code::

		aws iam attach-role-policy \
		    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
		    --role-name eksctl-<inferentia>-<nodegroup-ng-in>-NodeInstanceRole-<FI7HIYS3BS09>


You can learn more about deploying an application on EKS using inferentia `here <https://docs.aws.amazon.com/eks/latest/userguide/inferentia-support.html>`_.
