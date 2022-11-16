*  Working kubernetes cluster
*  Inf1/Trn1 instances as worker nodes with attached roles allowing:
   *  ECR read access policy to retrieve container images from ECR: **arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly**
*  :ref:`tutorial-docker-env-setup`: to install required packages in the worker nodes.
   With EKS, the `EKS optimized accelarated AMI <https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html#gpu-ami>`_ has the necessary neuron components installed
*  Kubernetes node object has instance-type set to inf1/trn1 types.
   For ex, ``"node.kubernetes.io/instance-type": "inf1.2xlarge"`` or
   ``"node.kubernetes.io/instance-type": "trn1.2xlarge"``