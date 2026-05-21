.. meta::
   :description: Deploy Neuron training and inference workloads directly on Amazon EC2 instances with Trainium and Inferentia hardware.
   :keywords: EC2, Neuron, Trainium, Inferentia, training, inference, DLAMI, deployment
   :date-modified: 04/20/2026

.. _deploy-ec2:
.. _amazon-ec2:

Amazon EC2
==========

Run training and inference workloads directly on Amazon EC2 instances with Neuron hardware. Use a :doc:`Deep Learning AMI </deploy/environments/dlami>` for the fastest setup, or install the Neuron SDK manually on a supported base AMI.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Train on EC2
      :link: /deploy/ec2/training
      :link-type: doc
      :class-card: sd-border-1

      Set up a Trn1 instance with a Deep Learning AMI and run your first training job using PyTorch.

   .. grid-item-card:: Run inference on EC2 (Inf2, Trn1)
      :link: /deploy/ec2/inference-inf2
      :link-type: doc
      :class-card: sd-border-1

      Compile and deploy models for inference on Inf2 and Trn1 instances.

   .. grid-item-card:: Run inference with DLC on EC2
      :link: /deploy/ec2/inference-dlc
      :link-type: doc
      :class-card: sd-border-1

      Pull a Neuron Deep Learning Container and run inference on EC2 with Docker.

   .. grid-item-card:: Run inference on EC2 (Inf1)
      :link: /deploy/ec2/inference
      :link-type: doc
      :class-card: sd-border-1

      Compile and deploy models for inference on Inf1 instances.

.. toctree::
   :maxdepth: 1
   :hidden:

   Train on EC2 </deploy/ec2/training>
   Inference (Inf2, Trn1) </deploy/ec2/inference-inf2>
   Inference with DLC </deploy/ec2/inference-dlc>
   Inference (Inf1) </deploy/ec2/inference>
