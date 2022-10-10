.. _neuron-devflows:

Developer Flows
===============
Neuron can be used in a wide selection of development flows. Each flow has its own starting point and requirements which are required to enable deep learning acceleration with AWS Neuron.

.. grid:: 2

   .. dropdown::  Deploy Containers with Neuron
         :class-title: sphinx-design-class-title-med
         :class-body: sphinx-design-class-body-small
         :animate: fade-in

         .. tab-set:: 

               .. tab-item:: Inference and Training

                  .. toctree::
                     :maxdepth: 1

                     /containers/index

   .. dropdown::  AWS EC2
         :class-title: sphinx-design-class-title-med
         :class-body: sphinx-design-class-body-small
         :animate: fade-in

         .. tab-set:: 

               .. tab-item:: Inference

                  .. toctree::
                     :maxdepth: 1

                     /general/devflows/inference/ec2-then-ec2-devflow

         .. tab-set:: 

               .. tab-item:: Training

                  .. toctree:: 
                     :maxdepth: 1

                     /general/devflows/training/ec2/ec2-training

   .. dropdown::  Amazon EKS
         :class-title: sphinx-design-class-title-med
         :class-body: sphinx-design-class-body-small
         :animate: fade-in

         .. tab-set:: 

               .. tab-item:: Inference

                  .. toctree:: 
                     :maxdepth: 1

                     /general/devflows/inference/dlc-then-eks-devflow


         .. tab-set:: 

               .. tab-item:: Training

                  .. note::

                     Amazon EKS support is coming soon.



   .. dropdown::  Amazon ECS
         :class-title: sphinx-design-class-title-med
         :class-body: sphinx-design-class-body-small
         :animate: fade-in

         .. tab-set:: 

               .. tab-item:: Inference

                  .. toctree:: 
                     :maxdepth: 1

                     /general/devflows/inference/dlc-then-ecs-devflow


         .. tab-set:: 

               .. tab-item:: Training

                  .. note::

                     Amazon ECS supports Trn1.

                     An example of how to train a model with Neuron using ECS is coming soon.

   .. dropdown::  AWS Sagemaker
         :class-title: sphinx-design-class-title-med
         :class-body: sphinx-design-class-body-small
         :animate: fade-in

         .. tab-set:: 

               .. tab-item:: Inference

                  .. toctree:: 
                     :maxdepth: 1

                     /general/devflows/inference/neo-then-hosting-devflow
                     /general/devflows/inference/byoc-hosting-devflow 

         .. tab-set:: 

               .. tab-item:: Training

                  .. note::

                     AWS Sagemaker support is coming soon.

   .. dropdown::  AWS ParallelCluster
         :class-title: sphinx-design-class-title-med
         :class-body: sphinx-design-class-body-small
         :animate: fade-in


         .. tab-set:: 

               .. tab-item:: Training

                  .. toctree::
                     :maxdepth: 1

                     /general/devflows/training/parallelcluster/parallelcluster-training
                     

         .. tab-set:: 

               .. tab-item:: Inference

                  .. note::

                     AWS ParallelCluster support is coming soon.





   .. dropdown::  AWS Batch
         :class-title: sphinx-design-class-title-med
         :class-body: sphinx-design-class-body-small
         :animate: fade-in

         .. tab-set:: 

               .. tab-item:: Inference

                  .. note::

                     AWS Batch supports Inf1.

                     An example of how to deploy a model with Neuron using Batch is coming soon.

         .. tab-set:: 

               .. tab-item:: Training

                  .. note::

                     AWS Batch support is coming soon.




