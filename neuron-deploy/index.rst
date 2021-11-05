.. _neuron-containers:

Containers
==========

It is recommended to deploy Neuron inside a preconfigured `Deep
Learning Container (DLC) <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/getting-started.html>`_ from AWS. Running Neuron inside a container on Inf1 requires Docker version 18 (or newer)
and a base AMI with aws-neuron-runtime-base and aws-neuron-dkms installed. 
It's possible to also use a Neuron container on any instance type without the base 
and dkms package, but this is typically limited to compilation and development 
when running on instances without a Inf1 Device (inferentia). DLC images for Neuron can be obtained from `here <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-inference-containers>`_.


Documentation is organized based on the target deployment environment
and use case.  In most cases, it is recommended to use a preconfigured
`Deep Learning Container <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/what-is-dlc.html>`_ from AWS.  Each DLC is pre-configured to have a recent
version of Neuron components installed and is specific to the chosen ML Framework you want.


.. _containers_quickstart:

Quick Start
-----------

.. toctree::
   :maxdepth: 1

   tutorials/tutorial-docker-env-setup
   tutorials/neuron-container
   

Deploy a Neuron Container
-------------------------

.. toctree::
   :maxdepth: 1

   dlc-then-ec2-devflow
   dlc-then-ecs-devflow
   dlc-then-eks-devflow
   container-sm-hosting-devflow


.. _containers-tutorials:

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorials/k8s-neuron-scheduler

Release Notes
-------------

.. toctree::
   :maxdepth: 1

   rn

Resources for Neuron Runtime 1.x Users
--------------------------------------

.. toctree::
   :maxdepth: 1

   v1/index



