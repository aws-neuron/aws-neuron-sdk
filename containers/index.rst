.. _neuron_containers:

Neuron Containers
=================

.. toctree::
    :maxdepth: 1

    Quickstart: Deploy a DLC with vLLM </containers/get-started/quickstart-configure-deploy-dlc>
    /containers/getting-started
    /containers/locate-neuron-dlc-image
    /containers/dlc-then-customize-devflow
    /containers/neuron-plugins
    /containers/how-to/how-to-ultraserver
    /containers/faq


In this section, you'll find resources to help you use containers for accelerating your deep learning models on Inferentia and Trainium instances.

----

.. grid:: 1
  :gutter: 1

  .. grid-item-card:: Quickstart: Configure and deploy a Deep Learning Container (DLC) with vLLM
    :link: quickstart_vllm_dlc_deploy
    :link-type: ref

    **Quickstart: Configure and deploy a DLC with vLLM**
    ^^^
    Get started by configuring and deploying a Deep Learning Container (DLC) with the AWS Neuron SDK and vLLM.
    +++
    Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``

----

Getting started with Neuron DLCs using Docker
---------------------------------------------
AWS Neuron Deep Learning Containers (DLCs) are a set of Docker images for training and serving models on AWS Trainium
and Inferentia instances using AWS Neuron SDK. To build a Neuron container using Docker, please refer to
:ref:`containers-getting-started`.

Neuron Deep Learning Containers
-------------------------------
In most cases, it is recommended to use a preconfigured `Deep Learning Container (DLC) <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/what-is-dlc.html>`_
from AWS. Each DLC is pre-configured to have all of the Neuron components installed and is specific to the chosen ML
Framework. For more details on Neuron Deep Learning Containers, please refer to :ref:`locate-neuron-dlc-image`.

Customize Neuron DLC
---------------------
Neuron DLC can be customized as needed. To learn more about how to customize  the Neuron Deep Learning Container (DLC)
to fit your specific project needs, please refer to :ref:`containers-dlc-then-customize-devflow`.

Neuron Plugins for Containerized Environments
---------------------------------------------
Neuron provides plugins for better observability and fault tolerance. For more information on the plugins, please refer
to :ref:`neuron-container-plugins`.

How to schedule MPI jobs to run on Neuron UltraServer on EKS
------------------------------------------------------------

Neuron provides Trn2 UltraServers to improve the performance of MPI jobs. For information on how to schedule MPI jobs
on UltraServers in EKS, please refer to :ref:`containers-how-to-ultraserver`.

Neuron Containers FAQ
----------------------
For frequently asked questions and troubleshooting, please refer to :ref:`container-faq`
