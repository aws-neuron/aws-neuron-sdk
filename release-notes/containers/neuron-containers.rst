.. _neuron-containers-release-notes:

Neuron Containers Release Notes
===============================

.. contents:: Table of contents
   :local:
   :depth: 1


Neuron 2.5.0
-------------

Date: 11/07/2022

- Neuron now supports trn1-based training in Sagemaker and Deep Learning Containers using PyTorch.  Find Neuron DLC containers here: https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers


Neuron 2.4.0
-------------

Date: 10/27/2022

- Neuron now supports Kubernetes work scheduling at the level of NeuronCore.  Updates on how to use the new core allocation method is captured in the Kubernetes documentation on this site.

Neuron 2.3.0
-------------

Date: 10/10/2022

- Now supporting TRN1 and INF1 EC2 instance types as part of Neuron.  There is an optional aws-neuronx-oci-hooks package users may install for conveince that supports use of the AWS_NEURON_VISIBLE_DEVICES environment variable when launching containers.  New DLC containers will be coming soon in support of training workloads on TRN1.

Neuron 1.19.0
-------------

Date: 04/29/2022

- Neuron Kubernetes device driver plugin now can figure out communication with the Neuron driver without the *oci hooks*.  Starting with *Neuron 1.19.0* release, installing ``aws-neuron-runtime-base`` and ``oci-add-hooks`` are no longer a requirement for Neuron Kubernetes device driver plugin.

Neuron 1.16.0
-------------

Date: 10/27/2021

New in this release
^^^^^^^^^^^^^^^^^^^

-  Starting with Neuron 1.16.0, use of Neuron ML Frameworks now comes
   with an integrated Neuron Runtime as a library, as a result it is
   no longer needed to deploy ``neuron-rtd``. Please visit :ref:`neuron-containers` for more
   information.
-  When using containers built with components from Neuron 1.16.0, or
   newer, please use ``aws-neuron-dkms`` version 2.1 or newer and the
   latest version of ``aws-neuron-runtime-base``. Passing additional
   system capabilities is no longer required.




