.. _neuron-containers-release-notes:

Neuron Containers Release Notes
===============================

.. contents::
   :local:
   :depth: 1


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




