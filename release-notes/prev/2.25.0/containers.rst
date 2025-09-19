.. _neuron-2-25-0-dlc:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Deep Learning Containers (DLC) component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: Neuron Deep Learning Containers release notes
====================================================================

**Date of release**: July 31, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`

Improvements
------------

* All Neuron packages and their dependencies have been upgraded to support vAWS Neuron SDK version 2.25.0.
* The ``pytorch-inference-vllm-neuronx`` Deep Learning Container has been upgraded to version ``0.9.1``.

Known issues
------------

*Something doesn't work. Check here to find out if we already knew about it. We hope to fix these soon!*

* ``pytorch-training-neuronx`` 2.7.0 DLC has two HIGH CVEs related to ``sagemaker-python-sdk`` package. We are actively working to resolve these high CVEs:
- * `CVE-2024-34072 <https://nvd.nist.gov/vuln/detail/CVE-2024-34072>`_
- * `CVE-2024-34073 <https://nvd.nist.gov/vuln/detail/CVE-2024-34073>`_
* ``pytorch-inference-vllm-neuronx`` 0.9.1 DLC has CRITICAL and HIGH CVEs . We are actively working to resolve these high CVEs:
- * `CVE-2024-35515 <https://nvd.nist.gov/vuln/detail/CVE-2024-35515>`_
- * `CVE-2022-4296 <https://nvd.nist.gov/vuln/detail/CVE-2022-42969>`_