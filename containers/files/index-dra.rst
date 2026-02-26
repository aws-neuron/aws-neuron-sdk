.. meta::
    :description: Templates supporting AWS Neuron Dynamic Resource Allocation (DRA) on Kubernetes.
    :keywords: AWS Neuron, Neuron DRA, Dynamic Resource Allocation, Kubernetes, K8s, Device Plugin
    :date-modified: 02/05/2026

AWS Neuron Dynamic Resource Allocation (DRA) on Kubernetes: Support files
=========================================================================

This directory contains templates supporting AWS Neuron Dynamic Resource Allocation (DRA) on Kubernetes. You can view and download these files from the links below.

**Directory structure**:

.. code-block:: text

   containers/files/
              └── specs/
                  ├── 1x4-connected-devices.yaml
                  ├── 2-node-inference-us.yaml
                  ├── 4-node-inference-us.yaml
                  ├── all-devices.yaml
                  ├── lnc-setting-trn2.yaml
                  ├── specific-driver-version.yaml
                  └── us-and-lnc-config.yaml

Resource Claim Specifications
-----------------------------

Example resource claim templates and pod specifications demonstrating different Neuron device allocation patterns for various workload requirements.

.. list-table::
   :header-rows: 1
   :widths: 30 55 15

   * - File Name
     - Description
     - Download
   * - 1x4-connected-devices.yaml
     - Resource claim template for allocating 4 connected Neuron devices with topology constraints for optimal performance.
     - :download:`Download <specs/1x4-connected-devices.yaml>`
   * - 2-node-inference-us.yaml
     - Multi-node inference configuration for distributed workloads across 2 Trainium nodes.
     - :download:`Download <specs/2-node-inference-us.yaml>`
   * - 4-node-inference-us.yaml
     - Large-scale inference setup for distributed workloads spanning 4 Trainium nodes.
     - :download:`Download <specs/4-node-inference-us.yaml>`
   * - all-devices.yaml
     - Resource claim template that allocates all available Neuron devices on a trn2.48xlarge instance.
     - :download:`Download <specs/all-devices.yaml>`
   * - lnc-setting-trn2.yaml
     - Logical NeuronCore configuration template optimized for Trainium2 instances.
     - :download:`Download <specs/lnc-setting-trn2.yaml>`
   * - specific-driver-version.yaml
     - Example configuration for requesting specific Neuron driver versions in resource claims.
     - :download:`Download <specs/specific-driver-version.yaml>`
   * - us-and-lnc-config.yaml
     - Example configuration for requesting UltraServer node with Logical NeuronCore configuration.
     - :download:`Download <specs/us-and-lnc-config.yaml>`

