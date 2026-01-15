.. meta::
    :description: Scripts, manifests, and templates supporting AWS Neuron Direct Resource Allocation (DRA) on Kubernetes.
    :keywords: AWS Neuron, Neuron DRA, Direct Resource Allocation, Kubernetes, K8s, Device Plugin
    :date-modified: 01/14/2026

AWS Neuron Direct Resource Allocation (DRA) on Kubernetes: Support files
=========================================================================

This directory contains scripts, manifests, and templates supporting AWS Neuron Direct Resource Allocation (DRA) on Kubernetes. You can view and download these files from the links below.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-card: sd-border-2

      **Download the scripts and YAML files as a TAR/GZIP archive**
      ^^^
      :download:`Neuron DRA support files as .tar.gz </containers/files/neuron-dra-beta-reinvent.tar.gz>`

Preserve the directory structure when you extract the archive. The driver installation script uses this relative folder structure to find the corresponding YAML files.

**Directory structure**:

.. code-block:: text

   containers/files/
              ├── manifests/
              │   ├── clusterrole.yaml
              │   ├── clusterrolebinding.yaml
              │   ├── daemonset.yaml
              │   ├── deviceclass.yaml
              │   ├── namespace.yaml
              │   └── serviceaccount.yaml
              └── examples/
                  ├── scripts/
                  │   └── install-dra-driver.sh
                  └── specs/
                      ├── 1x4-connected-devices.yaml
                      ├── 2-node-inference-us.yaml
                      ├── 4-node-inference-us.yaml
                      ├── all-devices.yaml
                      ├── lnc-setting-trn2.yaml
                      └── specific-driver-version.yaml

Installation Scripts
--------------------

These scripts automate the deployment and configuration of the Neuron DRA driver on your Kubernetes cluster.

.. list-table::
   :header-rows: 1
   :widths: 30 55 15

   * - File Name
     - Description
     - Download
   * - install-dra-driver.sh
     - Automated deployment script for the Neuron DRA driver that applies all necessary manifests and waits for successful deployment.
     - :download:`Download <scripts/install-dra-driver.sh>`

Kubernetes Manifests
--------------------

Core Kubernetes resources required to deploy and configure the Neuron DRA driver with proper RBAC permissions.

.. list-table::
   :header-rows: 1
   :widths: 30 55 15

   * - File Name
     - Description
     - Download
   * - clusterrole.yaml
     - ClusterRole definition with permissions required for the Neuron DRA driver to manage device resources.
     - :download:`Download <manifests/clusterrole.yaml>`
   * - clusterrolebinding.yaml
     - ClusterRoleBinding that associates the service account with the required cluster role permissions.
     - :download:`Download <manifests/clusterrolebinding.yaml>`
   * - daemonset.yaml
     - DaemonSet configuration for deploying the Neuron DRA driver on all compatible Trainium nodes.
     - :download:`Download <manifests/daemonset.yaml>`
   * - deviceclass.yaml
     - DeviceClass resource that defines the Neuron device class for DRA resource allocation.
     - :download:`Download <manifests/deviceclass.yaml>`
   * - namespace.yaml
     - Namespace definition for isolating Neuron DRA driver resources within the cluster.
     - :download:`Download <manifests/namespace.yaml>`
   * - serviceaccount.yaml
     - ServiceAccount configuration for the Neuron DRA driver with appropriate security context.
     - :download:`Download <manifests/serviceaccount.yaml>`

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

