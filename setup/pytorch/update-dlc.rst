.. meta::
   :description: Update PyTorch Neuron in a Deep Learning Container deployment
   :keywords: pytorch, neuron, dlc, container, docker, update, upgrade, driver
   :framework: pytorch
   :installation-method: container
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :content-type: installation-guide
   :date-modified: 2026-03-30

Update PyTorch in a Deep Learning Container
=============================================

Update your DLC-based PyTorch Neuron deployment to the latest release.

.. contents:: On this page
   :local:
   :depth: 2


Update the container image
---------------------------

DLC images are versioned and tagged with the Neuron SDK version. To update, pull the
latest image tag from ECR:

.. code-block:: bash

   # Training
   docker pull public.ecr.aws/neuron/pytorch-training-neuronx:<new_image_tag>

   # Inference
   docker pull public.ecr.aws/neuron/pytorch-inference-neuronx:<new_image_tag>

   # vLLM Inference
   docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:<new_image_tag>

Replace ``<new_image_tag>`` with the tag for the desired SDK version (e.g.,
``2.9.0-neuronx-py312-sdk2.29.0-ubuntu24.04``).

Check available tags at the ECR Public Gallery:

- `PyTorch Training <https://gallery.ecr.aws/neuron/pytorch-training-neuronx>`_
- `PyTorch Inference <https://gallery.ecr.aws/neuron/pytorch-inference-neuronx>`_
- `PyTorch vLLM Inference <https://gallery.ecr.aws/neuron/pytorch-inference-vllm-neuronx>`_

For the full list of available images and tags, see :doc:`/deploy/environments/dlc-images`.


Update Neuron driver on the host
---------------------------------

The Neuron driver runs on the host, not inside the container. Update it separately
when moving to a new Neuron SDK release.

.. tab-set::

   .. tab-item:: Ubuntu 24.04

      .. code-block:: bash

         sudo apt-get update
         sudo apt-get install -y aws-neuronx-dkms

   .. tab-item:: Ubuntu 22.04

      .. important::
         Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.

      .. code-block:: bash

         sudo apt-get update
         sudo apt-get install -y aws-neuronx-dkms

   .. tab-item:: Amazon Linux 2023

      .. code-block:: bash

         sudo dnf install -y aws-neuronx-dkms


Verify the update
------------------

Launch the new container and verify:

.. code-block:: bash

   docker run -it \
     --device=/dev/neuron0 \
     --cap-add SYS_ADMIN \
     --cap-add IPC_LOCK \
     public.ecr.aws/neuron/pytorch-training-neuronx:<new_image_tag> \
     bash

Inside the container:

.. code-block:: bash

   python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
   neuron-ls


.. dropdown:: ⚠️ Troubleshooting: Version mismatch between host driver and container
   :color: warning
   :animate: fade-in

   If you see runtime errors after updating the container image but not the host driver:

   1. Check the host driver version: ``modinfo neuron`` on the host
   2. Update the host driver to match the SDK version in the container
   3. Reboot if the driver update requires it: ``sudo reboot``
