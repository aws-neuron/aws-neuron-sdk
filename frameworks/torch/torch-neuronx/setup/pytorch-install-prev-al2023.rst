
.. _pytorch-neuronx-install-prev-al2023:

.. Install previous PyTorch NeuronX releases for Amazon Linux 2023

Use the tabs below to install a specific previous Neuron SDK release of PyTorch NeuronX on Amazon Linux 2023. Select the Neuron version you need.

.. tab-set::

    .. tab-item:: Neuron 2.29.1

       .. note::
          Currently, PyTorch 2.9 is not available on Amazon Linux 2023 and PyTorch 2.7 and 2.8 are no longer supported for Neuron. Use Ubuntu 24.04 for PyTorch 2.9 support. If you are using Neuron 2.28.0, `see the Amazon Linux 2023 setup documentation in the 2.28.0 version of the Neuron docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/amazon-linux/torch-neuronx-al2023.html>`__.
          
    .. tab-item:: Neuron 2.28.1

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.8.0 --neuron-version=2.28.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.27.1

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.8.0 --neuron-version=2.27.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami
            