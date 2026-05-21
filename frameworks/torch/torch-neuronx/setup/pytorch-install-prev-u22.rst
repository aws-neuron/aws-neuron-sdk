
.. _pytorch-neuronx-install-prev-u22:

.. Install previous PyTorch NeuronX releases for Ubuntu 22.04

Use the tabs below to install a specific previous Neuron SDK release of PyTorch NeuronX on Ubuntu 22.04. Select the Neuron version you need.

.. tab-set::

    .. tab-item:: Neuron 2.29.1

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.9.0 --neuron-version=2.29.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.28.1

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.9.0 --neuron-version=2.28.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.27.1

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.9.0 --neuron-version=2.27.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami
            