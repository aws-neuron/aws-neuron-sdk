.. _pytorch-neuronx-install-prev-al2023:

Install Previous PyTorch NeuronX Releases for AL2023 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1



This section will assist you to install previous Neuron releases.

.. tab-set::

    .. tab-item:: Neuron 2.19.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.1.2 --neuron-version=2.19.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.18.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.1.2 --neuron-version=2.18.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.17.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.17.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami
