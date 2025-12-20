.. _pytorch-neuronx-install-prev-u20:

Install Previous PyTorch NeuronX Releases for Ubuntu 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1



This section will assist you to install previous Neuron releases.

.. tab-set::
    
    .. tab-item:: Neuron 2.26.1

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.8.0 --neuron-version=2.26.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.25.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.7.0 --neuron-version=2.25.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.24.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.7.0 --neuron-version=2.24.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami
