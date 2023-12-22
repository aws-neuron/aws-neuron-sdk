.. _pytorch-neuronx-install-prev-al2:

Install Previous PyTorch Neuron Releases for Amazon Linux (``torch-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1



This section will assist you to install previous Neuron releases.


.. tab-set::

    .. tab-item:: Neuron 2.15.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.15.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.14.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.14.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.13.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.13.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami