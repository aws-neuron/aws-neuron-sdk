.. _tensorflow-neuronx-install-prev-u20:

Install Previous TensorFlow Neuron Releases for Ubuntu (``tensorflow-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1



This section will assist you to install previous Neuron releases.

.. tab-set::

    .. tab-item:: Neuron 2.17.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.17.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami


    .. tab-item:: Neuron 2.16.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.16.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami


    .. tab-item:: Neuron 2.15.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.15.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami
