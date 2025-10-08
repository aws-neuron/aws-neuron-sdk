.. _tensorflow-neuron-install-prev-al2023:

Install Previous Tensorflow Neuron Releases for Ubuntu (``tensorflow-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1


This section will assist you to install previous Neuron releases.

.. tab-set::

    .. tab-item:: Neuron 2.21.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.21.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.20.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.20.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=inf1 --ami=non-dlami
    
    .. tab-item:: Neuron 2.19.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.19.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=inf1 --ami=non-dlami
