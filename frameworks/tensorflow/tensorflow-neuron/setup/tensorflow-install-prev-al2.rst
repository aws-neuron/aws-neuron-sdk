.. _tensorflow-neuron-install-prev-al2:

Install Previous Tensorflow Neuron Releases for Amazon Linux (``tensorflow-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1


This section will assist you to install previous Neuron releases.

.. tab-set::

    .. tab-item:: Neuron 2.18.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.18.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.17.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.17.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami


    .. tab-item:: Neuron 2.16.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.16.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami
