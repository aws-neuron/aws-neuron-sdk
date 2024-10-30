.. _mxnet-neuron-install-prev-u22:

Install Previous MXNet Neuron Releases for Ubuntu 22 (``mxnet-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1


This section will assist you to install previous Neuron releases.


.. tab-set::

    .. tab-item:: Neuron 2.19.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --neuron-version=2.19.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.18.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --neuron-version=2.18.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.17.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --neuron-version=2.17.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.16.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --neuron-version=2.16.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=inf1 --ami=non-dlami
