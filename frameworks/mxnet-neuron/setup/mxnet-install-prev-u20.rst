.. _mxnet-neuron-install-prev-u20:

Install Previous MXNet Neuron Releases for Ubuntu 20(``mxnet-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1


This section will assist you to install previous Neuron releases.


.. tab-set::

    .. tab-item:: Neuron 2.10.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.5.1 --neuron-version=2.10.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.9.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.5.1 --neuron-version=2.9.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami


    .. tab-item:: Neuron 2.8.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.5.1 --neuron-version=2.8.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami
