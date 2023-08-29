
.. _tensorflow-neuron-u20-update:

Update to latest TensorFlow Neuron  (``tensorflow-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have a previous Neuron release installed, this section provide links that will assist you to update to latest Neuron release.


.. tab-set::

    .. tab-item:: TensorFlow 2.10.1

        .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=inf1 --ami=non-dlami


    .. tab-item:: TensorFlow 2.9.3

        .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=tensorflow --framework-version=2.9.3 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=inf1 --ami=non-dlami


    .. tab-item:: TensorFlow 2.8.4

        .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=tensorflow --framework-version=2.8.4 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=inf1 --ami=non-dlami
