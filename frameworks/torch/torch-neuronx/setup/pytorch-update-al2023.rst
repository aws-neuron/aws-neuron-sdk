
.. _pytorch-neuronx-al2023-update:

Update to latest PyTorch NeuronX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have a previous Neuron release installed, this section provide links that will assist you to update to latest Neuron release.


.. tab-set::

    .. tab-item:: PyTorch 2.8.0

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=2.8.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami

    .. tab-item:: PyTorch 2.7.0

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=2.7.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami
