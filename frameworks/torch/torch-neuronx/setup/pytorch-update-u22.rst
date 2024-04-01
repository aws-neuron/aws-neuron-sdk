
.. _pytorch-neuronx-ubuntu22-update:

Update to latest PyTorch NeuronX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have a previous Neuron release installed, this section provide links that will assist you to update to latest Neuron release.


.. tab-set::

    .. tab-item:: PyTorch 2.1.2

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. include:: /src/helperscripts/installationScripts/python_instructions.txt
            :start-line: 191
            :end-line: 192

    .. tab-item:: PyTorch 1.13.1

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami

    