
.. tensorflow-neuronx-u22-update:

Update to latest TensorFlow Neuron  (``tensorflow-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have a previous Neuron release installed, this section provide links that will assist you to update to latest Neuron release.


.. tab-set::

    .. tab-item:: Tensorflow 2.10.1

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu22 --instance=trn1 --ami=non-dlami --category=compiler_framework
