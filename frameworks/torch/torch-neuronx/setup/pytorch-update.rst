.. _pytorch-neuronx-update:

Update to latest PyTorch Neuron  (``torch-neuronx``)
====================================================

.. tab-set::

    .. tab-item:: PyTorch 1.13.0

        .. tab-set::

            .. tab-item:: Amazon Linux 2 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=all --framework=pytorch --framework-version=1.13.0 --neuron-version=2.7.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

            .. tab-item:: Ubuntu 20 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=all --framework=pytorch --framework-version=1.13.0 --neuron-version=2.7.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami


    .. tab-item:: PyTorch 1.12.0

        .. tab-set::

            .. tab-item:: Amazon Linux 2 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=all --framework=pytorch --framework-version=1.12.0 --neuron-version=2.6.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

            .. tab-item:: Ubuntu 20 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=all --framework=pytorch --framework-version=1.12.0 --neuron-version=2.6.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami


    .. tab-item:: PyTorch 1.11.0

        .. tab-set::

            .. tab-item:: Amazon Linux 2 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=all --framework=pytorch --framework-version=1.11.0 --neuron-version=2.4.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

            .. tab-item:: Ubuntu 20 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=all --framework=pytorch --framework-version=1.11.0 --neuron-version=2.4.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami
