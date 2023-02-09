.. _install-neuronx-2.6.0-pytorch:

Install PyTorch Neuron (Neuron 2.6.0)
======================================

.. tab-set::

	.. tab-item:: PyTorch 1.12.0

		.. tab-set::

			.. tab-item:: Amazon Linux 2 AMI

				.. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

				.. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=all --framework=pytorch --framework-version=1.12.0 --neuron-version=2.6.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

			.. tab-item:: Ubuntu 20 AMI

				.. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

				.. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=all --framework=pytorch --framework-version=1.12.0 --neuron-version=2.6.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami
