.. _pytorch-neuronx-install:

Install PyTorch Neuron  (``torch-neuronx``)
===========================================

.. contents:: Table of Contents
   :local:
   :depth: 2


Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: PyTorch 1.13.0

      .. tab-set::

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/trn1/dlami-notes.rst
                :start-line: 13
                :end-line: 18

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=all --framework=pytorch --framework-version=1.13.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/trn1/dlami-notes.rst
                :start-line: 19
                :end-line: 24

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=all --framework=pytorch --framework-version=1.13.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Pytorch

            .. include :: /general/setup/install-templates/trn1/dlami-notes.rst
                :start-line: 7
                :end-line: 9

         .. tab-item:: Ubuntu 20 DLAMI Pytorch

            .. include :: /general/setup/install-templates/trn1/dlami-notes.rst
                :start-line: 10
                :end-line: 12

         .. tab-item:: Amazon Linux 2

            .. include :: /general/setup/install-templates/trn1/dlami-notes.rst
                :start-line: 1
                :end-line: 3

         .. tab-item:: Ubuntu 20

            .. include :: /general/setup/install-templates/trn1/dlami-notes.rst
                :start-line: 4
                :end-line: 6