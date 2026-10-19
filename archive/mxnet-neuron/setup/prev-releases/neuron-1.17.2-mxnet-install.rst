.. _install-neuron-1.17.2-mxnet:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

Install MXNet Neuron
=====================

.. warning::

   This document is archived. MXNet is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/archive/frameworks/index`.


.. include:: /setup/install-templates/inf1/note-setup-cntr.rst


.. contents:: Table of contents
   :local:
   :depth: 2



Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /setup/install-templates/inf1/develop_mode.rst



.. tab-set::

   .. tab-item:: MXNet 1.8.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.17.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.17.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.17.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.17.2



   .. tab-item:: MXNet 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.17.2 --framework-version=mxnet-1.5.1


 


Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /setup/install-templates/inf1/compile_mode.rst



.. tab-set::

   .. tab-item:: MXNet 1.8.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.17.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.17.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.17.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.17.2



   .. tab-item:: MXNet 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.17.2 --framework-version=mxnet-1.5.1



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /setup/install-templates/inf1/deploy_mode.rst



.. tab-set::

   .. tab-item:: MXNet 1.8.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.17.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.17.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.17.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.17.2




   .. tab-item:: MXNet 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.17.2 --framework-version=mxnet-1.5.1

