.. _install-neuron-mxnet:

Install MXNet Neuron
=====================

.. include:: /general/setup/install-templates/inf1/note-setup-cntr.rst


.. contents:: Table of contents
   :local:
   :depth: 2


Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/develop_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst


.. tab-set::

   .. tab-item:: MXNet 1.8.0

      .. tab-set::

         .. tab-item:: Ubuntu 18 AMI/Ubuntu 20 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=ubuntu 

         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu 18 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux 2 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux


   .. tab-item:: MXNet 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu 18 AMI/Ubuntu 20 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=mxnet-1.5.1

         .. tab-item:: Ubuntu 18 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux 2 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux --framework-version=mxnet-1.5.1




Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/compile_mode.rst


.. tab-set::

   .. tab-item:: MXNet 1.8.0

      .. tab-set::

         .. tab-item:: Ubuntu 18 AMI/Ubuntu 20 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu 18 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux 2 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=amazonlinux


   .. tab-item:: MXNet 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu 18 AMI/Ubuntu 20 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=mxnet-1.5.1

         .. tab-item:: Ubuntu 18 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux 2 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=amazonlinux --framework-version=mxnet-1.5.1



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/deploy_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst


.. tab-set::

   .. tab-item:: MXNet 1.8.0

      .. tab-set::

         .. tab-item:: Ubuntu 18 AMI/Ubuntu 20 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu 18 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux 2 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=amazonlinux


   .. tab-item:: MXNet 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu 18 AMI/Ubuntu 20 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=mxnet-1.5.1

         .. tab-item:: Ubuntu 18 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux 2 DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=amazonlinux --framework-version=mxnet-1.5.1



