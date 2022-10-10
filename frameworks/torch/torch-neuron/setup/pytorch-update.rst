.. _update-neuron-pytorch:

Update to latest PyTorch Neuron (``torch-neuron``)
==================================================

.. include:: /general/setup/install-templates/inf1/note-setup-cntr.rst

.. contents:: Table of contents
   :local:
   :depth: 2


Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/develop_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst

.. tab-set::

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=amazonlinux

   .. tab-item:: PyTorch 1.10.2

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.10.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.10.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=ubuntu --framework-version=pytorch-1.10.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=amazonlinux --framework-version=pytorch-1.10.2


   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.9.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.9.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=ubuntu --framework-version=pytorch-1.9.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=amazonlinux --framework-version=pytorch-1.9.1


   .. tab-item:: PyTorch 1.8.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.8.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=ubuntu --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=amazonlinux --framework-version=pytorch-1.8.1



   .. tab-item:: PyTorch 1.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=develop --ami=dlami --os=amazonlinux --framework-version=pytorch-1.7.1






Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/compile_mode.rst


.. tab-set::

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=amazonlinux


   .. tab-item:: PyTorch 1.10.2

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.10.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.10.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=ubuntu --framework-version=pytorch-1.10.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=amazonlinux --framework-version=pytorch-1.10.2



   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.9.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.9.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=ubuntu --framework-version=pytorch-1.9.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=amazonlinux --framework-version=pytorch-1.9.1


   .. tab-item:: PyTorch 1.8.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.8.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=ubuntu --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=amazonlinux --framework-version=pytorch-1.8.1



   .. tab-item:: PyTorch 1.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=compile --ami=dlami --os=amazonlinux --framework-version=pytorch-1.7.1




Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/deploy_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst


.. tab-set::

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=amazonlinux

   .. tab-item:: PyTorch 1.10.2

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.10.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.10.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=ubuntu --framework-version=pytorch-1.10.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=amazonlinux --framework-version=pytorch-1.10.2


   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.9.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.9.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=ubuntu --framework-version=pytorch-1.9.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=amazonlinux --framework-version=pytorch-1.9.1


   .. tab-item:: PyTorch 1.8.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.8.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=ubuntu --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=amazonlinux --framework-version=pytorch-1.8.1



   .. tab-item:: PyTorch 1.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update pytorch --mode=deploy --ami=dlami --os=amazonlinux --framework-version=pytorch-1.7.1


