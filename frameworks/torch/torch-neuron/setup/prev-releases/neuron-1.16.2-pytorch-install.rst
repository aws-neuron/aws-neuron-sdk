.. _install-neuron-1.16.2-pytorch:

Install PyTorch Neuron (Neuron 1.16.2)
======================================

.. contents:: Table of contents
   :local:
   :depth: 2



Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/develop_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst

.. tab-set::

   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu  --neuron-version=1.16.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux  --neuron-version=1.16.2


   .. tab-item:: PyTorch 1.8.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1



   .. tab-item:: PyTorch 1.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1


   .. tab-item:: PyTorch 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1


 


Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/compile_mode.rst


.. tab-set::

   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu  --neuron-version=1.16.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux  --neuron-version=1.16.2


   .. tab-item:: PyTorch 1.8.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1



   .. tab-item:: PyTorch 1.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1


   .. tab-item:: PyTorch 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/deploy_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst


.. tab-set::

   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu  --neuron-version=1.16.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux  --neuron-version=1.16.2



   .. tab-item:: PyTorch 1.8.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.8.1



   .. tab-item:: PyTorch 1.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.7.1


   .. tab-item:: PyTorch 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux  --neuron-version=1.16.2 --framework-version=pytorch-1.5.1

