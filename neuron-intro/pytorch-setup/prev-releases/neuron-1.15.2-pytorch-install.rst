.. _install-neuron-1.15.2-pytorch:

Install Neuron PyTorch (Neuron 1.15.2)
======================================

.. contents::
   :local:
   :depth: 2



Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/develop_mode.rst



.. tabs::

   .. group-tab:: PyTorch 1.8.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.15.2

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.15.2

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.15.2



   .. group-tab:: PyTorch 1.7.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.7.1


   .. group-tab:: PyTorch 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.5.1


 


Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/compile_mode.rst



.. tabs::

   .. group-tab:: PyTorch 1.8.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.15.2

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.15.2

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.15.2



   .. group-tab:: PyTorch 1.7.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.7.1


   .. group-tab:: PyTorch 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.5.1



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/deploy_mode.rst



.. tabs::

   .. group-tab:: PyTorch 1.8.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.15.2

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.15.2

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.15.2



   .. group-tab:: PyTorch 1.7.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.7.1


   .. group-tab:: PyTorch 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.15.2 --framework-version=pytorch-1.5.1

