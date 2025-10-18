.. _install-neuron-1.14.2-tensorflow:

Install TensorFlow Neuron (Neuron 1.14.2)
======================================

.. contents:: Table of contents
   :local:
   :depth: 2




Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /setup/install-templates/inf1/develop_mode.rst


.. tab-set::

   .. tab-item:: TensorFlow 1.15.5

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.14.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.14.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.14.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.14.2



Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /setup/install-templates/inf1/compile_mode.rst



.. tab-set::

   .. tab-item:: TensorFlow 1.15.5

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.14.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.14.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.14.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.14.2



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /setup/install-templates/inf1/deploy_mode.rst



.. tab-set::

   .. tab-item:: TensorFlow 1.15.5

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.14.2

         .. tab-item:: Amazon Linux AMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.14.2

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.14.2

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.14.2



