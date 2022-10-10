.. _update-neuron-tensorflow:

Update to latest TensorFlow Neuron
===============================

.. include:: /general/setup/install-templates/inf1/note-setup-cntr.rst


.. contents:: Table of contents
   :local:
   :depth: 2


Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/develop_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst


.. tab-set::

   .. tab-item:: TensorFlow 2.8.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux


   .. tab-item:: TensorFlow 2.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-2.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.7.1


   .. tab-item:: TensorFlow 2.6.3

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.6.3

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.6.3

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-2.6.3

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.6.3


   .. tab-item:: TensorFlow 2.5.3

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.5.3

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.5.3

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-2.5.3

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.5.3




   .. tab-item:: TensorFlow 1.15.5

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-1.15.5

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-1.15.5    
         

Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/compile_mode.rst


.. tab-set::

   .. tab-item:: TensorFlow 2.8.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux

   .. tab-item:: TensorFlow 2.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-2.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.7.1


   .. tab-item:: TensorFlow 2.6.3

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.6.3

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.6.3

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-2.6.3

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.6.3


   .. tab-item:: TensorFlow 2.5.3

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.5.3

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.5.3

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-2.5.3

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.5.3



   .. tab-item:: TensorFlow 1.15.5

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-1.15.5

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-1.15.5   



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/deploy_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst


.. tab-set::

   .. tab-item:: TensorFlow 2.8.0

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux


   .. tab-item:: TensorFlow 2.7.1

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.7.1

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.7.1

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-2.7.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.7.1


   .. tab-item:: TensorFlow 2.6.3

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.6.3

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.6.3

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-2.6.3

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.6.3


   .. tab-item:: TensorFlow 2.5.3

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.5.3

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.5.3

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-2.5.3

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.5.3



   .. tab-item:: TensorFlow 1.15.5

      .. tab-set::

         .. tab-item:: Ubuntu AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. tab-item:: Amazon Linux AMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-1.15.5

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-1.15.5   





