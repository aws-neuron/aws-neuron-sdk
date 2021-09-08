.. _install-neuron-1.14.2-tensorflow:

Install Neuron TensorFlow (Neuron 1.14.2)
======================================

.. contents::
   :local:
   :depth: 2




Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/develop_mode.rst

.. tabs::

   .. group-tab:: TensorFlow 1.15.5

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.14.2

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.14.2

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.14.2

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.14.2



Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/compile_mode.rst


.. tabs::

   .. group-tab:: TensorFlow 1.15.5

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.14.2

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.14.2

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.14.2

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.14.2



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/deploy_mode.rst


.. tabs::

   .. group-tab:: TensorFlow 1.15.5

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.14.2

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.14.2

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.14.2

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.14.2



