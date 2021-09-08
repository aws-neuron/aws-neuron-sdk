.. _install-neuron-1.15.0-pytorch:

Install Neuron MXNet (Neuron 1.15.0)
======================================

.. contents::
   :local:
   :depth: 2



Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/develop_mode.rst


.. tabs::

   .. group-tab:: MXNet 1.8.0

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.15.0

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.15.0

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.15.0

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.15.0



   .. group-tab:: MXNet 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=ubuntu --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=non-dlami --os=amazonlinux --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux --neuron-version=1.15.0 --framework-version=mxnet-1.5.1


 


Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/compile_mode.rst


.. tabs::

   .. group-tab:: MXNet 1.8.0

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.15.0

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.15.0

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.15.0

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.15.0



   .. group-tab:: MXNet 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=ubuntu --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=non-dlami --os=amazonlinux --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=ubuntu --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=compile --ami=dlami --os=amazonlinux --neuron-version=1.15.0 --framework-version=mxnet-1.5.1



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/deploy_mode.rst


.. tabs::

   .. group-tab:: MXNet 1.8.0

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.15.0

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.15.0

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.15.0

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.15.0




   .. group-tab:: MXNet 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=ubuntu --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=non-dlami --os=amazonlinux --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=ubuntu --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=deploy --ami=dlami --os=amazonlinux --neuron-version=1.15.0 --framework-version=mxnet-1.5.1

