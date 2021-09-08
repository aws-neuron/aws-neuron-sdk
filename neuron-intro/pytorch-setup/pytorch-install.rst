.. _install-neuron-pytorch:

Install Neuron PyTorch
=====================

.. contents::
   :local:
   :depth: 2



Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/develop_mode.rst

.. tabs::

   .. group-tab:: PyTorch 1.8.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux



   .. group-tab:: PyTorch 1.7.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.7.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux --framework-version=pytorch-1.7.1


   .. group-tab:: PyTorch 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux --framework-version=pytorch-1.5.1


 


Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/compile_mode.rst


.. tabs::

   .. group-tab:: PyTorch 1.8.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux



   .. group-tab:: PyTorch 1.7.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.7.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux --framework-version=pytorch-1.7.1


   .. group-tab:: PyTorch 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=ubuntu --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=compile --ami=dlami --os=amazonlinux --framework-version=pytorch-1.5.1



Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/deploy_mode.rst


.. tabs::

   .. group-tab:: PyTorch 1.8.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux



   .. group-tab:: PyTorch 1.7.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.7.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu --framework-version=pytorch-1.7.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux --framework-version=pytorch-1.7.1


   .. group-tab:: PyTorch 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=pytorch-1.5.1

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=ubuntu --framework-version=pytorch-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=deploy --ami=dlami --os=amazonlinux --framework-version=pytorch-1.5.1

