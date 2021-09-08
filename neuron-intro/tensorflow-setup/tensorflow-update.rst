.. _update-neuron-tensorflow:

Update to latest Neuron TensorFlow
===============================

.. contents::
   :local:
   :depth: 2



Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/develop_mode.rst


.. tabs::

   .. group-tab:: TensorFlow 2.5.0

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux
 
   .. group-tab:: TensorFlow 2.4.2

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.4.2

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.4.2

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-2.4.2

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.4.2


   .. group-tab:: TensorFlow 2.3.3

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.3.3

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.3.3

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-2.3.3

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.3.3


   .. group-tab:: TensorFlow 2.2.3

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.2.3

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.2.3

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-2.2.3

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.2.3


   .. group-tab:: TensorFlow 2.1.4

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.1.4

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.1.4

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-2.1.4

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.1.4      


   .. group-tab:: TensorFlow 1.15.5

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-1.15.5

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-1.15.5    
         

Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/compile_mode.rst


.. tabs::

   .. group-tab:: TensorFlow 2.5.0

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux





   .. group-tab:: TensorFlow 2.4.2

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.4.2

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.4.2

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-2.4.2

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.4.2


   .. group-tab:: TensorFlow 2.3.3

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.3.3

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.3.3

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-2.3.3

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.3.3


   .. group-tab:: TensorFlow 2.2.3

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.2.3

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.2.3

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-2.2.3

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.2.3


   .. group-tab:: TensorFlow 2.1.4

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.1.4

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.1.4

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-2.1.4

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.1.4      


   .. group-tab:: TensorFlow 1.15.5

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-1.15.5

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=compile --ami=dlami --os=amazonlinux --framework-version=tensorflow-1.15.5   






Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/deploy_mode.rst


.. tabs::

   .. group-tab:: TensorFlow 2.5.0

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux





   .. group-tab:: TensorFlow 2.4.2

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.4.2

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.4.2

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-2.4.2

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.4.2


   .. group-tab:: TensorFlow 2.3.3

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.3.3

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.3.3

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-2.3.3

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.3.3


   .. group-tab:: TensorFlow 2.2.3

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.2.3

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.2.3

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-2.2.3

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.2.3


   .. group-tab:: TensorFlow 2.1.4

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-2.1.4

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-2.1.4

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-2.1.4

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-2.1.4      


   .. group-tab:: TensorFlow 1.15.5

      .. tabs::

         .. group-tab:: Ubuntu AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. group-tab:: Amazon Linux AMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=non-dlami --os=amazonlinux --framework-version=tensorflow-1.15.5

         .. group-tab:: Ubuntu DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. group-tab:: Amazon Linux DLAMI

            .. note ::

               For a successful installation or update, execute each line of the instructions below separately or 
               copy the contents of the code block into a script file and source its contents.

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --update tensorflow --mode=deploy --ami=dlami --os=amazonlinux --framework-version=tensorflow-1.15.5   





