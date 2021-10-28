

.. tabs::

   .. group-tab:: MXNet 1.8.0

      .. tabs::

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux

   .. group-tab:: MXNet 1.5.1

      .. tabs::

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux --framework-version=mxnet-1.5.1
