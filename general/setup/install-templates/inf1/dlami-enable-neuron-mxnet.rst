

.. tab-set::

   .. tab-item:: MXNet 1.8.0

      .. tab-set::

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux

   .. tab-item:: MXNet 1.5.1

      .. tab-set::

         .. tab-item:: Ubuntu DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=ubuntu --framework-version=mxnet-1.5.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install mxnet --mode=develop --ami=dlami --os=amazonlinux --framework-version=mxnet-1.5.1
