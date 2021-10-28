.. tabs::

   .. group-tab:: TensorFlow 2.5.1

      .. tabs::

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=ubuntu

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=amazonlinux




   .. group-tab:: TensorFlow 1.15.5

      .. tabs::

         .. group-tab:: Ubuntu DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=ubuntu --framework-version=tensorflow-1.15.5

         .. group-tab:: Amazon Linux DLAMI

            .. include :: /neuron-intro/install-templates/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install tensorflow --mode=develop --ami=dlami --os=amazonlinux --framework-version=tensorflow-1.15.5    

