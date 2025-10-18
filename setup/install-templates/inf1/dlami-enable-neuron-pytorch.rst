.. tab-set::

   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux

   .. tab-item:: PyTorch 1.8.1

      .. tab-set::

         .. tab-item:: Ubuntu DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=ubuntu --framework=pytorch-1.8.1

         .. tab-item:: Amazon Linux DLAMI

            .. include :: /setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --install pytorch --mode=develop --ami=dlami --os=amazonlinux --framework=pytorch-1.8.1


   
