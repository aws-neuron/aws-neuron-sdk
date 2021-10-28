.. important ::

   For successful installation or update to Neuron 1.16.0 and newer from previous releases:
      * Stop Neuron Runtime 1.x daemon (``neuron-rtd``) by running: ``sudo systemctl stop neuron-rtd``
      * Uninstall ``neuron-rtd`` by running: ``sudo apt remove aws-neuron-runtime`` or ``sudo yum remove aws-neuron-runtime``
      * Install or upgrade to latest Neuron driver (``aws-neuron-dkms``) by following the "Setup Guide" instructions.
      * Visit :ref:`introduce-libnrt` for more information.
