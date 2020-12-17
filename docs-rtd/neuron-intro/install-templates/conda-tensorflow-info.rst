.. note::

   The ``tensorflow-neuron`` Conda package comes with
   TensorBoard-Neuron. There is no standalone ``tensorboard-neuron``
   Conda package at this time.

.. note::

   .. container:: toggle-header

      .. code:: bash

         curl https://conda.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --import

         # This shows the version/build number of the package
         conda search tensorflow-neuron

         # Use the version/build number above to download the package and the signature
         wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2
         wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2.asc
         gpg --verify tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2.asc tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2
