.. _neuronx-distributed-index:


Neuron Distributed [Experimental]
=================================
Neuron Distributed is a package for supporting different distributed
training/inference mechanism for Neuron devices. It would provide xla
friendly implementations of some of the more popular distributed
training/inference techniques. As the size of the model scales, fitting
these models on a single device becomes impossible and hence we have to
make use of model sharding techniques to partition the model across
multiple devices. As part of this library, we enable support for Tensor
Parallel sharding technique with other distributed library supported to be 
added in future.

.. toctree::
    :maxdepth: 1
    :hidden:

    Setup </libraries/neuronx-distributed/setup/index>
    App Notes </libraries/neuronx-distributed/app_notes>
    API Reference Guide </libraries/neuronx-distributed/api-reference-guide>
    Developer Guide  </libraries/neuronx-distributed/developer-guide>
    Tutorials  </libraries/neuronx-distributed/tutorials/index>
    Misc  </libraries/neuronx-distributed/neuronx-distributed-misc>


.. include:: /libraries/neuronx-distributed/neuronx-distributed.txt