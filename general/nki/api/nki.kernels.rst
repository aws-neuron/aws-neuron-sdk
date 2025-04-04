NKI Kernels
==============

.. _nki_kernels:

nki.kernels
-------------

The source code of the kernels in the `neuronxcc.nki.kernels` namespace 
is available at the GitHub Repository `nki-samples <https://github.com/aws-neuron/nki-samples>`_. 
They are optimized kernels from the Neuron Team serving as samples. The repository also contains
numeric tests, performance benchmarks, as well as scripts to use them in real models.

You are welcome to customize them to fit your unique workloads, and contributing to the repository by opening a PR. 
Note that these kernels are already being deployed as part of the Neuron stack. With flash attention as an example,
`compiling Llama models with transformers-neuronx <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html>`_
will automatically invoke the `flash_fwd` kernel listed here. Therefore, replacing the framework operators with these 
NKI kernels likely won't result in extra performance benefit.

See the `README <https://github.com/aws-neuron/nki-samples>`_ page 
of the GitHub Repository `nki-samples <https://github.com/aws-neuron/nki-samples>`_ for more details.

The documentation of the kernels is available at the `nki-samples <https://github.com/aws-neuron/nki-samples>`_ repository.
