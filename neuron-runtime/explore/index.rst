.. _neuron-runtime-explore-home:

.. meta::
   :description: Topics that explore the AWS Neuron Runtime and tools in-depth, written by the AWS engineers who developed them.
   :keywords: AWS Neuron, deep dives, whitepapers, engineering

Neuron Runtime Deep Dives
==========================

.. toctree::
   :hidden:
   :maxdepth: 1

   Understand NEFF Files <work-with-neff-files>
   Compute-Communication Overlap <compute-comm-overlap>
   Neuron Device Memory <device-memory>
   Direct HBM Tensor Allocation <direct-hbm-tensor-alloc>
   Runtime Performance Tips <runtime-performance-tips>
   Neuron Runtime Core Dumps <core-dump-deep-dive>
   Inter-node Collectives <internode-collective-comm>
   Intra-node Collectives <intranode-collective-comm>

Curious about how the Neuron Runtime works? Looking for deeper explorations of the computer science, techniques, and algorithms used to develop it? This section provides topics that dive into the learnings and engineering behind the Neuron Runtime, written by the AWS engineers who developed it.

NeuronX Runtime Deep Dives
---------------------------

.. grid:: 2
        :gutter: 2

        .. grid-item-card:: Understand NEFF Files

                * :ref:`work-with-neff-files`

                Explore the structure and contents of NEFF files, the compiled model format used by the Neuron Runtime.

        .. grid-item-card:: Compute-Communication Overlap

                * :ref:`neuron-runtime-explore-compute-comm`
  
        .. grid-item-card:: Neuron Device Memory

                * :ref:`neuron-device-memory-deep-dive`

                Learn how the Neuron Runtime overlaps computation and communication to maximize performance on AWS Inferentia and Trainium chips.
  
        .. grid-item-card:: Neuron Device Memory

                * :ref:`neuron-device-memory-deep-dive`

                Understand, monitor, and optimize memory usage on AWS Neuron devices including tensors, model constants, scratchpad allocations, and more.

        .. grid-item-card:: Direct HBM Tensor Allocation

                * :ref:`direct-hbm-tensor-alloc`
  
                Optimize performance by allocating tensors directly into High Bandwidth Memory (HBM) on Neuron devices, eliminating CPU-device memory transfer overhead.

        .. grid-item-card:: Runtime Performance Tips

                * :ref:`runtime-performance-tips`
  
                Best practices and optimization techniques for achieving optimal performance with the AWS Neuron Runtime. 

        .. grid-item-card:: Neuron Runtime Core Dumps   

                * :ref:`runtime-core-dump-deep-dive`

                Dive into the structure and analysis of Neuron Runtime core dumps to troubleshoot and debug runtime issues effectively.

Neuron Collectives Deep Dives
-----------------------------

.. grid:: 2
        :gutter: 2

        .. grid-item-card:: Inter-node Collectives Communication

                * :doc:`internode-collective-comm`

                Explore Ring, Mesh, and Recursive Doubling-Halving algorithms for coordinating data exchange across multiple nodes via EFA networks.

        .. grid-item-card:: Intra-node Collectives Communication

                * :doc:`intranode-collective-comm`

                Learn about Ring, Mesh, KangaRing, and RDH algorithms optimized for high-bandwidth NeuronLink communication within single nodes.
