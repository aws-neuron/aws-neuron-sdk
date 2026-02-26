.. _nrt_api_reference:

Neuron Runtime API Reference
=============================

This section provides comprehensive API reference documentation for the Neuron Runtime (NRT) and Neuron Driver Library (NDL). These APIs enable low-level access to AWS Neuron devices and provide interfaces for model loading, execution, memory management, and collective operations.

**Source code for these APIs can be found at**: https://github.com/aws-neuron/aws-neuron-sdk.

Core Runtime APIs
-----------------

.. list-table::
   :widths: 40 60

   * - :doc:`NRT API </neuron-runtime/api/nrt>`
     - Main Neuron Runtime API for model loading, execution, and tensor management
   * - :doc:`NRT Status </neuron-runtime/api/nrt_status>`
     - Status codes and error handling for runtime operations
   * - :doc:`NRT Version </neuron-runtime/api/nrt_version>`
     - Version information and compatibility checking

Asynchronous Execution APIs
----------------------------

.. list-table::
   :widths: 40 60

   * - :doc:`NRT Async </neuron-runtime/api/nrt_async>`
     - Asynchronous execution API for non-blocking operations
   * - :doc:`NRT Async Send/Recv </neuron-runtime/api/nrt_async_sendrecv>`
     - Asynchronous tensor send and receive operations

Profiling and Debugging APIs
-----------------------------

.. list-table::
   :widths: 40 60

   * - :doc:`NRT Profile </neuron-runtime/api/nrt_profile>`
     - Profiling API for performance analysis and optimization
   * - :doc:`NRT System Trace </neuron-runtime/api/nrt_sys_trace>`
     - System trace capture and event fetching
   * - :doc:`Debug Stream </neuron-runtime/api/ndebug_stream>`
     - Debug event streaming from Logical Neuron Cores

Collective Operations API
--------------------------

.. list-table::
   :widths: 40 60

   * - :doc:`NEC API </neuron-runtime/api/nec>`
     - Neuron Elastic Collectives (NEC) for distributed operations

Neuron Driver Library (NDL) APIs
---------------------------------

.. list-table::
   :widths: 40 60

   * - :doc:`NDL API </neuron-runtime/api/ndl>`
     - Low-level Neuron Driver Library for device access and control
   * - :doc:`Neuron Driver Shared </neuron-runtime/api/neuron_driver_shared>`
     - Shared definitions between runtime and driver
   * - :doc:`Tensor Batch Operations </neuron-runtime/api/neuron_driver_shared_tensor_batch_op>`
     - Batch operation structures for tensor transfers

Neuron Datastore API
--------------------

.. list-table::
   :widths: 40 60

   * - :doc:`Neuron Datastore </neuron-runtime/api/neuron_ds>`
     - Neuron Datastore (NDS) for sharing metrics and model information

Experimental APIs
-----------------

.. list-table::
   :widths: 40 60

   * - :doc:`NRT Experimental </neuron-runtime/api/nrt_experimental>`
     - Experimental features and APIs (subject to change)

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Core Runtime APIs

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Debugging

   Debug Stream APIs </neuron-runtime/api/debug-stream-api>


.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Asynchronous Execution APIs

   NRT Async <nrt_async>
   NRT Async Send/Recv <nrt_async_sendrecv>

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Profiling and Debugging APIs

   NRT Profile <nrt_profile>
   NRT System Trace <nrt_sys_trace>
   Debug Stream <ndebug_stream>

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Collective Operations API

   NEC API <nec>

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Neuron Driver Library APIs

   NDL API <ndl>
   Neuron Driver Shared <neuron_driver_shared>
   Tensor Batch Operations <neuron_driver_shared_tensor_batch_op>

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Neuron Datastore API

   Neuron Datastore <neuron_ds>

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: Experimental APIs

   NRT Experimental <nrt_experimental>
