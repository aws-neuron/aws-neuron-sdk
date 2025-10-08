.. _neuron-2-26-0-nki:

.. meta::
   :description: The official release notes for the AWS Neuron Kernel Interface (NKI) component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: Neuron Kernel Interface (NKI) release notes
=================================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Improvements
------------

New nki.language APIs
^^^^^^^^^^^^^^^^^^^^^

* :doc:`nki.language.gelu_apprx_sigmoid <../../nki/api/generated/nki.language.gelu_apprx_sigmoid>` - Gaussian Error Linear Unit activation function with sigmoid approximation.

Updated nki.language APIs
^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`nki.language.tile_size.total_available_sbuf_size <../../nki/api/generated/nki.language.tile_size>` constant - Added a new field, ``total_available_sbuf_size``, that contains the returned total available SBUF size.

New nki.isa APIs
^^^^^^^^^^^^^^^^

* :doc:`nki.isa.select_reduce <../../nki/api/generated/nki.isa.select_reduce>` - Selectively copy elements with maximum reduction.
* :doc:`nki.isa.sequence_bounds <../../nki/api/generated/nki.isa.sequence_bounds>` - Compute sequence bounds of segment IDs.
* :doc:`nki.isa.dma_transpose <../../nki/api/generated/nki.isa.dma_transpose>` - Enhanced with:

  * ``axes`` parameter to define 4D transpose for supported cases
  * ``dge_mode`` parameter to specify Descriptor Generation Engine (DGE)

* :doc:`nki.isa.activation <../../nki/api/generated/nki.isa.activation>` - Supports the new ``nl.gelu_apprx_sigmoid`` nki.language operation.

Improvements and fixes
^^^^^^^^^^^^^^^^^^^^^^

* **nki.language.store()** - Supports PSUM buffer with extra additional copy inserted.

Documentation and tutorial updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Added documentation and example for :doc:`nki.isa.dma_transpose <../../nki/api/generated/nki.isa.dma_transpose>` API
* Improved :doc:`nki.simulate_kernel <../../nki/api/generated/nki.simulate_kernel>` example
* Updated :doc:`tutorial<nki_tutorials>` code to use ``nl.fp32.min`` instead of a magic number

Previous release notes
----------------------

* :ref:`nki_rn`

