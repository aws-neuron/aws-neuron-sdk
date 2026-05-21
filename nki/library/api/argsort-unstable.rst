.. meta::
    :description: Perform unstable argsort on 1D input buffer. Elements with equal values may appear in any order relative to their origin
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.subkernels

Argsort Unstable Kernel API Reference
=====================================

Perform unstable argsort on 1D input buffer. Elements with equal values may appear in any order relative to their original positions.

For example: data = [5, 2, 5, 3] Pass 0 (ascending mode): max8 vals = [5, 5, 3, 2, ...] nc_match_replace8 matches: pos 0, pos 2, pos 3, pos 1 reversed output indices: [1, 3, 2, 0] Result: indices = [1, 3, 2, 0]  (values in order: 2, 3, 5, 5) Indices [2, 0] corresponding to values [5, 5] are not in original order.

Background
-----------

The ``argsort_unstable`` kernel performs an unstable argsort on a 1D input buffer, returning indices that would sort the data in ascending or descending order. Elements with equal values may appear in any order relative to their original positions.

API Reference
--------------

**Source code for this kernel API can be found at**: `argsort_unstable.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/subkernels/argsort_unstable.py>`_

argsort_unstable
^^^^^^^^^^^^^^^^

.. py:function:: argsort_unstable(data, descending = False, output_in_sbuf = False)

   Perform unstable argsort on 1D input buffer. Elements with equal values may appear in any order relative to their original positions.

   :param data: [1, N] int32/float32 tensor in HBM or SBUF.
   :param descending: When True, return indices for descending order. Defaults to ascending.
   :param output_in_sbuf: When True, return SBUF output. Defaults to HBM output.
   :return: [1, N] uint32 tensor in HBM or SBUF containing the argsort indices.
   :rtype: ``nl.ndarray``

