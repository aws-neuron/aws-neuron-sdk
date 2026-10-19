.. meta::
    :description: Build stable destination-major source indices without an [N, EP] HBM table.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.collectives.a2av_train

Build Packed Routing Kernel API Reference
=========================================

Build stable destination-major source indices without an [N, EP] HBM table.

``ids[N]`` contains peer IDs and ``counts[1, EP]`` their exact histogram. The output is a permutation of ``range(N)``, grouped by peer while retaining source order within each peer. A row's packed position is its peer's starting offset plus the number of earlier rows assigned to that peer. Each tile scans at most 128 rows in SBUF; per-peer cursors carry the prefix across tiles. Float32 prefix arithmetic is exact up to 2**24 rows. Run with LNC=2, with core zero owning the cursors.

Background
-----------

The ``build_packed_routing`` kernel builds stable, destination-major source indices for all-to-all-v token routing without materializing an ``[N, EP]`` HBM table. Its output is the ``packed_source_indices`` input of ``permute_a2av`` and ``unpermute_a2av``.

API Reference
--------------

**Source code for this kernel API can be found at**: `build_packed_routing.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/collectives/a2av_train/build_packed_routing.py>`_

build_packed_routing
^^^^^^^^^^^^^^^^^^^^

.. py:function:: build_packed_routing(ids, counts)

   Build stable destination-major source indices without an [N, EP] HBM table.

   :param ids: ``[N]`` destination peer ID of each source row, in HBM. ``1 <= N <= 2**24``.
   :type ids: ``nl.NkiTensor``
   :param counts: ``[1, EP]`` exact histogram of ``ids`` (rows per peer), in HBM. ``2 <= EP <= 128``.
   :type counts: ``nl.NkiTensor``
   :return: ``[N, 1]`` int32 permutation of ``range(N)`` grouped by destination peer, preserving source order within each peer. Requires LNC=2.
   :rtype: ``nl.NkiTensor``


