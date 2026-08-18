.. meta::
    :description: Top-k over the last dimension using the GpSIMD nisa.topk instruction.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.topk

GPSIMD Top-K Kernel API Reference
=================================

Top-k over the last dimension using the GpSIMD ``nisa.topk`` instruction.

Background
-----------

The ``gpsimd_topk`` kernel computes top-k over the last dimension using the GpSIMD ``nisa.topk`` instruction. The companion ``create_gpsimd_topk_config`` builds a ``GpsimdTopkConfig`` from an input shape (2D or 3D) and parameters.

API Reference
--------------

**Source code for this kernel API can be found at**: `gpsimd_topk.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/topk/gpsimd_topk.py>`_

create_gpsimd_topk_config
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: create_gpsimd_topk_config(inp_shape: Tuple, inp_dtype: np.dtype, k: int, sorted: bool = True, num_programs: int = 2) -> GpsimdTopkConfig

   Build a GpsimdTopkConfig from an input shape (2D or 3D) and parameters.


gpsimd_topk
^^^^^^^^^^^

.. py:function:: gpsimd_topk(inp: nl.NkiTensor, config: GpsimdTopkConfig) -> Tuple[nl.NkiTensor, nl.NkiTensor]

   Top-k over the last dimension using the GpSIMD nisa.topk instruction.

   :param inp: [BxS, V] bfloat16 input tensor in HBM.
   :type inp: ``nl.NkiTensor``
   :param config: GpsimdTopkConfig describing the problem and sharding.
   :type config: ``GpsimdTopkConfig``

   **Notes**:

   * Each vocab row is loaded into its 16-partition nisa.topk snake with a BLOCKED / contiguous DMA: partition p reads the contiguous HBM run inp[row, p*n_cols:(p+1)*n_cols] (free-stride 1), so snake position s = p + 16*c holds vocab index p*n_cols + c. The 16-partition snake LAYOUT is mandated by nisa.topk, but the ORDER of the placement is a free bijection (any order yields the same top-k value set). The blocked order is chosen so the load is contiguous; the alternative "snake position i == vocab index i" fill gives an identity remap but forces a transpose-on-load (.ap [[1,16],[16,n]], free-stride 16 -> non-contiguous per partition), which is avoided here. The returned snake-position indices are remapped back to vocab space on-chip before the index store.
   * 8 rows (groups of 16 partitions) are processed per nisa.topk call.
   * The hardware nisa.topk output order is not relied upon: the K values + paired snake-position indices are de-snaked and then sorted descending on-chip; the snake->vocab index remap is applied to the final k indices.
   * Two-phase structure: phase 1 runs nisa.topk per 8-row tile and de-snakes the K (value, index) pairs into per-row HBM buffers; phase 2 runs the descending sort ONCE over up to 128 rows (one row per partition) instead of once per 8-row tile. max8 / nc_match_replace8 / nc_n_gather are per-partition free-dim ops, so widening the sort from 8 to up to 128 partitions is free Vector-engine parallelism and removes the redundant per-tile sort passes (the measured HW bottleneck).
   * config.sorted gates only the phase-2 descending sort. When False the sort is skipped and the K results are compacted to [0, k) in arbitrary order; the value set and value<->index pairing are unchanged.

   **Dimensions**:

   * BxS: number of rows (flattened batch*sequence)
   * V: vocab size (reduction dimension), 8 <= V < 65536

