nki.language.tile\_size
=======================

.. currentmodule:: nki.language

.. class:: tile_size

   Hardware tile size constants (pmax, psum_fmax, gemm_stationary_fmax, etc.).

   .. rubric:: Attributes

   .. attribute:: pmax

      Maximum partition dimension of a tile.

   .. attribute:: psum_fmax

      Maximum free dimension of a tile on PSUM buffer, in FP32 elements.

   .. attribute:: psum_fmax_bytes

      Maximum free dimension of a tile on PSUM buffer, in bytes.

   .. attribute:: psum_num_banks

      Number of usable PSUM banks per partition.

      Returns 7 when ``dma_transpose`` is lowered to PE transpose
      (``NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=true`` on trn2+), since bank 7 is
      reserved for ``nc_transpose``. Otherwise returns 8.

   .. attribute:: sbuf_size_bytes

      Total SBUF capacity in bytes (all partitions combined).

   .. attribute:: sbuf_fmax

      Maximum free dimension of a tile on SBUF buffer, in FP32 elements.

   .. attribute:: sbuf_fmax_bytes

      Maximum free dimension of a tile on SBUF buffer, in bytes.

   .. attribute:: gemm_stationary_fmax

      Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine.

   .. attribute:: gemm_moving_fmax

      Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine.

   .. attribute:: bn_stats_fmax

      Maximum free dimension of BN_STATS.

   .. attribute:: psum_min_align

      Minimum byte alignment requirement for PSUM free dimension address.

   .. attribute:: sbuf_min_align

      Minimum byte alignment requirement for SBUF free dimension address.

   .. attribute:: total_available_sbuf_size

      Usable SBUF free dimension per partition, in bytes.

      .. deprecated:: 0.4.0b4
         Despite the name, this returns the usable SBUF capacity *per
         partition*, not the total SBUF capacity. Use
         :py:attr:`sbuf_size_bytes` for the total SBUF capacity across all
         partitions, or :py:attr:`sbuf_fmax_bytes` for the usable
         per-partition size.
