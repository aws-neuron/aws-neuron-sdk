.. meta::
    :description: Cross entropy kernel implements memory-efficient cross entropy loss for large vocabularies using online log-sum-exp algorithm.
    :date-modified: 02/13/2026

.. currentmodule:: nkilib.experimental.loss

Cross Entropy Kernel API Reference
===================================

Implements memory-efficient cross entropy loss computation for large vocabularies using the online log-sum-exp algorithm with batched processing.

The kernel supports:

* Memory-efficient computation for large vocabularies
* Online log-sum-exp algorithm to avoid numerical overflow
* Forward and backward pass kernels
* Batched processing for improved throughput
* Optimized for LNC2 (2 cores) architecture
* Configurable chunk sizes and batch sizes
* Support for bfloat16 and float32 data types

Background
-----------

The ``cross_entropy_forward`` kernel is designed for efficient computation of cross entropy loss in large vocabulary scenarios, such as language modeling. Traditional cross entropy implementations require loading the entire vocabulary for each position, which can be memory-intensive. This kernel uses an online log-sum-exp algorithm that processes the vocabulary in chunks, maintaining numerical stability while reducing memory requirements.

A companion ``cross_entropy_backward`` kernel computes gradients with respect to logits using the saved log-sum-exp state from the forward pass.

.. note::
    This kernel is optimized for Trainium2 (TRN2) and uses batched processing where each core processes multiple positions simultaneously with vectorized operations.

API Reference
--------------

**Source code for this kernel API can be found at**: `cross_entropy.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/loss/cross_entropy.py>`_

cross_entropy_forward
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: cross_entropy_forward(logits_hbm: nl.ndarray, targets_hbm: nl.ndarray, positions_per_batch: int = 32, chunk_size: int = 32768, dtype: nki.dtype = nl.bfloat16) -> tuple[nl.ndarray, nl.ndarray]

   Cross entropy forward pass using online log-sum-exp algorithm with batching.

   This kernel computes cross entropy loss for large vocabularies using a memory-efficient
   online log-sum-exp algorithm. Optimized for LNC2 (2 cores) with batched processing where
   each core processes multiple positions in batches with vectorized operations.

   :param logits_hbm: Input logits tensor in HBM with shape [num_positions, V]. Supported dtypes: nl.bfloat16, nl.float32. MUST be 2D (already flattened).
   :type logits_hbm: ``nl.ndarray``
   :param targets_hbm: Target indices tensor in HBM with shape [num_positions]. dtype: nl.int32. MUST be 1D (already flattened).
   :type targets_hbm: ``nl.ndarray``
   :param positions_per_batch: Number of positions to process together. Default: 32. Larger batches improve HBM bandwidth and SBUF utilization. Candidate values (powers of 2): 8, 16, 32, 64, 128. Must satisfy: positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB.
   :type positions_per_batch: ``int``
   :param chunk_size: Size of vocabulary chunks. Default: 32768 (32K). Must not exceed vocabulary size V or hardware limit (65535). Candidate values: 65535 (F_MAX, ideal for 128K-256K vocabs, bf16 only), 49152 (3/4 of F_MAX), 40960 (Good balance), 32768 (Standard, good for 32K-128K vocabs), 16384 (Half of 32K), 8192 (Quarter of 32K), 4096 (Small vocab fallback), 2048 (Minimum practical).
   :type chunk_size: ``int``
   :param dtype: Data type for internal computations. Default: nl.bfloat16. Supported types: nl.bfloat16 (2 bytes), nl.float32 (4 bytes). Controls precision of intermediate calculations and memory usage.
   :type dtype: ``nki.dtype``
   :return: A tuple containing: loss_hbm (Cross entropy loss per position in HBM with shape [num_positions], dtype matches dtype parameter), lse_state_hbm (Log-sum-exp values per position in HBM with shape [num_positions], dtype matches dtype parameter, saved for backward pass).
   :rtype: ``tuple[nl.ndarray, nl.ndarray]``

   **Notes**:

   * Batched version for LNC2 (2 cores): Each core processes multiple positions in batches
   * Positions assigned in strided pattern (core_id, core_id + 2, core_id + 4, ...)
   * Vectorized operations across batch dimension for efficiency
   * chunk_size must not exceed vocabulary size V
   * positions_per_batch must be in range (0, 128]
   * Per-allocation size constraint: positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB
   * Performance tuning: Increase positions_per_batch for better throughput (up to memory limit)
   * Performance tuning: Use larger chunk_size to reduce loop iterations (up to V and memory limit)

Implementation Details
-----------------------

The kernel implementation includes several key optimizations:

1. **Online Log-Sum-Exp Algorithm**: Processes vocabulary in chunks while maintaining running maximum and sum of exponentials to avoid numerical overflow.

2. **Batched Processing**: Each core processes multiple positions simultaneously using vectorized operations for improved throughput.

3. **Memory Efficiency**: Uses configurable chunk sizes to balance memory usage and computational efficiency.

4. **Load Balancing**: Distributes positions across cores in a strided pattern for optimal load distribution.

5. **Numerical Stability**: Maintains numerical stability through careful handling of maximum values and exponential computations.

**Chunk Size Selection Guide**:

* V ≤ 32K: Use chunk_size = V (single chunk)
* 32K < V ≤ 128K: Use chunk_size = 32768 or 40960
* 128K < V ≤ 256K: Use chunk_size = 65535 (bf16) or 32768 (fp32)
* Always verify: positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB

cross_entropy_backward
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: cross_entropy_backward(logits_hbm: nl.ndarray, targets_hbm: nl.ndarray, lse_state_hbm: nl.ndarray, reduction: str = "mean", positions_per_batch: int = 32, chunk_size: int = 32768, dtype: nki.dtype = nl.bfloat16, inplace: bool = True) -> nl.ndarray

   Cross entropy backward pass computing gradients with respect to logits.

   Computes the gradient of cross entropy loss with respect to input logits using the formula:
   ``grad_logits[i, j] = grad_scale * (softmax(logits[i, j]) - 1{j == target[i]})``
   where softmax is computed using the saved LSE state from the forward pass, and ``grad_scale``
   is determined by the reduction parameter.

   Optimized for LNC2 (2 cores) with batched processing where each core processes multiple
   positions in batches with vectorized operations.

   :param logits_hbm: Input logits tensor in HBM with shape ``[num_positions, V]``. Supported dtypes: ``nl.bfloat16``, ``nl.float32``. MUST be 2D (already flattened). Same tensor used in forward pass.
   :type logits_hbm: ``nl.ndarray``
   :param targets_hbm: Target indices tensor in HBM with shape ``[num_positions]``. dtype: ``nl.int32``. MUST be 1D (already flattened). Same tensor used in forward pass.
   :type targets_hbm: ``nl.ndarray``
   :param lse_state_hbm: Log-sum-exp values from forward pass in HBM with shape ``[num_positions]``. dtype matches ``dtype`` parameter. Saved state from ``cross_entropy_forward``.
   :type lse_state_hbm: ``nl.ndarray``
   :param reduction: How to scale gradients. ``'mean'``: scale by ``1/num_positions`` (matches PyTorch default). ``'sum'``: scale by ``1.0``. Default: ``'mean'``.
   :type reduction: ``str``
   :param positions_per_batch: Number of positions to process together. Default: 32. Must satisfy: ``positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB``.
   :type positions_per_batch: ``int``
   :param chunk_size: Size of vocabulary chunks. Default: 32768.
   :type chunk_size: ``int``
   :param dtype: Data type for internal computations. Default: ``nl.bfloat16``. Supported types: ``nl.bfloat16``, ``nl.float32``.
   :type dtype: ``nki.dtype``
   :param inplace: If ``True``, write gradients directly over ``logits_hbm`` to save HBM memory. Default: ``True``. When ``True``, ``logits_hbm`` is overwritten and cannot be used after.
   :type inplace: ``bool``
   :return: Gradient with respect to logits in HBM with shape ``[num_positions, V]``. If ``inplace=True``, this is the same tensor as ``logits_hbm``.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Uses the saved LSE state from ``cross_entropy_forward`` to compute softmax without recomputing the full forward pass
   * ``inplace=True`` saves ``num_positions × vocab_size × dtype_bytes`` of HBM memory
   * Same chunking and batching strategy as the forward pass for consistent performance