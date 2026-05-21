.. meta::
    :description: Retrieve the current RNG state from the GPSIMD engine.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.rng

Rng Kernel API Reference
========================

Retrieve the current RNG state from the GPSIMD engine.

Reads all 128 lanes of RNG state from the GPSIMD engine into SBUF, then copies only lane 0's seeds to a new output HBM tensor. Input shape range is constant [1, NUM_RNG_SEEDS]

Background
-----------

The ``get_rng_state_gpsimd`` kernel retrieves the current random number generator state from the GPSIMD engine by reading all 128 lanes of RNG state into SBUF and copying lane 0's seeds to an output HBM tensor.

API Reference
--------------

**Source code for this kernel API can be found at**: `rng.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/rng/rng.py>`_

get_rng_state_gpsimd
^^^^^^^^^^^^^^^^^^^^

.. py:function:: get_rng_state_gpsimd(tensor_state: nl.ndarray)

   Retrieve the current RNG state from the GPSIMD engine.

   :param tensor_state: [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor used only for shape/dtype reference.
   :type tensor_state: ``nl.ndarray``
   :return: [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor containing the 6 RNG seeds from lane 0.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * L: Number of GPSIMD lanes (128)

set_rng_state_gpsimd
^^^^^^^^^^^^^^^^^^^^

.. py:function:: set_rng_state_gpsimd(tensor_state: nl.ndarray)

   Set the RNG state for the GPSIMD engine by broadcasting seeds to all lanes.

   :param tensor_state: [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor containing the 6 seeds to broadcast.
   :type tensor_state: ``nl.ndarray``
   :return: [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor echoing back the seeds that were set.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * L: Number of GPSIMD lanes (128)

generate_random
^^^^^^^^^^^^^^^

.. py:function:: generate_random(output: nl.ndarray, n_elements: int)

   Generate random int32 values, tiling to fit SBUF.

   :param output: [1, n_elements], dtype int32, HBM tensor to be filled with random values.
   :type output: ``nl.ndarray``
   :param n_elements: Number of random int32 values to generate.
   :type n_elements: ``int``
   :return: [1, n_elements], dtype int32, HBM tensor filled with random values.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Uses sequential_range (not affine_range) due to loop-carried RNG state dependency
   * Remainder tile is handled separately after full tiles

   **Dimensions**:

   * N: Number of random elements to generate (n_elements)

