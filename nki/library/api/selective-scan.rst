.. meta::
    :description: Selective scan (SSM) as in Mamba models.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.scan

Selective Scan Kernel API Reference
===================================

Selective scan (SSM) as in Mamba models.

Performs fused discretization, linear recurrence, and output projection in a single kernel. For each state dimension n and time step t: decay[t] = exp(dt[t] * A[:, n]) inp[t] = dt[t] * x[t] * B[:, n, t] state[t] = decay[t] * state[t-1] + inp[t] y[t] += C[:, n, t] * state[t] y += D * x  (optional skip connection)

Background
-----------

The ``selective_scan`` kernel implements the selective scan state space model (SSM) used in Mamba models, performing fused discretization, linear recurrence, and output projection in a single kernel.

API Reference
--------------

**Source code for this kernel API can be found at**: `selective_scan.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/scan/selective_scan.py>`_

selective_scan
^^^^^^^^^^^^^^

.. py:function:: selective_scan(x: nl.ndarray, dt: nl.ndarray, A: nl.ndarray, B: nl.ndarray, C: nl.ndarray, D: nl.ndarray = None, initial_state: nl.ndarray = None) -> tuple

   Selective scan (SSM) as in Mamba models.

   :param x: Input tensor of shape [B_dim, channels, L].
   :type x: ``nl.ndarray``
   :param dt: Time step tensor of shape [B_dim, channels, L]. Should be positive.
   :type dt: ``nl.ndarray``
   :param A: State transition matrix of shape [channels, state_size]. Typically negative.
   :type A: ``nl.ndarray``
   :param B: Input projection matrix of shape [B_dim, state_size, L].
   :type B: ``nl.ndarray``
   :param C: Output projection matrix of shape [B_dim, state_size, L].
   :type C: ``nl.ndarray``
   :param D: Skip connection weights of shape [channels]. Default: None.
   :type D: ``nl.ndarray``
   :param initial_state: Initial hidden state of shape [B_dim, channels, state_size]. Default: None (zeros).
   :type initial_state: ``nl.ndarray``
   :return: (y, final_state) - y (nl.ndarray): Output tensor of shape [B_dim, channels, L] with same dtype as x. - final_state (nl.ndarray): Final hidden state of shape [B_dim, channels, state_size] in float32.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Uses float32 accumulation internally for numerical stability
   * A should be negative for stable recurrence (decay < 1)
   * dt should be positive; discretization computes exp(dt * A)
   * The scan is sequential along the L dimension but parallel across channels
   * Accumulation across state dimensions uses SBUF per free tile to avoid HBM read-modify-write (which requires trn2 shared memory)
   * Carries between free tiles are stored in the final_state HBM tensor

   **Dimensions**:

   * B_dim: Batch size
   * channels: Number of channels (partition dimension, tiled at P_MAX=128)
   * L: Sequence length (free dimension, tiled at F_TILE_SIZE=512)

