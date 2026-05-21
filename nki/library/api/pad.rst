.. meta::
    :description: Pad a tensor with constant, replicate, reflect, or circular mode.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.pad

Pad Kernel API Reference
========================

Pad a tensor with constant, replicate, reflect, or circular mode.

Equivalent to ``torch.nn.functional.pad(x, padding, mode=mode, value=value)``.

Background
-----------

The ``pad`` kernel pads a tensor using one of four modes: constant, replicate, reflect, or circular. It follows PyTorch's ``torch.nn.functional.pad`` semantics with innermost-first padding specification.

API Reference
--------------

**Source code for this kernel API can be found at**: `pad.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/pad/pad.py>`_

pad
^^^

.. py:function:: pad(x_ref, padding, mode = 'replicate', value = 0)

   Pad a tensor with constant, replicate, reflect, or circular mode.

   :param x_ref: Input tensor (any number of batch dims + up to 3 spatial dims).
   :param padding: Padding amounts in PyTorch convention (innermost-first).
   :param mode: ``"constant"``, ``"replicate"``, ``"reflect"``, or ``"circular"``.
   :param value: Fill value for constant mode (default 0, ignored for other modes).

