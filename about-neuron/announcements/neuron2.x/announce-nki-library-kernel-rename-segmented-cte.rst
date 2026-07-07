.. post:: June 28, 2026
    :language: en
    :tags: announce-component-change, nki-library, nki

.. _announce-nki-library-kernel-rename-segmented-cte:

NKI Library renames the kv_parallel_segmented_prefill kernel starting with Neuron 2.31.0
-----------------------------------------------------------------------------------------

Starting with Neuron SDK 2.31.0, the NKI Library ``kv_parallel_segmented_prefill`` kernel has been renamed to ``attention_kv_parallel_segmented_cte``. The source file moves from ``core/attention/kv_parallel_segmented_prefill.py`` to ``core/attention/attention_kv_parallel_segmented_cte.py``.

Callers must update both their import path and the function name. For the kernel reference, see :doc:`Attention KV-Parallel Segmented CTE </nki/library/api/attention-kv-parallel-segmented-cte>`.
