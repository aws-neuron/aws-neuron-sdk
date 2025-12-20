.. post:: December 16, 2025
    :language: en
    :tags: announce-eos-pytorch-profling-api

.. _announce-eos-pytorch-profling-api:

End of Support for PyTorch Experimental Profiling API starting in a future release
------------------------------------------------------------------------------------

What's changing
^^^^^^^^^^^^^^^^

Neuron will end support for the ``torch_neuronx.experimental.profiler.profile`` API in a future release of Neuron (planned for v2.29.0). This experimental API will be replaced by native PyTorch profiling support using the standard ``torch.profiler.profile()`` API.

How does this impact you
^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using ``torch_neuronx.experimental.profiler.profile,`` before April/May 2026:

* Update your code to use native PyTorch profiling API:

.. code-block:: python

    # Before (Experimental API)
    from torch_neuronx.experimental import profiler
    with profiler.profile(output_path="/tmp/profile") as prof:
        output = model(input)

    # After (Native API)
    import torch.profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.NEURON],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("/tmp/profile")
    ) as prof:
        output = model(input)

After Neuron 2.29.0 releases (planned):

* Experimental API will no longer be supported
* To continue using the experimental API, you must pin to Neuron SDK 2.28 or earlier (not recommended)

