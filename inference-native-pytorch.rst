.. meta::
   :description: Native PyTorch inference on AWS Neuron — running native PyTorch models for inference on AWS Trainium, how the vLLM Neuron and vLLM Omni Neuron plugins support native PyTorch, and the torch-neuronx and libtorch-neuronx-lite packages.
   :keywords: AWS Neuron, Trainium, native PyTorch, TorchNeuron, torch-neuronx, libtorch-neuronx-lite, inference, vLLM Neuron, disaggregated inference, torch.compile, NEFF
   :date-modified: 2026-10-13

.. _inference-native-pytorch:

==================================
Native PyTorch inference on Neuron
==================================

Native PyTorch inference on AWS Trainium runs through TorchNeuron, which compiles the model to a
Neuron executable (NEFF) and runs it on the Neuron Runtime. The vLLM Neuron and vLLM Omni Neuron
plugins handle this compilation and provide a complete serving stack. Without a plugin, you
compile and run the model yourself and build the serving around it.

If the vLLM Neuron or vLLM Omni Neuron plugin already supports your model, use the
:ref:`vLLM Neuron <vllm-neuron-home>` or
:doc:`vLLM Omni Neuron </vllm-omni-neuron/docs/index>` documentation to deploy it.

This page is for running native PyTorch models for inference on Trainium without a plugin:

- Implementing a custom model for inference on Trainium directly on native PyTorch.
- Building a custom inference stack, or adapting a model-serving framework other than vLLM on
  Trainium.

How vLLM Neuron and vLLM Omni Neuron support native PyTorch
-----------------------------------------------------------

The :ref:`vLLM Neuron plugin <vllm-neuron-home>` serves LLMs and brings the full vLLM serving
stack to Trainium — continuous batching, paged KV cache, prefix caching, speculative decoding
(EAGLE3), structured outputs, multimodal support, parallelism strategies, and disaggregated
inference — through the standard ``vllm serve`` command and an OpenAI-compatible API. It
compiles each model to a NEFF through the TorchNeuron ``torch.compile`` backend.

vLLM Neuron supports disaggregated inference, which splits prefill and decode across separate
instances. Disaggregated inference uses a serving runtime provided by
**libtorch-neuronx-lite** — the TorchNeuron compile path (PyTorch model to NEFF) packaged with
that runtime. vLLM Neuron therefore runs on ``libtorch-neuronx-lite`` rather than
``torch-neuronx``.

The :doc:`vLLM Omni Neuron plugin </vllm-omni-neuron/docs/index>` serves diffusion and
multimodal generation models. It supports native PyTorch through the same
``libtorch-neuronx-lite`` package that vLLM Neuron uses.

Both plugins install ``libtorch-neuronx-lite`` automatically, so you do not install
or interact with it directly. For the full feature set and the design of each component, see the
:ref:`vLLM Neuron plugin documentation <vllm-neuron-home>`.

Running custom native PyTorch inference on Trainium
---------------------------------------------------

This covers running native PyTorch inference without the vLLM Neuron or vLLM Omni Neuron
plugins. A custom inference stack compiles the model to a NEFF through the TorchNeuron
``torch.compile`` backend, runs it, and implements the serving layer. The vLLM Neuron plugin
serves as a reference implementation.

Two packages provide the native PyTorch foundation. Which one you install depends on whether
you need disaggregated inference, and the package also determines the available execution modes.

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Package
     - When to use
     - Execution modes
   * - ``torch-neuronx``
     - Native PyTorch inference without disaggregated inference. The full native PyTorch
       (TorchNeuron) package, providing the ``neuron`` device and the ``torch.compile`` backend.
     - Eager mode and ``torch.compile``.
   * - ``libtorch-neuronx-lite``
     - Native PyTorch inference that uses disaggregated inference, in a custom stack or through
       the plugins. Provides the runtime required for disaggregated inference, and vendors the
       necessary TorchNeuron compile components into its own wheel rather than depending on
       ``torch-neuronx``, so it installs as a single self-contained package. This is the package
       the plugins use. For how the plugins build on the TorchNeuron ``torch.compile`` backend, see
       :doc:`How vLLM is integrated with TorchNeuron </vllm-neuron/docs/design/compilation/how_vllm_is_integrated_with_torchneuron>`.
     - ``torch.compile`` only. No eager execution on the Neuron device: eager operations fall
       back to the host CPU.

For the full set of TorchNeuron features and supported parameters, see the
:doc:`TorchNeuron reference </torch/reference/index>`.

Using libtorch-neuronx-lite in a custom inference stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your custom stack needs disaggregated inference, use ``libtorch-neuronx-lite``. Keep the
following in mind when you build on it:

- **Use** ``torch.compile`` **for all model execution.** ``libtorch-neuronx-lite`` runs models on
  the Neuron device only through ``torch.compile``; it has no eager execution on the device.
  Eager operations on ``neuron`` tensors fall back to the host CPU, which is slow, and some of
  them, such as non-contiguous views and dtype casts, raise an error. Move tensors to the device
  (for example, ``tensor.to("neuron:0")``) only as inputs to, and outputs from, compiled code.
- **Use the** ``gloo`` **backend for** ``torch.distributed``. Collectives inside a compiled model
  are compiled into the NEFF, so the process group is only needed for host-side coordination.
  Do not create a ``neuron`` process group: it belongs to ``torch-neuronx``. vLLM Neuron uses
  ``gloo`` for the same reason.
- **Select the native compile path and register its backend.** Set
  ``VLLM_NEURON_BACKEND=neuron_native`` in the environment before the process starts.
  ``libtorch-neuronx-lite`` reads this variable to select its native compile path, also in stacks
  that do not use vLLM. Importing ``libtorch_neuronx_lite`` does not register the
  ``torch.compile`` backend, so register it and pass the returned name:

  .. code-block:: python

     import torch
     import libtorch_neuronx_lite
     from libtorch_neuronx_lite.compile.native_backend import register

     compiled = torch.compile(model, backend=register())

- **Run compiled code with gradients disabled.** Use ``torch.no_grad()`` or
  ``torch.inference_mode()``, or call ``model.requires_grad_(False)``. With gradients enabled
  and parameters that require them, compilation fails with ``neuron backend doesn't support
  events``.
- **To reuse compiled graphs across restarts, save them and keep the grad mode fixed.** With
  ``VLLM_NEURON_BACKEND=neuron_native`` set, ``libtorch-neuronx-lite`` turns on the PyTorch
  precompile cache. After warmup, call ``torch.compiler.save_cache_artifacts()`` to write it.
  A restart reuses the saved graphs only when it runs them in the same grad mode as warmup:
  ``torch.no_grad()`` and ``torch.inference_mode()`` count as different modes. In any other
  mode, PyTorch traces and compiles the graph again without an error.

Installing both packages in one environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch-neuronx`` and ``libtorch-neuronx-lite`` can be installed in the same Python environment.
Both register the ``neuron`` device, and only one of them can own it in a given process.

``torch-neuronx`` loads automatically whenever a process runs ``import torch``. In a process that
uses ``libtorch-neuronx-lite``, including ``vllm serve`` with vLLM Neuron or vLLM Omni Neuron,
turn off that automatic loading:

.. code-block:: bash

   export TORCH_DEVICE_BACKEND_AUTOLOAD=0

Set the variable in the environment before the process starts. PyTorch reads it once, while
``import torch`` runs, so setting it later in Python code has no effect. Without it, importing
``libtorch-neuronx-lite`` fails with ``Tried to register multiple backend fallbacks for the same
dispatch key PrivateUse1``.

Processes that use ``torch-neuronx`` need no change: leave the variable unset.

Documentation
-------------

The TorchNeuron documentation covers the native PyTorch workflow on Trainium — setup,
quickstarts, the ``torch.compile`` pipeline and its supported parameters, operator coverage, and
troubleshooting. The vLLM Neuron and vLLM Omni Neuron documentation covers serving with the
plugins, which run on the same TorchNeuron compile backend.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: TorchNeuron (overview)
      :link: torch-neuron-home
      :link-type: ref
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      TorchNeuron, the ``PrivateUse1`` backend, and the eager and ``torch.compile`` execution
      modes.

   .. grid-item-card:: Get started with TorchNeuron
      :link: /torch/get-started/index
      :link-type: doc
      :class-card: sd-border-2

      :bdg-primary:`quickstart`
      ^^^
      Running a first model on Trainium with ``torch.compile``.

   .. grid-item-card:: torch.compile on Neuron
      :link: /torch/compile/index
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`deep dive`
      ^^^
      The Neuron ``torch.compile`` backend and its supported parameters.

   .. grid-item-card:: TorchNeuron reference
      :link: /torch/reference/index
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`reference`
      ^^^
      Operator coverage, environment variables, best practices, and troubleshooting.

   .. grid-item-card:: vLLM Neuron plugin
      :link: vllm-neuron-home
      :link-type: ref
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      Serving LLMs on Trainium with the full vLLM serving stack.

   .. grid-item-card:: vLLM Omni Neuron plugin
      :link: /vllm-omni-neuron/docs/index
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      Serving diffusion and multimodal generation models on Trainium.

Related
-------

- :ref:`Inference on Neuron <inference-home>` — choosing an inference path on Neuron.
- :ref:`vLLM Neuron plugin <vllm-neuron-home>` — the full vLLM serving stack on Trainium.
- :doc:`vLLM Omni Neuron </vllm-omni-neuron/docs/index>` — diffusion and multimodal serving.
- :ref:`TorchNeuron <torch-neuron-home>` — native PyTorch (``torch-neuronx``) on Neuron.
