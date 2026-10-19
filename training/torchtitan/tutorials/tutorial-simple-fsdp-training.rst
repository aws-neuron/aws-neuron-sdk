.. meta::
   :description: SimpleFSDP training with Llama and DeepSeek on TorchTitan-on-Neuron. Compiler-based sharding on Trainium with the Neuron JIT GraphTrainer path.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, tutorial, SimpleFSDP, FSDP, Llama, DeepSeek, pre-training, torch.compile, GraphTrainer
   :date-modified: 2026-10-15

.. _torchtitan-neuron-tutorial-simple-fsdp:

================================================================
Tutorial: Train Llama and DeepSeek with SimpleFSDP on Neuron
================================================================

This tutorial walks you through SimpleFSDP training with two model families —
Llama and DeepSeek — on AWS Trainium using TorchTitan with TorchNeuron. When
you have completed it, you will have run both models sharded by SimpleFSDP on
a full Trainium node.

Overview
--------

:doc:`/training/torchtitan/tutorials/tutorial-basic-training` already shards a
model with FSDP — its ``qwen3_8b_tp4_fsdp16_sac`` config runs TP4 × FSDP16 in
eager mode.

**SimpleFSDP** takes a different approach. Instead of wrapping modules with
PyTorch's ``fully_shard``, it expresses sharding as collectives inside the
traced graph. That makes the whole forward-backward step, communication
included, visible to the compiler as one graph, which is what lets the Neuron
backend schedule and optimize it, and is why
:doc:`/torch/guides/torch-distributed` recommends SimpleFSDP for
``torch.compile`` workloads on Neuron. The trade is compile time: a larger
graph takes longer to compile.

To choose between FSDP v2 and SimpleFSDP for a workload of your own rather than
follow this tutorial, that guide compares them and covers the rest of what
TorchNeuron supports through the ``neuron`` distributed backend.

Llama comes first, then DeepSeek. They are deliberately different cases: Llama
3.1 is a dense transformer, while DeepSeek-V3 is a mixture-of-experts model
whose MoE layers run on NKI kernels on Neuron. Running both shows the
SimpleFSDP workflow is not tuned to one architecture.

.. important::

   On Neuron, SimpleFSDP is currently supported **only with the JIT compile
   mode**. The recipes in this tutorial set that for you — the shared
   converter, ``neuron_simple_fsdp``, builds a
   ``GraphTrainerCompileConfig`` with ``mode="jit"``, and the recipe modules
   describe themselves as the Neuron JIT path. Do not override
   ``--compile.mode``; the other GraphTrainer modes are not supported on
   Neuron in this release. Every run prints a deprecation warning for
   ``mode="jit"``, because upstream TorchTitan is retiring it; in this release
   that warning is expected rather than a problem.

Where the recipes live
~~~~~~~~~~~~~~~~~~~~~~

Neuron-specific recipes ship in the ``torchtitan_neuron`` package, installed
alongside ``torchtitan``. SimpleFSDP recipes are under its ``experiments``
tree:

.. code-block:: text

   torchtitan_neuron/
   ├── models/        Neuron Trainer recipes (plain FSDP, eager and compiled)
   └── experiments/   GraphTrainer recipes, including SimpleFSDP

Because these modules are not in TorchTitan's built-in shorthand list, you
select them by **fully qualified module path** rather than a short name like
``llama3``. Every launch command below does this.

Before you start
----------------

Make sure you have:

- A working TorchTitan installation on Neuron, verified by
  :doc:`/training/torchtitan/tutorials/tutorial-basic-training`.
- Familiarity with FSDP on PyTorch — how sharding parameters, gradients, and
  optimizer state across ranks affects memory.

Model details
-------------

**Llama 3.1 8B** — module
``torchtitan_neuron.experiments.graph_trainer.llama3``:

.. list-table::
   :header-rows: 1
   :widths: 44 24 32

   * - Recipe
     - Activation checkpointing
     - Notes
   * - ``simple_fsdp_llama3_8b``
     - selective
     - Less recomputation, more activation memory. See the warning in `Step 1 <Step 1: Train Llama with SimpleFSDP_>`_.
   * - ``simple_fsdp_llama3_8b_fullac``
     - full
     - Same model, more recomputation. The baseline for this tutorial.

Both build on the upstream ``llama3_8b`` recipe and add the Neuron pieces: the
SDPA attention backend, the prefix-cache rotary path, ``torch.compile`` on the
Neuron backend, covering the model, the loss, and compiled gradient-norm
clipping.
Neither pins a parallelism degree, so SimpleFSDP shards across every core in
the run.

**DeepSeek-V3 16B** — module
``torchtitan_neuron.experiments.graph_trainer.deepseek_v3``:

.. list-table::
   :header-rows: 1
   :widths: 56 20 24

   * - Recipe
     - Activation checkpointing
     - Notes
   * - ``simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16``
     - selective
     - TP4 × FSDP16, NKI MoE kernels.
   * - ``simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16_fac``
     - full
     - Same recipe with full activation checkpointing.

Both derive from the Neuron dense recipe
``deepseek_v3_16b_neuron_tp4_fsdp16``, which is where the TP4 layout, the NKI
MoE conversion, and the expert load-balancing hook come from. The SimpleFSDP
versions then adjust the batch and sequence sizing.

**Limitations:**

- Targets a single ``trn2.48xlarge`` node (64 NeuronCores).
- JIT is the only supported compile mode for SimpleFSDP on Neuron in this
  release.
- The module also ships a ``simple_fsdp_llama3_70b_fullac`` config. The 70B
  needs far more memory than the 8B one, so this tutorial does not cover it.

.. note::

   The attention backend these recipes use applies causal attention but does
   not consume TorchTitan's packed-document boundaries. A sequence that holds
   more than one document therefore lets tokens attend across the boundary into
   earlier, unrelated documents. The Neuron recipes do not currently offer a
   backend that masks this, and the backend is fixed by the recipe rather than
   settable on the command line. If strict cross-document isolation matters for
   your workload, raise it before you build on this path.

Prerequisites
-------------

- Trn2 or Trn3 instance. The commands and the memory figures below assume a
  ``trn2.48xlarge`` with 64 NeuronCores.
- Neuron SDK 2.33+
- PyTorch 2.14
- ``torch-neuronx`` (native TorchNeuron) verified — run
  :doc:`/torch/get-started/quickstart-torch-neuron` and confirm
  ``device='neuron'`` resolves before continuing
- TorchTitan installed and verified by
  :doc:`/training/torchtitan/tutorials/tutorial-basic-training`
- A Hugging Face token with access to the Llama repositories, for the
  tokenizer download in `Step 3 <Step 3: Tune the run_>`_. DeepSeek needs no token.

----

Prepare your environment
------------------------

Work from the ``torchtitan-neuron`` directory you installed in
:doc:`/training/torchtitan/get-started/setup-torchtitan-neuron` and verified in
:doc:`/training/torchtitan/tutorials/tutorial-basic-training`.

Launches use the ``run_train.sh`` wrapper and the ``NGPU`` / ``MODULE`` /
``CONFIG`` environment variables introduced in the basic tutorial, under
"Neuron environment variables of interest". The only difference here is that
``MODULE`` takes a fully qualified path. Extra arguments pass straight through
to ``torchtitan.train``, which is how every override below works.

.. important::

   Steps 1 and 2 pin ``--dataloader.dataset c4_test`` and
   ``--training.steps 10``, and use a bundled tokenizer, because they are about
   confirming the path works, not about training a model. All three ship with
   the repository, so neither step needs network access. A loss curve from a
   run like that tells you the pipeline is wired correctly and little else.
   `Step 3 <Step 3: Tune the run_>`_ drops all three for a real run.

Step 1: Train Llama with SimpleFSDP
------------------------------------

In this step, you will launch a SimpleFSDP training run.

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.experiments.graph_trainer.llama3 \
   CONFIG=simple_fsdp_llama3_8b_fullac \
   ./run_train.sh --training.steps 10 --dataloader.dataset c4_test \
     --hf_assets_path=./tests/assets/tokenizer

Nothing to download, and no Hugging Face token: ``./tests/assets/tokenizer`` is
a tokenizer bundled with the repository, which is enough to tokenize the
bundled ``c4_test`` split. `Step 3 <Step 3: Tune the run_>`_ is where each model's own tokenizer comes
in.

.. warning::

   This step uses ``simple_fsdp_llama3_8b_fullac``, the full
   activation-checkpointing recipe, rather than the selective
   ``simple_fsdp_llama3_8b``. The selective recipe runs out of device memory at
   its default settings on a ``trn2.48xlarge``: with no parallelism override it
   resolves to TP1 × FSDP64, and the run fails during allocation before the
   first step completes.

   The selective recipe does run once you shrink the per-core footprint with
   tensor parallelism. `Step 3 <Step 3: Tune the run_>`_ shows it on a TP4 × FSDP16 layout, with
   measured memory for every combination that fits.

Apart from the dataset and step overrides, that is the whole change from a
plain FSDP run: a different recipe module and a different recipe name. You do
not pass ``--compile.enable``,
``--compile.backend``, ``--compile.mode``, or the Neuron overrides by hand.

It is worth reading the recipe that does set them, because it is short and it
is the authoritative answer to "what is this run actually doing?" In
``torchtitan_neuron/experiments/graph_trainer/llama3/config_registry.py``:

.. code-block:: python

   def _neuron_simple_fsdp_llama3_config(
       config_factory,
       *,
       full_ac: bool = False,
       converters: list[ModelConfigConverter.Config] | None = None,
   ) -> GraphTrainer.Config:
       base = config_factory()
       base.model_spec = model_registry(
           base.model_spec.flavor, attn_backend="sdpa", converters=converters
       )
       # Neuron does not support the RoPE gather; use the prefix-cache positions path.
       for layer in base.model_spec.model.layers:
           layer.attention.rope.use_prefix_cache_for_positions = True
       base.activation_checkpoint = FullAC.Config() if full_ac else SelectiveAC.Config()
       base.compile = CompileConfig(
           enable=True,
           components=["model", "loss", "grad_clip"],
           backend="neuron",
       )
       config = neuron_simple_fsdp(base, model_registry, model_name="simple_fsdp/llama3")
       return config

Reading it top to bottom: it starts from the upstream recipe
(``config_factory`` is ``llama3_8b``), pins SDPA attention, applies the
prefix-cache rotary path, chooses the activation-checkpointing mode, turns on
``torch.compile`` with the Neuron backend, and converts the whole thing to a
SimpleFSDP recipe.

Note which components get compiled: the model, the loss, and ``grad_clip``.
Gradient-norm clipping is compiled along with the rest rather than left to run
eagerly, which is what Neuron needs.

The JIT mode comes from that conversion step rather than from the recipe. In
``torchtitan_neuron/experiments/graph_trainer/configs.py``, ``neuron_simple_fsdp``
carries the base recipe's compile settings over and fixes the mode:

.. code-block:: python

   graph_config.compile = GraphTrainerCompileConfig(
       enable=compile_config.enable,
       components=list(compile_config.components),
       backend=compile_config.backend,
       mode="jit",
   )

That single ``mode="jit"`` is why the limitation above applies to every
SimpleFSDP recipe on Neuron, not just this one — they all go through this
converter.

.. note::

   Expect the first step to take far longer than the ones after it, and the
   first run to take longer than later runs. SimpleFSDP traces and compiles the
   whole step, and the Neuron stack caches the result on disk, so the cost is
   paid once per configuration. See :doc:`/torch/reference/env-vars` for the
   cache directories.

Step 2: Train DeepSeek with SimpleFSDP
---------------------------------------

In this step, you will run the same workflow against a mixture-of-experts
model to confirm the pattern generalizes.

Launch it:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.experiments.graph_trainer.deepseek_v3 \
   CONFIG=simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16 \
   ./run_train.sh --training.steps 10 --dataloader.dataset c4_test

Two things differ from the Llama run beyond the model. This recipe pins
tensor-parallel degree 4, so SimpleFSDP shards across 16 ranks rather than 64.
And every MoE layer runs on NKI kernels, with an optimizer hook that updates
the expert bias from the per-expert token counts each step — auxiliary-loss-free
load balancing. Neither is something you configure; both come with the recipe.

Step 3: Tune the run
---------------------

In the final step, you will move from a validation run to a real one and
adjust the two knobs that decide whether a configuration fits: activation
checkpointing and the parallelism layout. Llama comes first, then DeepSeek.

The runs so far have been ten-step checks on a bundled dataset and a bundled
tokenizer. The runs below drop all three: they read the real C4 corpus with
each model's own tokenizer, and follow the recipe's own schedule.

Download the tokenizers first. Llama repositories are gated, so that one needs
your Hugging Face token; DeepSeek needs none:

.. code-block:: bash

   python scripts/download_hf_assets.py \
     --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=<your-token>

   python scripts/download_hf_assets.py \
     --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer

Assets land in ``./assets/hf/Llama-3.1-8B/`` and
``./assets/hf/deepseek-moe-16b-base/``. The Llama recipe already expects the
first path, so its commands below pass no asset path; the DeepSeek commands
point at the second.

.. note::

   These are full training runs, so they take substantially longer than the
   earlier steps and the instance bills for the whole time.

   You do not need to let one finish to complete this tutorial. Once you have
   watched the loss fall over a stretch of steps, stop the run with ``Ctrl-C``
   in the terminal running ``run_train.sh`` — that signals the ``torchrun``
   launcher, which shuts down every rank.

Llama 3.1 8B
~~~~~~~~~~~~

Two knobs matter here. Activation checkpointing trades recomputation for
memory, and the parallelism layout changes how much of the model each core
holds. Both commands below move the run onto a TP4 × FSDP16 layout and differ
only in which recipe they use, so swapping ``CONFIG`` between them is the
activation-checkpointing knob on its own.

``data_parallel_shard_degree=-1`` tells SimpleFSDP to take whatever cores
tensor parallelism did not, so you set the tensor-parallel degree and let
sharding fill in the rest. Both commands also switch to ``c4``, which streams
the real C4 corpus from the Hugging Face Hub, and the asset path is already the
one you downloaded in Step 1.

They also pin ``--training.global_batch_size 64``. The recipes set a *local*
batch size — one sequence per rank — and leave the global batch to follow the
data-parallel degree, so moving from FSDP64 to TP4 × FSDP16 would quarter it
from 64 to 16 as a side effect. Pinning it keeps the batch fixed at 64
sequences and lets TorchTitan reach that with gradient accumulation, so the
only thing your override changes is the thing you meant to change.

Selective checkpointing on the TP4 × FSDP16 layout:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.experiments.graph_trainer.llama3 \
   CONFIG=simple_fsdp_llama3_8b \
   ./run_train.sh \
     --training.global_batch_size 64 \
     --dataloader.dataset c4 \
     --parallelism.tensor_parallel_degree=4 \
     --parallelism.data_parallel_shard_degree=-1

Full checkpointing on the same layout, which recomputes more and so holds less
in memory:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.experiments.graph_trainer.llama3 \
   CONFIG=simple_fsdp_llama3_8b_fullac \
   ./run_train.sh \
     --training.global_batch_size 64 \
     --dataloader.dataset c4 \
     --parallelism.tensor_parallel_degree=4 \
     --parallelism.data_parallel_shard_degree=-1

For a sense of what the knobs are worth, here is the peak device memory each
combination reached on a ``trn2.48xlarge``, out of 24 GiB per core:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Layout
     - Selective
     - Full
   * - TP1 × FSDP64
     - Out of memory
     - 11.91 GiB
   * - TP4 × FSDP16
     - 8.21 GiB
     - 4.71 GiB

Either knob buys a lot of headroom, and neither costs much throughput in these
runs. To see the layout difference for yourself, drop
``--parallelism.tensor_parallel_degree`` from the second command so it runs on
pure FSDP64; because the global batch is pinned, that is the only thing that
changes.

DeepSeek-V3 16B
~~~~~~~~~~~~~~~

DeepSeek needs neither of the overrides the Llama commands used: the recipe
already pins TP4 × FSDP16, and it sets a global batch size of its own. Two
things change instead. The dataset moves to ``c4``, and the asset path needs
redirecting — the recipe points at a bundled test tokenizer rather than the one
you downloaded in Step 2:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.experiments.graph_trainer.deepseek_v3 \
   CONFIG=simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16 \
   ./run_train.sh \
     --dataloader.dataset c4 \
     --hf_assets_path=./assets/hf/deepseek-moe-16b-base

The activation-checkpointing knob works the same way here. This recipe is the
selective one; ``simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16_fac`` is the full
counterpart, and swapping ``CONFIG`` is the whole change.

Confirmation
------------

The run succeeded if all of the following hold, for each model you ran:

- Training starts stepping and per-step loss lines appear.
- Loss values are finite — never ``nan`` — and trend downward over the first
  several steps.
- ``neuron-top`` shows the NeuronCores in use during the run.
- The second step completes substantially faster than the first, which is the
  one-time trace and compile cost being paid on step one and not again.

Congratulations! You have now trained both Llama and DeepSeek with SimpleFSDP
on AWS Trainium. If you encountered any issues, see the **Common issues**
section below.

----

Common issues
-------------

- **Out of memory**: if you are on a selective recipe, move to the ``fullac``
  variant of the same model. If you are already on ``fullac``, set a
  tensor-parallel degree as `Step 3 <Step 3: Tune the run_>`_ shows, which shrinks the per-core
  activation footprint. Reach for batch size last. See
  :doc:`/training/torchtitan/best-practices`.
- **The run hangs at startup**: ``NGPU`` must match the NeuronCores actually
  available — 64 on a ``trn2.48xlarge``. Confirm with ``neuron-ls``.
- **"Config function not found"**: ``MODULE`` needs the path to the Neuron
  recipe module, not a short name. ``llama3`` resolves to the upstream
  registry, which has no SimpleFSDP recipes.
- **Compilation fails**: confirm you have not overridden ``--compile.mode``.
  JIT is the only supported mode for SimpleFSDP on Neuron in this release.
- **The tokenizer download fails for Llama**: the repositories are gated. Pass
  ``--hf_token`` and confirm your account has been granted access.

Clean up
--------

Stop the training processes (``Ctrl-C`` on the ``torchrun`` launcher). Remove
the output TorchTitan wrote under ``./outputs`` if you no longer need it, and
stop or terminate the instance, since it bills while running.

.. include:: /includes/torch/trn3-sizing-note.txt

Next steps
----------

- :doc:`/training/torchtitan/tutorials/tutorial-gpt-oss-training` — GPT-OSS
  end-to-end, which also ships a SimpleFSDP recipe.
- :doc:`/training/torchtitan/best-practices` — tuning beyond the defaults.

Further reading
---------------

- :doc:`/training/torchtitan/tutorials/tutorial-basic-training`
- :doc:`/training/torchtitan/about-torchtitan-neuron`
- :doc:`/torch/guides/torch-distributed` — how FSDP and SimpleFSDP differ on
  Neuron, and when to reach for each.
