.. meta::
   :description: GPT-OSS pre-training with TorchTitan on AWS Neuron. End-to-end mixture-of-experts pre-training on Trainium with NKI attention and MoE kernels.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, tutorial, GPT-OSS, MoE, NKI, pre-training, SimpleFSDP
   :date-modified: 2026-10-15

.. _torchtitan-neuron-tutorial-gpt-oss:

=====================================================
Tutorial: Pre-train GPT-OSS with TorchTitan Neuron
=====================================================

This tutorial walks you through pre-training GPT-OSS with TorchTitan on AWS
Trainium using TorchNeuron. When you have completed it, you will have run
GPT-OSS 20B sharded across a full Trainium node, with NKI kernels handling the
attention and mixture-of-experts layers.

Overview
--------

GPT-OSS is a mixture-of-experts model: each token is routed to a small subset
of expert feed-forward networks rather than passing through all of them. The
dense parts behave like any other transformer, but the MoE layers dominate both
the memory footprint and the communication pattern, and they are where the
Neuron-specific work shows up.

Two things make the Neuron path for GPT-OSS different from a stock TorchTitan
run. Attention uses a NKI kernel with sliding-window support rather than SDPA
or varlen attention, and each MoE layer is converted to a NKI implementation.
Both come from the recipe; neither is something you select by hand.

If you have not run TorchTitan on Trainium before, do
:doc:`/training/torchtitan/tutorials/tutorial-basic-training` first. It
establishes the launcher, the environment variables, and the tokenizer
workflow that this tutorial reuses without re-explaining.

Where the recipes live
~~~~~~~~~~~~~~~~~~~~~~

Neuron-specific recipes ship in the ``torchtitan_neuron`` package, installed
alongside ``torchtitan``:

.. code-block:: text

   torchtitan_neuron/
   ├── models/        Neuron Trainer recipes
   └── experiments/   GraphTrainer recipes, including SimpleFSDP

Because these modules are not in TorchTitan's built-in shorthand list, you
select them by **fully qualified module path** rather than a short name like
``gpt_oss``. Every launch command below does this.

Before you start
----------------

This tutorial assumes that you have experience in the following areas:

- Running TorchTitan on Neuron, verified by
  :doc:`/training/torchtitan/tutorials/tutorial-basic-training`.
- What a mixture-of-experts layer is: routing, top-k selection, and why expert
  load balancing matters.

Model details
-------------

This tutorial trains **GPT-OSS 20B**. Two Neuron recipes are available, and
they differ in how sharding is implemented rather than in what they train:

.. list-table::
   :header-rows: 1
   :widths: 30 44 26

   * - Recipe
     - Module
     - Sharding
   * - ``gpt_oss_20b_nki_tp4_fsdp16``
     - ``torchtitan_neuron.models.gpt_oss``
     - FSDP v2
   * - ``simple_fsdp_gpt_oss_20b_nki_tp4_fsdp16``
     - ``torchtitan_neuron.experiments.graph_trainer.gpt_oss``
     - SimpleFSDP

Both build the ``20b`` model flavor with the NKI attention backend, convert
every MoE layer with ``to_nki_moe_config``, apply the prefix-cache rotary path,
turn on ``torch.compile`` with the Neuron backend, pin tensor-parallel degree 4
with sharding filling the remaining cores, set expert-parallel degree 1, and
use full activation checkpointing.

.. note::

   The upstream ``gpt_oss`` registry ships recipes of its own —
   ``gpt_oss_debugmodel``, ``gpt_oss_debugmodel_flex``, ``gpt_oss_20b``, and
   ``gpt_oss_120b``. None of them do any of the above: they are the stock
   TorchTitan configurations, without the NKI kernels, the Neuron compile
   backend, or the rotary workaround Neuron needs. They are not substitutes for
   the two recipes here.

You will run both in `Step 2`_. Which one you would reach for afterwards depends
on what you are optimizing for:

- **FSDP v2** wraps modules with PyTorch's ``fully_shard``. It is the path the
  basic tutorial uses.
- **SimpleFSDP** instead represents sharding as collectives inside the
  computation graph, which keeps the whole step visible to the compiler.
  TorchTitan describes it as achieving memory and throughput improvements over
  eager FSDP2, and :doc:`/torch/guides/torch-distributed` recommends it on
  Neuron for ``torch.compile`` workloads. The trade is compile time: a larger
  graph takes longer to compile. One caveat on Neuron: it is supported only
  with the JIT compile mode, so do not override ``--compile.mode``. Runs
  print a deprecation warning for that mode; in this release that is expected
  rather than a problem.

:doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training` covers
SimpleFSDP in depth, including that limitation and where the recipes set it.

**Limitations:**

- Targets a single ``trn2.48xlarge`` node (64 NeuronCores). Both recipes shard
  with tensor-parallel degree 4 and data-parallel sharding filling the
  remaining cores.
- Expert parallelism is not used for GPT-OSS on Neuron. Both recipes scale with
  tensor parallelism and data-parallel sharding, and ship expert-parallel
  degree 1. Raising it fails — the NKI mixture-of-experts implementation shards
  experts with tensor parallelism only.
- ``gpt_oss_120b`` needs more than one node and is out of scope here.

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

- Trn2 or Trn3 instance. The commands below assume a ``trn2.48xlarge`` with
  64 NeuronCores.
- Neuron SDK 2.33+
- PyTorch 2.14
- ``torch-neuronx`` (native TorchNeuron) verified — run
  :doc:`/torch/get-started/quickstart-torch-neuron` and confirm
  ``device='neuron'`` resolves before continuing
- TorchTitan installed and verified by
  :doc:`/training/torchtitan/tutorials/tutorial-basic-training`

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

   `Step 2`_ runs ten steps on ``c4_test``, a small split bundled with the
   repository rather than the C4 corpus, because this tutorial is about
   confirming the path works, not about training a model. A loss curve from a
   run like that tells you the pipeline is wired correctly and little else.
   `Step 3 <Step 3: Confirm success and move to a real run_>`_ shows what to change for a real run.

Step 1: Prepare the tokenizer
------------------------------

The recipe reads its tokenizer from a small bundled path, so you can run
`Step 2`_ with no download at all. Do the download now only if you intend to
follow `Step 3 <Step 3: Confirm success and move to a real run_>`_ and train on real data:

.. code-block:: bash

   python scripts/download_hf_assets.py \
     --repo_id openai/gpt-oss-20b --assets tokenizer

Assets land in ``./assets/hf/gpt-oss-20b/``. The dataset needs no download
either: both recipes read ``c4_test``, a small split bundled with the
repository, so `Step 2`_ needs no network access at all. `Step 3 <Step 3: Confirm success and move to a real run_>`_ is where
that changes.

.. _Step 2:

Step 2: Launch the training run
-------------------------------

In this step, you will launch GPT-OSS 20B with NKI kernels on all 64
NeuronCores, once with each sharding implementation.

Start with FSDP v2:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.models.gpt_oss \
   CONFIG=gpt_oss_20b_nki_tp4_fsdp16 \
   ./run_train.sh --training.steps 10

Then SimpleFSDP. The model, the kernels, and the parallelism layout are the
same; the sharding implementation, the recipe module, and the batch size
change:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.experiments.graph_trainer.gpt_oss \
   CONFIG=simple_fsdp_gpt_oss_20b_nki_tp4_fsdp16 \
   ./run_train.sh --training.steps 10

Both should step, and loss should drop sharply at the start and keep trending
down. Expect it to bounce on the way — individual steps can rise while the
trend falls, so read the trend rather than each step. Getting both recipes to
run is the useful outcome here — the Neuron compile path works for this model
under either implementation.

.. note::

   The two recipes ship different batch sizes, so what you have just run is not
   a measurement of one against the other.

You do not pass ``--compile.enable``, ``--compile.backend``, an attention
backend, or the MoE conversion by hand. The recipe does all of it, and reading
it is the fastest way to see what the run is actually doing. In
``torchtitan_neuron/models/gpt_oss/config_registry.py``:

.. code-block:: python

   def gpt_oss_20b_nki_tp4_fsdp16() -> Trainer.Config:
       model_spec = model_registry("20b", attn_backend="nki")
       for layer in model_spec.model.layers:
           layer.moe = to_nki_moe_config(layer.moe)
           # Neuron does not support the RoPE gather; use the prefix-cache path.
           layer.attention.rope.use_prefix_cache_for_positions = True
       return Trainer.Config(
           compile=CompileConfig(
               enable=True,
               components=["model", "loss", "grad_clip"],
               backend="neuron",
           ),
           # ... remaining configuration unchanged
       )

The first three lines are the Neuron-specific part: the ``20b`` flavor built
with the ``nki`` attention backend, every MoE layer converted to the NKI
implementation, and the rotary workaround applied per layer. The rest of the
recipe is ordinary TorchTitan configuration.

Note which components get compiled: the model, the loss, and ``grad_clip``.
Gradient-norm clipping is compiled along with the rest rather than left to run
eagerly, which is what Neuron needs. The SimpleFSDP recipe inherits this,
because it is built by wrapping this one.

.. note::

   Expect the first step to take far longer than the ones after it. This recipe
   compiles the model with the Neuron backend, and the NKI attention and MoE
   kernels are compiled too. The Neuron stack caches the results on disk, so
   later runs of the same recipe start much faster. See
   :doc:`/torch/reference/env-vars` for the cache directories.

Step 3: Confirm success and move to a real run
-----------------------------------------------

In this step, you will confirm the `Step 2`_ runs were healthy, then launch a
longer run on real data.

Check them against the `Confirmation`_ list below before going further.

Moving to real data changes three things. The dataset becomes ``c4``, the asset
path moves to the tokenizer you downloaded in `Step 1 <Step 1: Prepare the tokenizer_>`_, and the step count has
to be set explicitly — these recipes ship a ten-step schedule, so dropping
``--training.steps`` leaves you at ten rather than handing you a longer run.

Run both recipes again, this time on real data. ``--training.global_batch_size
64`` is the one addition: the recipes ship different per-rank batch sizes, so
pinning the global batch keeps both runs training on the same number of
sequences per optimizer step. A real workload would choose its own batch size
rather than inherit this one.

FSDP v2:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.models.gpt_oss \
   CONFIG=gpt_oss_20b_nki_tp4_fsdp16 \
   ./run_train.sh \
     --training.steps 100 \
     --training.global_batch_size 64 \
     --dataloader.dataset c4 \
     --hf_assets_path=./assets/hf/gpt-oss-20b

SimpleFSDP:

.. code-block:: bash

   NGPU=64 \
   MODULE=torchtitan_neuron.experiments.graph_trainer.gpt_oss \
   CONFIG=simple_fsdp_gpt_oss_20b_nki_tp4_fsdp16 \
   ./run_train.sh \
     --training.steps 100 \
     --training.global_batch_size 64 \
     --dataloader.dataset c4 \
     --hf_assets_path=./assets/hf/gpt-oss-20b

.. note::

   You do not need to let either finish to complete the tutorial. Once you have
   watched the loss fall over a stretch of steps, stop the run with ``Ctrl-C``
   in the terminal running ``run_train.sh``, which signals the ``torchrun``
   launcher and shuts down every rank.

Training a mixture-of-experts model on a shortened learning rate schedule
carries risks to convergence, so do not read these runs as production training.
The first fifty or so steps swing hard — loss can move by several points
between consecutive steps — and then the curve settles and falls steadily, with
the step-to-step movement shrinking to a fraction of a point. Training with a
longer warmup over more steps gives more stable training. Both implementations
behave the same way here.

The GPT-OSS 20B configs on Neuron do not use expert parallelism; both scale
with tensor parallelism and FSDP.

Confirmation
------------

The `Step 2`_ runs succeeded if all of the following hold, for each of the two
recipes:

- Training starts stepping and per-step loss lines appear.
- Loss values are finite — never ``nan`` — and lower by the tenth step than
  the first.
- ``neuron-top`` shows the NeuronCores in use during the run.
- The second step completes far faster than the first, which is the one-time
  trace and compile cost being paid on step one and not again.

.. note::

   Each recipe compiles separately, so the SimpleFSDP launch pays its own
   first-step cost even after the FSDP v2 one has run. Expect it to be the
   longer of the two.

Congratulations! You have now pre-trained GPT-OSS with TorchTitan on AWS
Trainium. If you encountered any issues, see the **Common issues** section
below.

----

Common issues
-------------

- **Out of memory**: the recipe already uses full activation checkpointing, so
  the next lever is sequence length or batch size. Halve
  ``--training.seq_len`` before changing the model configuration. See
  :doc:`/training/torchtitan/best-practices`.
- **The run hangs at startup**: ``NGPU`` must match the NeuronCores actually
  available — 64 on a ``trn2.48xlarge``. Confirm with ``neuron-ls``.
- **"Config function not found"**: ``MODULE`` needs the path to the Neuron
  recipe module, not a short name. ``gpt_oss`` resolves to the upstream
  registry, which has no Neuron recipes.
- **Tokenizer or dataset mismatch**: the recipe points at a bundled test
  tokenizer by default. If you overrode ``--hf_assets_path``, confirm it matches
  the directory ``download_hf_assets.py`` created.
- **Compilation fails**: confirm you have not overridden the attention backend.
  The NKI attention path is selected by the recipe and is what the rest of the
  configuration assumes. For the SimpleFSDP recipe, also confirm you have not
  overridden ``--compile.mode``; JIT is the only mode supported for SimpleFSDP
  on Neuron in this release.
- **One recipe runs and the other does not**: they share the model, the
  kernels, and the parallelism layout, so this narrows the problem to the
  sharding implementation rather than to any of those.

Clean up
--------

Stop the training processes (``Ctrl-C`` on the ``torchrun`` launcher). Remove
the output TorchTitan wrote under ``./outputs`` if you no longer need it, and
stop or terminate the instance, since it bills while running.

.. include:: /includes/torch/trn3-sizing-note.txt

Next steps
----------

- :doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training` —
  the same sharding path with Llama and DeepSeek, in depth.
- :doc:`/training/torchtitan/best-practices` — tuning for GPT-OSS workloads.

Further reading
---------------

- :doc:`/training/torchtitan/tutorials/tutorial-basic-training`
- :doc:`/training/torchtitan/about-torchtitan-neuron`
