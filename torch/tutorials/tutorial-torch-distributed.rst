.. meta::
   :description: Train with distributed PyTorch on Trainium2: matched 64-rank CPU and Neuron DDP, FSDP2, and TP examples, then compiled SimpleFSDP plus TP.
   :keywords: TorchNeuron, torch-neuronx, AWS Neuron, Trainium2, tutorial, torch.distributed, FSDP2, SimpleFSDP, torch.compile, DTensor, tensor parallelism, DDP, 64 ranks
   :date-modified: 2026-10-15

.. _torchneuron-tutorial-torch-distributed:

==============================================================
Tutorial: Train with DDP, FSDP2, and DTensor on Neuron
==============================================================

Run the same training job three ways on a full ``trn2.48xlarge`` instance:
DDP, FSDP2, and DTensor tensor parallelism. Each job uses 64 workers and
records its outputs, gradients, and parameter updates. Run the same distributed
job on CPU with Gloo, then on Neuron, and compare the collected results rank by rank.
Then combine SimpleFSDP and TP inside compiled forward and backward graphs.

Overview
--------

Use the same launch script, rank count, parallelism plan, and training step
for the CPU and Neuron jobs. Change ``--device`` and ``--backend`` between runs.
Both jobs use the same model, global batch, optimizer, and initial parameters.
Compiled jobs also change the compiler backend for the selected device.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Mode
     - What each worker owns
     - Training data per worker
   * - DDP
     - A complete model replica; gradients are averaged across workers.
     - Eight samples from the global batch.
   * - FSDP2
     - Parameter and gradient shards; layers gather weights when needed.
     - Eight samples from the global batch.
   * - DTensor / TP
     - A partition of each linear layer.
     - The same global batch of 512 samples on every worker.

Before you start
----------------

Read :doc:`How to use torch.distributed on Neuron
</torch/guides/torch-distributed>` first. That guide explains process-group
initialization, rank assignment, parallelism choices, EFA for multi-node
execution, and HostCC overlap.

Complete the single-device check in
:doc:`/torch/get-started/quickstart-torch-neuron`. You should also be familiar
with a PyTorch training loop: forward, loss, backward, and optimizer step.

Model details
-------------

- Two linear layers: 128 input features, 256 hidden features, and 128 outputs.
- ReLU activation and mean squared error loss.
- Float32 parameters and deterministic synthetic data.
- SGD with a learning rate of ``0.01``.
- Ten training steps, seed ``0``, and a global batch of 512 samples.

The comparison tool checks separately collected CPU and Neuron distributed results.
The training scripts materialize parameter shards for verification. Use
these checks to establish correctness before performance tuning.
A per-rank ``PASS`` line confirms that the run completed. Each comparison must
report ``PASS comparison ranks=64 steps=10`` to establish numerical correctness.

Prerequisites
-------------

- Exclusive access to a ``trn2.48xlarge`` with all 64 logical NeuronCores
  available under logical NeuronCore configuration 2.
- A native TorchNeuron environment with ``torch-neuronx`` and PyTorch 2.14.
- A successful single-device run.

For multi-node training beyond this single-node tutorial, install EFA on every
node. Follow **Step 3: Install EFA** in the `Neuron manual setup instructions
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html#installation-steps>`_,
which explicitly cover Trn2.

Prepare your environment
------------------------

Activate your TorchNeuron environment, then make all cores available to the
launcher. Set the logical-core configuration before starting Python.

.. code-block:: bash

   unset NEURON_RT_VISIBLE_CORES NEURON_RT_NUM_CORES TORCH_NEURONX_ENABLE_STREAMS
   export NEURON_LOGICAL_NC_CONFIG=2
   export TORCH_NEURONX_ENABLE_HOST_CC=0
   export OMP_NUM_THREADS=1
   neuron-ls
   mkdir -p neuron-distributed-tutorial
   cd neuron-distributed-tutorial

Confirm that ``neuron-ls`` reports logical-core configuration 2, core IDs
``0`` through ``63``, and no other processes using the cores.
Streams are enabled by default; the command above removes any previous
streams override. These correctness runs keep HostCC disabled. Follow the
distributed guide to enable HostCC overlap after establishing parity.

Step 1: Save the shared code and run distributed CPU references
---------------------------------------------------------------

Download these files into the current directory:

- :download:`training_common.py <examples/training_common.py>`: model, data,
  training loop, and editable configuration defaults.
- :download:`distributed_train.py <examples/distributed_train.py>`: distributed
  DDP, FSDP2, and TP launch script for both CPU and Neuron.
- :download:`compare_training.py <examples/compare_training.py>`: result comparison.
- :download:`reference_train.py <examples/reference_train.py>`: optional CPU/Gloo
  entry point. It calls the same distributed trainer with CPU defaults.

CPU reference execution is a separate distributed job. Each CPU and Neuron
worker saves its own results, and the comparison tool checks matching ranks
after training. These saved tensors are correctness artifacts,
not throughput measurements.

The device, process-group backend, rank count, TP degree, seed, and training
steps are configurable at the top of ``training_common.py``:

.. literalinclude:: examples/training_common.py
   :language: python
   :start-at: # Edit these defaults
   :end-at: LEARNING_RATE =

The launch scripts also accept ``--device``, ``--backend``,
``--expected-world-size``, ``--seed``, and ``--steps``. Set device and backend
options together: ``--device cpu --backend gloo`` for the reference and
``--device neuron --backend neuron`` for Neuron. Both jobs use 64 ranks.
Compiled scripts also accept ``--compile-backend``: use ``aot_eager`` on CPU
and ``neuron`` on Neuron. The CPU job captures the same forward and backward
graphs with PyTorch's AOTAutograd backend.

``make_model`` calls ``torch.manual_seed(seed)`` before creating CPU-initialized
parameters. Both independent runs use seed ``0``, the same deterministic
synthetic data, and the same optimizer. Generate a ten-step distributed CPU
reference for each eager mode:

.. code-block:: bash

   for mode in ddp fsdp tp; do
       torchrun --standalone --nproc-per-node=64 distributed_train.py \
           --parallelism "$mode" --device cpu --backend gloo \
           --seed 0 --steps 10 --output-dir "results/cpu-$mode"
   done

The CPU reference uses the same DDP wrapper, FSDP2 sharding, or TP plan as its
Neuron counterpart. It performs distributed forward and backward computation
and collective communication. Use PyTorch 2.14 for both jobs.
``reference_train.py`` is a convenience alias with CPU/Gloo defaults, for example:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=64 reference_train.py \
       --parallelism ddp --seed 0 --steps 10 --output-dir results/cpu-ddp

The shared training loop is:

.. literalinclude:: examples/training_common.py
   :language: python
   :pyobject: train

The distributed configuration is:

.. literalinclude:: examples/distributed_train.py
   :language: python

For DDP and FSDP2, each rank receives a different eight-sample slice of the
global batch. Averaging those workers' gradients matches the gradient of
the CPU loss over all 512 samples. For TP, every rank receives the complete
batch; the layers partition the feature dimension.

Step 2: Train with DDP
----------------------

Launch 64 workers, one per logical NeuronCore:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=64 \
       distributed_train.py --parallelism ddp --seed 0 --steps 10 \
       --output-dir results/ddp
   python compare_training.py --reference-dir results/cpu-ddp \
       --actual-dir results/ddp --world-size 64

``DDP(model)`` creates a model replica on each worker and reduces gradients
during backward. The comparison tool checks the recorded averaged gradients and SGD updates
against the matching distributed CPU/Gloo job.

Every worker should complete ten steps and print
``PASS rank=<rank> mode=ddp world_size=64``. This line confirms that the run
finished. The comparison is the numerical correctness check and must report
``PASS comparison ranks=64 steps=10`` before you continue.

Step 3: Train with FSDP2
------------------------

Run the same workload with sharded parameters:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=64 \
       distributed_train.py --parallelism fsdp --seed 0 --steps 10 \
       --output-dir results/fsdp
   python compare_training.py --reference-dir results/cpu-fsdp \
       --actual-dir results/fsdp --world-size 64

The script applies ``fully_shard`` to each linear layer, then to the root
model, before creating the optimizer. Layers ``0`` and ``2`` are linear
layers with parameters; layer ``1`` is ReLU and has no parameters to shard.
Each worker owns four rows of the
first layer's weight and two rows of the second layer's weight.

The CPU and Neuron jobs use the same mesh and layer sharding.
The script records full gradients and parameters for the separate comparison.
Every worker should finish with
``PASS rank=<rank> mode=fsdp world_size=64``.

Step 4: Train with DTensor tensor parallelism
---------------------------------------------

Run the same model with partitioned linear layers:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=64 \
       distributed_train.py --parallelism tp --seed 0 --steps 10 \
       --output-dir results/tp
   python compare_training.py --reference-dir results/cpu-tp \
       --actual-dir results/tp --world-size 64

``ColwiseParallel`` partitions the first layer's 256 output features into
four features per rank. ``RowwiseParallel`` consumes that partitioned input
in the second layer and returns a complete output on every worker.

.. note::

   This step uses all 64 ranks as one TP group to check the sharding path.
   For training workloads, start with chip-local TP4 and use the remaining
   ranks for data parallelism, as in Step 5. Wider TP needs a supported rank
   placement and measured benefit for your model. See
   :doc:`Best practices for training on AWS Neuron </training/best-practices>`
   for topology-aware mesh choices, including a non-default TP8 layout.

All workers process the same global batch. The comparison tool checks their
outputs, full gradients, and updates against the matching CPU TP ranks.
Every worker should finish with ``PASS rank=<rank> mode=tp world_size=64``.

Step 5: Compile SimpleFSDP and TP together
------------------------------------------

SimpleFSDP represents parameter gathering and gradient reduction as
DTensor operations inside the model computation. The compiler can capture
those collectives together with tensor-parallel computation.

Download :download:`simplefsdp_tp_train.py <examples/simplefsdp_tp_train.py>`.
This complete example uses a ``16 x 4`` mesh: 16 FSDP ranks and four TP ranks,
for 64 workers total. It uses a bias-free version of the two-layer model,
and records ten steps for comparison with the same distributed script on CPU.
Both jobs use the same mesh and SimpleFSDP parameter getter.
The shared model factory applies the same seed before initialization.

Each worker stores one sixty-fourth of both weights:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Parameter
     - DTensor placements (FSDP, TP)
     - Local shape
   * - First weight (256 x 128)
     - ``(Shard(1), Shard(0))``
     - ``(64, 8)``
   * - Second weight (128 x 256)
     - ``(Shard(0), Shard(1))``
     - ``(8, 64)``

The parameter getter gathers each TP partition across the FSDP dimension.
Its backward operation sums and reduce-scatters gradients to the stored
shards. Checkpointing the getter recomputes gathers needed by backward,
rather than retaining the gathered weights.

.. literalinclude:: examples/simplefsdp_tp_train.py
   :language: python
   :pyobject: SimpleFSDPWeight

The complete model is compiled with
``torch.compile(sharded, backend="neuron", fullgraph=True)``. The captured
forward includes FSDP all-gathers, both linear layers, ReLU, and the TP
output all-reduce. AOTAutograd generates the backward graph, including
FSDP gradient reduce-scatter. The optimizer and result collection run outside these graphs. CPU training
and result comparison run in separate jobs.

Launch the complete script:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=64 simplefsdp_tp_train.py \
       --device cpu --backend gloo --compile-backend aot_eager \
       --seed 0 --steps 10 --output-dir results/cpu-simplefsdp-tp
   torchrun --standalone --nproc-per-node=64 simplefsdp_tp_train.py \
       --seed 0 --steps 10 --output-dir results/simplefsdp-tp
   python compare_training.py --reference-dir results/cpu-simplefsdp-tp \
       --actual-dir results/simplefsdp-tp --world-size 64

Each data-parallel replica processes a different 32-sample slice of the
global batch; its four TP workers receive the same slice. The script divides
the local mean loss by 16 before backward because the parameter getter uses
sum reduction across FSDP workers.

Confirm ten completed steps and
``PASS rank=<rank> mode=simplefsdp_tp world_size=64`` on every worker, then
``PASS comparison ranks=64 steps=10``. The separate comparison checks outputs,
full gradients, updated parameters, update deltas, and global loss.

The comparison tool prints each step's global CPU and Neuron loss after
checking every rank. The bias-free model differs from the eager model
in Steps 2 to 4.

This compact example follows the DTensor parameter-gather pattern in
`TorchTitan's SimpleFSDP implementation <https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/simple_fsdp.py>`_.
For the upstream compilation workflow, see the
`Graph Trainer examples <https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/README.md>`_.

Confirmation
------------

For each Neuron job in Steps 2 to 5, confirm all of the following:

- The launcher exits with status 0.
- Ranks ``0`` through ``63`` each print a final ``PASS`` line.
- Each rank completes ten training steps; the comparison reports
  ``parity=passed`` against the matching CPU rank for all 64 ranks.
- The global losses match the corresponding CPU job within the script's
  numerical tolerances. For the eager modes in Steps 2 to 4, they also follow
  the same trajectory across modes.

For each recorded step, the comparison checks outputs, gradients, updated
parameters, update deltas, and global loss. The FSDP2 and TP setup also checks
local weight shapes to confirm that parameters are sharded.

Each comparison prints a ten-step table with CPU and Neuron global losses.
Use the table for the matching mode and mesh.
Small numerical differences within the script's tolerances are acceptable.

You have now run a training job across all 64 logical NeuronCores using
DDP, FSDP2, and tensor parallelism.

Common issues
-------------

- **A worker cannot allocate a core:** Check for other Neuron processes with
  ``neuron-ls``. Remove inherited core visibility limits and use logical-core
  configuration 2 before launching 64 workers.
- **Workers hang in a collective:** Check that every rank runs the same mode
  and step count. Keep batch shapes and collective order consistent.
- **A parity assertion fails after changing the data:** DDP/FSDP2 workers
  must receive different, equal-sized slices of the global batch. TP workers
  must receive the same complete batch. Use the same data and mesh in both jobs.
- **A sharding assertion fails:** Keep the layer dimensions divisible by the
  selected mesh dimensions. The defaults in this tutorial use 64 workers.

For more diagnoses, see :doc:`/torch/troubleshooting` and
:doc:`/torch/guides/torch-distributed`.

Clean up
--------

The script destroys its process group in ``finally`` and the workers exit.
If you provisioned an instance for this tutorial, stop it when you finish.

Next steps
----------

- :doc:`/training/best-practices` -- Choose TP, FSDP, and hybrid meshes for
  training workloads.
- :doc:`/torch/guides/torch-distributed` -- Configure FSDP prefetching, HostCC
  streams, pipeline parallelism, or multi-node execution.
- `torch-neuronx operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`__ -- Check supported APIs
  and their limits before adapting the example.
- :doc:`/torch/reference/best-practices-pytorch-neuron` -- Apply performance
  and accuracy practices to a larger workload.
