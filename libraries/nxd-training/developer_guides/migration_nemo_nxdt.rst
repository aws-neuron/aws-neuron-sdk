.. _nxdt_developer_guide_migration_nemo_nxdt:

Migrating from NeMo to Neuronx Distributed Training
===================================================

Neuronx Distributed Training (NxDT) is built on top of `NeMo-1.14 <https://github.com/NVIDIA/NeMo/tree/v1.14.0>`_.
The framework reuses modules from NeMo and exposes them via similar config interface.

.. note::

    At the moment, NxDT only allows running training of decoder LLM models.

This document goes over steps on how to migrate the NeMo training workload to NxDT training workload.

.. contents:: Table of contents
   :local:
   :depth: 2


Model Integration
------------------

**Model already Exists in NxDT Model Hub:**

If the model you want to train is already included in the NxDT model hub, and the training workflow
(e.g., pre-training, fine-tuning) is supported in NxDT, migrate your NeMo YAML configuration file to
the NxDT YAML file. Follow the mapping table in the :ref:`nxdt_nemo_nxdt_config_mapping`.

**Custom/New Model**

If your model is not part of the NxDT model hub, please use the guide
:ref:`nxdt_developer_guide_integrate_new_model`.


Dataloader Integration
----------------------

**Dataloader already exposed via one of the NxDT configs**

In this case, please map the NeMo YAML config parameters to NxDT config parameters using the
mapping table provided here :ref:`nxdt_nemo_nxdt_config_mapping`.

**Custom/New Dataloader**

If the dataloader is not part of the hub, please use the guide
:ref:`nxdt_developer_guide_integrate_new_dataloader`.

Optimizer/LR Scheduler Integration
----------------------------------

Since NxDT is built on top of NeMo, all the optimizers/LR schedulers provided by NeMo can be enabled
from the config.

Optimal Partitioning
--------------------

NxDT is built on top of
`NeuronxDistributed (NxD) <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html>`_
primitives and exposes different model parallelism techniques. All of them can be configured using
the ``distributed_strategy`` config.

Fusions/kernels
---------------

All the fused kernels available inside the NeMo config are not available in NxDT. This is because fused
kernels in NeMo are built specifically for GPUs. Neuron have a different set of kernels that can be
enabled from the config. Also, since Neuron uses a graph based approach, the compiler can optimize
some of the modules and do fusions wherever required.

Checkpoint Saving/loading
-------------------------

#.
   NeMo combines the model weights, optimizers and other state_dicts into a single ``state_dict``
   and dumps a file of the format: ``tp_rank_0*_pp_rank_00*/model_optim_rng.ckpt``. However, with NxDT, we
   save the model ``state_dict`` and the optimizer separately. The model statedict is saved in a folder
   of the form: ``model/dp_rank_00_tp_rank_00_pp_rank_00.pt`` and the optimizer is saved into a separate folder
   as: ``optim/dp_rank_00_tp_rank_00_pp_rank_00.pt``. This is mainly done so that when we use
   `zero1 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html?highlight=zero1#neuron-zero1-optimizer>`_,
   each DP rank can save its own optimizer shard.

#.
   NxDT doesn’t support ``.nemo`` style checkpoint saving. If users have a ``.nemo`` checkpoint, they would
   have to unpack it themselves and build a checkpoint conversion script to load the checkpoint into NxDT.

#.
   In NeMo, if we are using pipeline parallel, each pipeline stage creates an independent model. So
   lets say we have a model with 32 layers and we use PP=4, then NeMo would create 4 chunks with layers 0-7.
   So each PP rank would have a ``model_state_dict`` with keys going from layer-0-7. However, in NxDT, the model
   is created as a whole and then sharded. So the layer numbers are preserved.

#.
   One would have to write up a checkpoint conversion script similar to the checkpoint conversion from
   NeMo to NxDT.

For a more detailed mapping of NeMo parameters to NxDT parameters, follow the guide
:ref:`nxdt_nemo_nxdt_config_mapping`.

.. _nxdt_nemo_nxdt_config_mapping:

Config Mapping
--------------

Here is a detailed mapping for all the parameters in the config file. For the below mapping, we chose
the Llama example across both NeMo and NxDT frameworks. The same mapping is also true for other models.

.. csv-table::
   :file: nemo_nxdt_mapping.csv
   :header-rows: 1
   :widths: 20, 20, 40

.. note::

   For parameters that are not supported by NxDT, please create a feature request with specific use-case
   for the parameter, if needed.
