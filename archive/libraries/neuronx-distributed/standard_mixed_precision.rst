
.. _standard_mixed_precision:

Developer guide for Standard Mixed Precision
============================================

This document will introduce the concept of Standard Mixed Precision in NxD. It's
newly introduced in neuron release 2.20. It is recommended to use this setting for
training large models using NxD. When enabled, the optimizer will maintain a copy of
weights and their grads in FP32 data type.

.. note::
   Using this can increase memory pressure as we are using master weights and also performing
   optimiizer updates in higher precision. This can result in increased memory pressure and a
   slighly lower throughpout

Standard Mixed Precision offers few config settings that can be tuned by users

Compared to legacy mixed precision setting (i.e. before this feature's addition), Standard Mixed Precision
includes these components:

- Use FP32 for precision sensitive operators
- Use FP32 master weights and optimizer states for ZeRO-1 optimizer
- Use FP32 in local gradients accumulation
- Turn off stochastic rounding

.. note::
   The feature is tightly integrated with the :code:`NeuronZero1Optimizer`, to make
   Standard Mixed Precision take effect, ZeRO-1 optimizer needs to be enabled.

NxD Config Update
'''''''''''''''''

Newly introduced NxD config is as below:

.. code:: ipython3

   mixed_precision_config = {
       "use_master_weights": True,
       "use_fp32_grad_acc": True,
       "use_master_weights_in_ckpt": False,
   }

   config = {
       ...
       "mixed_precision_config": mixed_precision_config,
   }

In NxD training config, a new field :code:`mixed_precision_config` (default value is :code:`None`,
see details in the following sections) is added. It contains three sub-fields: :code:`use_master_weights`,
:code:`use_fp32_grad_acc`, and :code:`use_master_weights_in_ckpt`. Default value of
:code:`use_master_weights` and :code:`use_fp32_grad_acc` is whether ZeRO-1 optimizer is enabled.
Field :code:`use_master_weights` controls whether to use FP32 master weights. Field :code:`use_fp32_grad_acc`
controls whether to enable FP32 gradient accumulation buffer. Default value of :code:`use_master_weights_in_ckpt`
is :code:`False`. This field controls whether to save master weights in checkpoints.

.. code:: ipython3

   # same as `mixed_precision_config = None`
   mixed_precision_config = {
       "use_master_weights": optimizer_config["zero_one_enabled"],
       "use_fp32_grad_acc": optimizer_config["zero_one_enabled"],
       "use_master_weights_in_ckpt": False,
   }

   config = {
       ...
       "mixed_precision_config": mixed_precision_config,
   }

Note that only when ZeRO-1 optimizer is enabled, Standard Mixed Precision will take effect.

To disable this Standard Mixed Precision setting, just change NxD config:

.. code:: ipython3

   mixed_precision_config = {
       "use_master_weights": False,
       "use_fp32_grad_acc": False,
       "use_master_weights_in_ckpt": False,
   }

   config = {
       ...
       "mixed_precision_config": mixed_precision_config,
   }
