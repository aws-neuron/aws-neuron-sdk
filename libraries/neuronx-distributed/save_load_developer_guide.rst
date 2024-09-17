
.. _save_load_developer_guide:

Developer guide for save/load checkpoint
===================================================================

This document will introduce how to use `nxd.save_checkpoint` and `nxd.load_checkpoint`
to save and load checkpoint for distributed model training. This two methods handle all
checkpoint in a single method: model, optimize, learning rate scheduler and any user contents.

For a complete api guide, refer to :ref:`API GUIDE<api_guide>`.

Save checkpoint:
''''''''''''''''

A sample usage:

.. code:: ipython3

   nxd.save_checkpoint(
       args.checkpoint_dir,  # checkpoint path
       tag=f"step_{total_steps}",  # tag, sub-directory under checkpoint path
       model=model,
       optimizer=optimizer,
       scheduler=lr_scheduler,
       user_content={"total_steps": total_steps, "batch_idx": batch_idx, "cli_args": args.__dict__},
       use_xser=True,
       async_save=True,
   )

Users can choose to not save every thing. For example, model states only:

.. code:: ipython3

   nxd.save_checkpoint(
       args.checkpoint_dir,  # checkpoint path
       tag=f"step_{total_steps}",  # tag, sub-directory under checkpoint path
       model=model,
       use_xser=True,
       async_save=True,
   )

To only keep several checkpoints (e.g. 5), just use :code:`num_kept_ckpts=5`.

Load checkpoint:
''''''''''''''''

A sample usage, note that if no user contents detected, it will return ``None``:

.. code:: ipython3

   user_content = nxd.load_checkpoint(
       args.checkpoint_dir,  # checkpoint path
       tag=f"step_{args.loading_step}",  # tag
       model=model,
       optimizer=optimizer,
       scheduler=lr_scheduler,
   )

Leave ``tag`` not provided, this loading method will try to automatically resume from the
latest checkpoint.

.. code:: ipython3

   user_content = nxd.load_checkpoint(
       args.checkpoint_dir,  # checkpoint path
       model=model,
       optimizer=optimizer,
       scheduler=lr_scheduler,
   )
