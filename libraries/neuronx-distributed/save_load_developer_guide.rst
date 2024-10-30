
.. _save_load_developer_guide:

Developer guide for save/load checkpoint
========================================

This document will introduce how to use `nxd.save_checkpoint` and `nxd.load_checkpoint`
to save and load checkpoint for distributed model training. This two methods handle all
checkpoint in a single method: model, optimize, learning rate scheduler and any user contents.

Model states are saved on data parallel rank-0 only. When ZeRO-1 optimizer is not turned on,
optimizer states are also saved like this; while when ZeRO-1 optimizer is turned on, states
are saved on all ranks. Scheduler and user contents are saved on master rank only.

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

ZeRO-1 Optimizer State Offline Conversion:
''''''''''''''''''''''''''''''''''''''''''

ZeRO-1 optimizer checkpoint are sharded states stored for each rank. When user want to
load ZeRO-1 optimizer states with different cluster setting (e.g. with DP degree changed),
they can run the offline ZeRO-1 optimizer checkpoint conversion tool. This tool supports
conversion from sharded states to full states, from full to sharded, and from sharded to sharded.

.. code:: ipython3
   # sharded to sharded or full to sharded
   nxd_convert_zero_checkpoints --input_dir <input path> --output_dir <output path> --convert_to_sharded --dp_size <new dp degree>
   # sharded to full
   nxd_convert_zero_checkpoints --input_dir <input path> --output_dir <output path> --convert_to_full
