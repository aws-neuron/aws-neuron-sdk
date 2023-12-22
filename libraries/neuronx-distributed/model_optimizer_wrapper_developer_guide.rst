.. _model_optimizer_wrapper_developer_guide:

Developer guide for model and optimizer wrapper (``neuronx-distributed`` )
==========================================================================

Model and optimizer wrapper are useful tools to wrap original model and optimizer
while keep the API unchanged. We recommend to always use model and optimizer wrappers,
it's helpful to apply optimizations and hide the complexity from the optimizations.
Users need to care about the implementation details of the optimization, just use
the wrappers as you normally use ``torch.nn.Module`` and ``torch.optim.Optimizer``.

For a complete api guide, refer to :ref:`API GUIDE<api_guide>`.

Create training config:
'''''''''''''''''''''''

To use model and optimizer wrapper, we need to create ``neuronx_distributed``
config firstly.

A sample config use tensor parallel, pipeline parallel, ZeRO-1 optimizer,
sequence parallel and activation checkpointing:

.. code:: ipython3

   nxd_config = nxd.neuronx_distributed_config(
       tensor_parallel_size=args.tensor_parallel_size,
       pipeline_parallel_size=args.pipeline_parallel_size,
       pipeline_config={
           "transformer_layer_cls": LlamaDecoderLayer,
           "num_microbatches": args.num_microbatches,
           "output_loss_value_spec": (True, False),
           "input_names": ["input_ids", "attention_mask", "labels"],
           "pipeline_cuts": pipeline_cuts,
           "trace_file_path": args.trace_file_path,
           "param_init_fn": None,
           "leaf_module_cls": [LlamaRMSNorm.__name__],
           "autowrap_modules": [mappings],
           "use_zero1_optimizer": args.use_zero1_optimizer > 0,
           "use_optimizer_wrapper": True,
       },
       optimizer_config={
           "zero_one_enabled": args.use_zero1_optimizer > 0,
           "grad_clipping": True,
           "max_grad_norm": 1.0,
       },
       sequence_parallel=args.use_sequence_parallel,
       activation_checkpoint_config=CoreAttention if args.use_selective_checkpoint > 0 else "full",
       model_init_config=model_init_config,
   )

Use model wrapper:
''''''''''''''''''

When we wrap a model with model wrapper, we need to implement a model getter
function. The model getter function will be called to initialize model on CPU and
then model will be moved to XLA device serially. Then, let's pass ``nxd_config``,
model getter function and its inputs to method ``initialize_parallel_model``:

.. code:: ipython3

   model = nxd.initialize_parallel_model(nxd_config, get_model, config)

If pipeline parallel is enabled, to run a training iteration, user must use
``run_train``, it handles pipeline partitioned forward and backward in it:

.. code:: ipython3

   loss = model.run_train(*inputs)

Otherwise, users can use either ``run_train`` or:

.. code:: ipython3

   loss = model(*inputs)
   loss.backward()

To access the wrapped model:

.. code:: ipython3

   model.local_module()

Model wrapper also has short cuts to access some common fields of hugging
face transformers model;

.. code:: ipython3

   model.dtype  # get model's dtype
   model.config  # get model's config
   model.name_or_path  # get model's name or path

Use optimizer wrapper:
''''''''''''''''''''''

When we wrap an optimizer with optimizer wrapper, we need ``nxd_config``,
original optimizer class and its inputs (parameters and optimizer arguments):

.. code:: ipython3

   optimizer = nxd.initialize_parallel_optimizer(
       nxd_config, torch.optim.AdamW, param_groups, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
   )

One useful feature is that user can access grad norm value from wrapped optimizer
directly:

.. code:: ipython3

   # It's a XLA tensor
   optimizer.grad_norm

Note that if optimizer has not been executed or ``grad_clipping`` is disable,
access ``grad_norm`` will get ``None``.
