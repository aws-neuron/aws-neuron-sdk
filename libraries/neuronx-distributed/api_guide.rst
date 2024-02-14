.. _api_guide:

API Reference Guide (``neuronx-distributed`` )
===============================================

NeuronX Distributed (NxD) is XLA based library for distributed training and inference on Neuron devices.
As part of this library, we support 3D parallelism: Tensor-Parallelism, Pipeline-Parallelism
and Data-Parallelism. We also support Zero1 optimizer to shard the optimizer weights.
To support tensor-parallelism on Neuron, we adopted the Apex Library
built for CUDA devices. We modified the implementations to work with
XLA. This document enlist the different APIs and modules provided by the library

Parallel Model State:
^^^^^^^^^^^^^^^^^^^^^

Initialize Model Parallelism:
'''''''''''''''''''''''''''''

::

   def neuronx_distributed.parallel_state.initialize_model_parallel(
       tensor_model_parallel_size=1,
       pipeline_model_parallel_size=1,
   )

This module would initialize the distributed model training and allows
users to set the number of tensor_parallel world size.

Parameters:

- ``tensor_model_parallel_size`` : This should set the number of tensor
  parallel workers. Note the default value is set to 1
- ``pipeline_model_parallel_size`` : This should set the number of pipeline
  parallel workers. Note the default value is set to 1

Other helper APIs:
''''''''''''''''''

-  ``neuronx_distributed.parallel_state.get_data_parallel_size()`` :
   Returns the data parallel world size depending on the number of
   global workers and tensor parallel workers.
-  ``neuronx_distributed.parallel_state.get_tensor_model_parallel_size()``
   : Returns the tensor parallel world size.
-  ``neuronx_distributed.parallel_state.get_tensor_model_parallel_rank()``
   : Returns the rank of the worker within the tensor parallel group
-  ``neuronx_distributed.parallel_state.get_pipeline_model_parallel_size()``
   : Returns the pipeline parallel world size.
-  ``neuronx_distributed.parallel_state.get_pipeline_model_parallel_rank()``
   : Returns the rank of the worker within the pipeline parallel group
-  ``neuronx_distributed.parallel_state.get_data_parallel_rank()`` :
   Returns the rank of the worker in the data parallel group.
-  ``neuronx_distributed.parallel_state.get_data_parallel_group(as_list=False)``
   : Returns the data parallel group after taking into account the
   tensor parallel size and the global world size. as_list argument when
   set to True, would return the group as a List[List] otherwise it
   would return a torch.distributed.group.
-  ``neuronx_distributed.parallel_state.get_tensor_model_parallel_group(as_list=False)``
   : Returns the tensor parallel group after taking into account the
   tensor parallel size and the global world size. as_list argument when
   set to True, would return the group as a List[List] otherwise it
   would return a torch.distributed.group.
-  ``neuronx_distributed.parallel_state.get_pipeline_model_parallel_group(as_list=False)``
   : Returns the pipeline parallel group after taking into account the
   pipeline parallel size and the global world size. as_list argument when
   set to True, would return the group as a List[List] otherwise it
   would return a torch.distributed.group.
- ``move_model_to_device(model, device)``: This api moves the model to device by
  preserving tensor parallel attributes.

Parallel Layers:
^^^^^^^^^^^^^^^^

Majority of parameters within the transformer based model reside in the
Embedding and Linear layers. Hence, to reduce the number of parameters
on a single device because of these layers, we provided sharded
Embedding and Linear layers.

Parallel Embedding:
'''''''''''''''''''

::

   class neuronx_distributed.parallel_layers.ParallelEmbedding(
       num_embeddings, embedding_dim, init_method=init.normal_,
       dtype=torch.float32, device=None)

This module is intended to replace torch.nn.Embedding . In cases where
the vocab size is too large, we can shard the Embedding table across
workers. Note: The embedding table would be sharded across all the
tensor-parallel workers.

.. _parameters-1:

Parameters:

-  ``num_embeddings (int)`` : size of the dictionary of embeddings
-  ``embedding_dim (int)`` : the size of each embedding vector
-  ``init_method: (torch.nn.init)`` : Initialization function for the
   embedding weights.
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU

ColumnParallel Linear Layer:
''''''''''''''''''''''''''''

::

   class neuronx_distributed.parallel_layers.ColumnParallelLinear(
       input_size, output_size, bias=True, gather_output=True,
       sequence_parallel_enabled=False, dtype=torch.float32, device=None)

This module would perform a Column wise partition of the weight matrix.
Linear layer is defined as ``Y = XA + b`` , here A is parallelized along
second dimension as ``A = [A_1, A_2 .... A_p]`` . ``Note``: This layer
is designed to operate on 3-dimensional inputs.

.. _parameters-2:

Parameters:

-  ``input_size: (int)`` : First dimension of the weight matrix
-  ``output_size: (int)`` : Second dimension of the weight matrix
-  ``bias: (bool)``: If set to True, bias would be added
-  ``gather_output: (bool)`` : If true, call all-gather on output and
   make Y available to all Neuron devices, otherwise, every Neuron
   device will have its output which is Y_i = XA_i
- ``sequence_parallel_enabled: (bool)`` : When sequence-parallel is enabled, it would
   gather the inputs from the sequence parallel region and perform the forward and backward
   passes
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU

RowParallel Linear Layer:
'''''''''''''''''''''''''

::

   class neuronx_distributed.parallel_layers.RowParallelLinear(
       input_size, output_size, bias=True, input_is_parallel=False,
       sequence_parallel_enabled=False, dtype=torch.float32, device=False
   )

The linear layer is defined as ``Y = XA + b``. A is parallelized along
its first dimension and X along its second. ``Note``: This layer is
designed to operate on 3-dimensional inputs.

.. _parameters-3:

Parameters:

-  ``input_size: (int)`` : First dimension of the weight matrix
-  ``output_size: (int)`` : Second dimension of the weight matrix
-  ``bias: (bool)`` : If set to True, bias would be added
-  ``input_is_parallel: (bool)`` : If true, we assume that the input is
   already split across the Neuron devices and we do not split again.
   This is useful when we have a ColumnParallel Layer just before the
   Row Parallel layer
-  ``sequence_parallel_enabled: (bool)`` : When sequence-parallel is enabled, it would
   gather the inputs from the sequence parallel region and perform the forward and backward
   passes
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU


Padding Tensor-Parallel Layers
''''''''''''''''''''''''''''''

::

   def neuronx_distributed.parallel_layers.pad.pad_model(
      model, tp_degree, n_heads, wrapped_classes=(), pad_hook_fn=None)


Pads a generic model to function to a desired tensor parallelism degree by padding the
number of attention heads. Returns the original model modified with padding.
Uses 1-axis padding strategy: pads the sharded dim of the ParallelLinear layers to the
size it would have been for the padded number of heads.

.. _parameters-4:

Parameters:

- ``model (torch.nn.Module)`` : model to be padded
- ``tp_degree (int)`` : tensor parallel degree
- ``n_heads (int)`` : the number of heads the given model to be padded has. This can
   typically be found in the config
- ``wrapped_classes (Tuple[any], *optional*, defaults to `()`)`` : tuple of classes
   (and their submodules) which should be padded
- ``pad_hook_fn (Callable[any, float], *optional*, defaults to `None`)`` : a hook
   function that is called whenever encountering a class to pad. Receives an instance
   of the class to pad and the tgt_src_ratio (num_heads_padded / num_heads)as its argument

Usage:
   When modifying the Attention layer, typically you must divide by TP degree like so:
   ::
      self.num_heads = neuronx_dist_utils.divide(self.num_heads, get_tensor_model_parallel_size())

   This line must be modified like so:
   ::
      self.num_heads = neuronx_dist_utils.divide(
         self.num_heads + get_number_of_extra_heads(self.num_heads, get_tensor_model_parallel_size()),
         get_tensor_model_parallel_size())

   Then, after initializing the model, you must call this wrapper:
   ::
      model = get_model(config=desired_config)
      model = pad_model(model, tp_degree=32, desired_config.num_heads)  # Use the model as desired after this point

   You can specify a specific layer or class for your model to pad, so you aren't unnecessarily padding.
   Typically, this layer will be your Attention layer
   ::
      model = pad_model(model, tp_degree=32, desired_config.num_heads, wrapped_classes=[MyAttention])

   You can also specify a pad_hook_fn, to be called whenever encountering an instance of wrapped_class,
   passing in said instance as a parameter, along with the tgt_src_ratio (num_heads_padded / num_heads).
   ::
      def my_hook(attention_to_pad, tgt_src_ratio):
         attention_to_pad.split_size = int(model.split_size * tgt_src_ratio)
         model = pad_model(
                  model,
                  tp_degree=32,
                  desired_config.num_heads,
                  wrapped_classes=[MyAttention],
                  pad_hook_fn=my_hook
               )


Loss functions:
''''''''''''''''''

When you shard the final MLP layer using tensor-parallelism, instead of
recollecting all the outputs from each TP rank, we can use the
ParallelCrossEntropy loss function. This function would take the parallel
logits produced by final parallel MLP and produce a loss by taking into
account that the logits are sharded across multiple workers.


::

   def neuronx_distributed.parallel_layers.loss_functions.parallel_cross_entropy(
       parallel_logits, labels, label_smoothing=0.0)

.. _parameters-6:

Parameters:


-  ``parallel_logits (Tensor)`` : Sharded logits from the previous MLP
-  ``labels (Tensor)`` : Label for each token. Labels should not be sharded,
   and the parallel_cross_entropy would take care of sharding the labels internally
-  ``label_smoothing (float)`` : A float in [0.0, 1.0]. Specifies the amount of
   smoothing when computing the loss, where 0.0 means no smoothing




Pipeline parallelism:
^^^^^^^^^^^^^^^^^^^^

Neuron Distributed Pipeline Model
'''''''''''''''''''''''''''''''''

::

   class NxDPPModel(module: torch.nn.Module,
        transformer_layer_cls: Optional[Any] = None,
        num_microbatches: int = 1,
        output_loss_value_spec: Optional[Union[Dict, Tuple]] = None,
        return_mb_loss: bool = False,
        broadcast_and_average_loss: bool = False,
        pipeline_cuts: Optional[List[str]] = None,
        input_names: Optional[List[str]] = None,
        leaf_module_cls: Optional[List[Any]] = None,
        autowrap_functions: Optional[Tuple[ModuleType]] = None,
        autowrap_modules: Optional[Tuple[Callable, ...]] = None,
        tracer_cls: Optional[Union[str, Any]] = None,
        param_init_fn: Optional[Any] = None,
        trace_file_path: Optional[str] = None,
        use_zero1_optimizer: bool = False,
   )

Parameters:

- ``module``: Module to be distributed with pipeline parallelism

- ``transformer_layer_cls``: The module class of transformer layers

- ``num_microbatches``: Number of pipeline microbatchs

- ``output_loss_value_spec``:
      The ``output_loss_value_spec`` value can be specified to disambiguate
      which value in the output of `forward` is the loss value on which NxDPPModel should apply
      backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
      you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
      a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
      ``output_loss_value_spec={'loss': True, 'model_out': False}``
      referred from `this <https://github.com/pytorch/PiPPy/blob/main/pippy/IR.py#L697>`__

- ``return_mb_loss``: Whether return a list of loss for all microbatchs

- ``broadcast_and_average_loss``:Whether to broadcast loss to all PP ranks and average across dp ranks, when set to True return_mb_loss must be False

- ``pipeline_cuts``: A list of layer names that will be used to annotate pipeline stage boundaries

- ``input_names``:The input names that will be used for tracing, which will be the same as the model inputs during runtime.

- ``leaf_module_cls``:A list of module classes that should be treated as leaf nodes during tracing. Note transformer layer class will be by default treat as leaf nodes.

- ``autowrap_modules``: (symbolic tracing only)
      Python modules whose functions should be wrapped automatically
      without needing to use fx.wrap().
      reference `here <https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L241>`__

- ``autowrap_functions``: (symbolic tracing only)
      Python functions that should be wrapped automatically without
      needing to use fx.wrap().
      reference `here <https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L241>`__

- ``tracer_cls``:User provided tracer class for symbolic tracing. It can be "hf", "torch" or any tracer class user created.

- ``param_init_fn``:
      Function used to initialize parameters. This is useful if user wants to use meta device to do
      delayed parameter initialization. param_init_fn should take a module as input and initialize the
      parameters that belongs to this module only (not for submodules).

- ``use_zero1_optimizer``: Whether to use the zero1 optimizer. When setting to True the gradient average will be handled over.

Common used APIs

::

   NxDPPModel.run_train(**kwargs)

Train the model with PP schedule, which will run both forward and backward in a PP manner.
The kwargs should be the same as the input_names provided to the trace function.
Will output the loss that provided by user from output_loss_value_spec.

::

   NxDPPModel.run_eval(**kwargs)

Eval the model with PP schedule, which will run forward only.
The kwargs should be the same as the input_names provided to the trace function.
Will output the loss that provided by user from output_loss_value_spec.

::

   NxDPPModel.local_named_parameters(**kwargs)

The parameters that are local to this PP rank. This must be called after the model is partitioned.

::

   NxDPPModel.local_named_modules(**kwargs)

The graph modules that are local to this PP rank. This must be called after the model is partitioned.


Checkpointing:
^^^^^^^^^^^^^^

These are set of APIs for saving and loading the checkpoint. These APIs
take care of saving and loading the shard depending the tensor parallel
rank of the worker.

Save Checkpoint:
''''''''''''''''

::

   def neuronx_distributed.parallel_layers.save(state_dict, save_dir, save_serially=True, save_xser: bool=False, down_cast_bf16=False)

.. note:: This method will be deprecated, use ``neuronx_distributed.trainer.save_checkpoint`` instead.

This API will save the model from each tensor-parallel rank in the
save_dir . Only workers with data parallel rank equal to 0 would be
saving the checkpoints. Each tensor parallel rank would be creating a
``tp_rank_ii_pp_rank_ii`` folder inside ``save_dir`` and each ones saves its shard
in the ``tp_rank_ii_pp_rank_ii`` folder.
If ``save_xser`` is enabled, the folder name would be ``tp_rank_ii_pp_rank_ii.tensors``
and there will be a ref data file named as ``tp_rank_ii_pp_rank_ii`` in save_dir for each rank.

.. _parameters-4:

Parameters:


-  ``state_dict: (dict)`` : Model state dict. Its the same dict that you
   would save using torch.save
-  ``save_dir: (str)`` : Model save directory.
-  ``save_serially: (bool)``: This flag would save checkpoints one model-parallel rank at a time.
   This is particularly useful when we are checkpointing large models.
-  ``save_xser: (bool)``: This flag would save the model with torch xla serialization.
   This could significantly reduce checkpoint saving time when checkpointing large model, so it's recommended
   to enable xser when the model is large.
   Note that if a checkpoint is saved with ``save_xser``, it needs to be loaded with ``load_xser``, vice versa.
-  ``down_cast_bf16: (bool)``: This flag would downcast the state_dict to bf16 before saving.

Load Checkpoint
'''''''''''''''

::

   def neuronx_distributed.parallel_layers.load(
       load_dir, model_or_optimizer=None, model_key='model', load_xser=False, sharded=True)

.. note:: This method will be deprecated, use ``neuronx_distributed.trainer.load_checkpoint`` instead.

This API will automatically load checkpoint depending on the tensor
parallel rank. For large models, one should pass the model object to the
load API to load the weights directly into the model. This could avoid
host OOM, as the load API would load the checkpoints for one tensor
parallel rank at a time.

.. _parameters-5:

Parameters:


-  ``load_dir: (str)`` : Directory where the checkpoint is saved.
-  ``model_or_optimizer``: (torch.nn.Module or torch.optim.Optimizer): Model or Optimizer object.
-  ``model``: (torch.nn.Module or torch.optim.Optimizer): Model or Optimizer object, equivilant to ``model_or_optimizer``
-  ``model_key: (str)`` : The model key used when saving the model in the
   state_dict.
-  ``load_xser: (bool)`` : Load model with torch xla serialization.
   Note that if a checkpoint is saved with ``save_xser``, it needs to be loaded with ``load_xser``, vice versa.
-  ``sharded: (bool)`` : If the checkpoint is not sharded, pass False.
   This is useful (especially during inference) when the model is
   trained using a different strategy and you end up saving a single
   unsharded checkpoint. You can then load this unsharded checkpoint
   onto the sharded model. When this attribute is set to ``False`` , it
   is necessary to pass the model object. Note: The keys in the
   state-dict should have the same name as in the model object, else it
   would raise an error.

Gradient Clipping:
''''''''''''''''''

With tensor parallelism, we need to handle the gradient clipping as we
have to accumulate the total norm from all the tensor parallel ranks.
This should be handled by the following API

::

   def neuronx_distributed.parallel_layers.clip_grad_norm(
       parameters, max_norm, norm_type=2)

.. _parameters-6:

Parameters:


-  ``parameters (Iterable[Tensor] or Tensor)`` : an iterable of Tensors
   or a single Tensor that will have gradients normalized
-  ``max_norm (float or int)`` :max norm of the gradients
-  ``norm_type (float or int)`` : type of the used p-norm. Can be ‘inf’
   for infinity norm.

Neuron Zero1 Optimizer:
'''''''''''''''''''''''

In Neuronx-Distributed, we built a wrapper on the Zero1-Optimizer present in torch-xla.

::

   class NeuronZero1Optimizer(Zero1Optimizer)

This wrapper takes into account the tensor-parallel degree and computes the grad-norm
accordingly. It also provides two APIs: save_sharded_state_dict and load_sharded_state_dict.
As the size of the model grows, saving the optimizer state from a single rank can result in OOMs.
Hence, the api to save_sharded_state_dict can allow saving states from each data-parallel rank. To
load this sharded optimizer state, there is a corresponding load_sharded_state_dict that allows each
rank to pick its corresponding shard from the checkpoint directory.

::

   optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
   ]

   optimizer = NeuronZero1Optimizer(
        optimizer_grouped_parameters,
        AdamW,
        lr=flags.lr,
        pin_layout=False,
        sharding_groups=parallel_state.get_data_parallel_group(as_list=True),
        grad_norm_groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
    )

The interface is same as Zero1Optimizer in torch-xla

::

   save_sharded_state_dict(output_dir, save_serially = True)

.. note:: This method will be deprecated, use ``neuronx_distributed.trainer.save_checkpoint`` instead.

.. _parameters-7:

Parameters:


-  ``output_dir (str)`` : Checkpoint directory where the sharded optimizer states need to be saved
-  ``save_serially (bool)`` : Whether to save the states one data-parallel rank at a time. This is
    especially useful when we want to checkpoint large models.

::

   load_sharded_state_dict(output_dir, num_workers_per_step = 8)

.. note:: This method will be deprecated, use ``neuronx_distributed.trainer.load_checkpoint`` instead.

.. _parameters-8:

Parameters:


-  ``output_dir (str)`` : Checkpoint directory where the sharded optimizer states are saved
-  ``num_workers_per_step (int)`` : This argument controls how many workers are doing model load
   in parallel.

Neuronx-Distributed Training APIs:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Neuronx-Distributed, we provide a series of APIs under `neuronx_distributed` directly that helps
user to apply optimizations in NxD easily. These APIs cover configuration, model/optimizer initialization
and saving/loading checkpoint.


Initialize NxD config:
''''''''''''''''''''''

::

   def neuronx_distributed.trainer.neuronx_distributed_config(
       tensor_parallel_size=1,
       pipeline_parallel_size=1,
       pipeline_config=None,
       optimizer_config=None,
       activation_checkpoint_config=None,
       pad_model=False,
       sequence_parallel=False,
       model_init_config=None,
   )

This method initialize NxD training config and initialize model parallel. This config
maintains all optimization options of the distributed training, and it's a global config
(the same for all processes).

Parameters:

- ``tensor_parallel_size (int)`` : Tensor model parallel size. Default: :code:`1`.
- ``pipeline_parallel_size (int)`` : Pipeline model parallel size. Default: :code:`1`.
- ``pipeline_config (dict)`` : Pipeline parallel config. For details please refer to
  `pipeline parallel guidance <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/pp_developer_guide.html>`__.
  Default: :code:`None`.
- ``optimizer_config (dict)`` : Optimizer config.
  Default: :code:`{"zero_one_enabled": False, "grad_clipping": True, "max_grad_norm": 1.0}`.

  - Enable ZeRO-1 by setting ``zero_one_enabled`` to ``True``.
  - Enable grad clipping by setting ``grad_clipping`` to ``True``.
  - Change maximum grad norm value by setting ``max_grad_norm``.

- ``activation_checkpoint_config (str of torch.nn.Module)`` : Activation checkpoint config,
  accept value: ``"full"``, ``None``, or any ``torch.nn.Module``. When set to ``full``,
  regular activation checkpoint enabled (every transformer layer will be re-computed).
  When set to ``None``, activation checkpoint disabled. When set to any ``torch.nn.Module``,
  selective activation checkpoint enabled, any provided module will be re-computed.
  Default: :code:`None`.
- ``pad_model (bool)`` : Whether to pad attention heads of model. Default: :code:`False`.
- ``sequence_parallel (bool)`` : Whether to enable sequence parallel. Default: :code:`False`.
- ``model_init_config (dict)`` : Model initialization config.
  Default: :code:`{"sequential_move_factor": 11, "meta_device_init": False, "param_init_fn": None}`.

  - ``sequential_move_factor``: num of processes instantiating model on host at the same time.
    This is done to avoid the host OOM. Range: 1-32.
  - ``meta_device_init``: whether to initialize model on meta device.
  - ``param_init_fn``: method that initialize parameters of modules, should be provided when
    ``param_init_fn`` is ``True``.

Initialize NxD Model Wrapper:
'''''''''''''''''''''''''''''

::

   def neuronx_distributed.trainer.initialize_parallel_model(nxd_config, model_fn, *model_args, **model_kwargs)

This method initialize NxD model wrapper, return a wrapped model that can be used as
a regular ``torch.nn.Module``, while has all the model optimizations in config applied.
This wrapper is designed to hide the complexity of optimizations such as pipeline model
parallel, so that users can simply use the wrapped model as the unwrapped version.

Parameters:

- ``nxd_config (dict)``: config generated by ``neuronx_distributed_config``.
- ``model_fn (callable)``: user provided function to get the model for training.
- ``model_args`` and ``model_kwargs``: arguments that will be passed to ``model_fn``.

Model wrapper class and its methods:

::

   class neuronx_distributed.trainer.model.NxDModel(torch.nn.Module):
       def local_module(self):
           # return the unwrapped local module

       def run_train(self, *args, **kwargs):
           # method to run one iteration, when pipeline parallel enabled,
           # user have to use this instead of forward+backward

       def named_parameters(self, *args, **kwargs):
           # only return parameters on local rank.
           # same for `parameters`, `named_buffers`, `buffers`

       def named_modules(self, *args, **kwargs):
           # only return modules on local rank.
           # same for `modules`, `named_children`, `children`

P.S.: as a short cut, users can call ``model.config`` or ``model.dtype`` from wrapped model
if original model is hugging face transformers pre-trained model.

Initialize NxD Optimizer Wrapper:
'''''''''''''''''''''''''''''''''

::

   def neuronx_distributed.trainer.initialize_parallel_optimizer(nxd_config, optimizer_class, parameters, **defaults)

This method initialize NxD optimizer wrapper, return a wrapped optimizer that can be used as
a regular ``torch.optim.Optimizer``, while has all the optimizer optimizations in config applied.

This optimizer wrapper is inherited from ``toch.optim.Optimizer``. It takes in the ``nxd_config`` and
configures the optimizer to work with different distributed training regime.

The `step` method of the wrapped optimizer contains necessary all-reduce operations and grad clipping.
Other methods and variables work the same as the unwrapped optimizer.

Parameters:

- ``nxd_config (dict)``: config generated by ``neuronx_distributed_config``.
- ``optimizer_class (Type[torch.optim.Optimizer])``: optimizer class to create the optimizer.
- ``parameters (iterable)``: parameters passed to the optimizer.
- ``defaults``: optimizer options that will be passed to the optimizer.

Save Checkpoint:
''''''''''''''''

Method to save checkpoint, return ``None``.

This method saves checkpoints for model, optimizer, scheduler and user contents sequentially.
Model states are saved on data parallel rank-0 only. When ZeRO-1 optimizer is not turned on,
optimizer states are also saved like this; while when ZeRO-1 optimizer is turned on, states
are saved on all ranks. Scheduler and user contents are saved on master rank only. Besides,
users can use ``use_xser=True`` to boost saving performance and avoid host OOM. It's achieved
by saving tensors one by one simultaneously and keeping the original data structure.

::

   def neuronx_distributed.trainer.save_checkpoint(
       path,
       tag="",
       model=None,
       optimizer=None,
       scheduler=None,
       user_content=None,
       num_workers=8,
       use_xser=False,
       num_kept_ckpts=None,
   )

Parameters:

- ``path (str)``: path to save the checkpoints.
- ``tag (str)``: tag to save the checkpoints.
- ``model (torch.nn.Module)``: model to save, optional.
- ``optimizer (torch.optim.Optimizer)``: optimizer to save, optional.
- ``scheduler``: scheduler to save, optional.
- ``user_content``: user contents to save, optional.
- ``num_workers (int)``: num of processes saving data on host at the same time.
  This is done to avoid the host OOM, range: 1-32.
- ``use_xser (bool)``: whether to use torch-xla serialization. When enabled, ``num_workers``
  will be ignored and maximum num of workers will be used. Default: :code:`False`.
- ``num_kept_ckpts (int)``: number of checkpoints to keep on disk, optional. Default: :code:`None`.

Load Checkpoint:
''''''''''''''''

Method to load checkpoint saved by ``save_checkpoint``, return user contents if exists otherwise ``None``.
If ``tag`` not provided, will try to use the newest tag tracked by ``save_checkpoint``.

Note that the checkpoint to be loaded must have the same model parallel degrees as in current use,
and if ZeRO-1 optimizer is used, must use the same data parallel degrees.

::

   def neuronx_distributed.trainer.load_checkpoint(
       path,
       tag=None,
       model=None,
       optimizer=None,
       scheduler=None,
       num_workers=8,
       strict=True,
   )

Parameters:

- ``path (str)``: path to load the checkpoints.
- ``tag (str)``: tag to load the checkpoints.
- ``model (torch.nn.Module)``: model to load, optional.
- ``optimizer (torch.optim.Optimizer)``: optimizer to load, optional.
- ``scheduler``: scheduler to load, optional.
- ``num_workers (int)``: num of processes loading data on host at the same time.
  This is done to avoid the host OOM, range: 1-32.
- ``strict (bool)``: whether to use strict mode when loading model checkpoint. Default: :code:`True`.

**Sample usage:**

::

   import neuronx_distributed as nxd

   # create config
   nxd_config = nxd.neuronx_distributed_config(
       tensor_parallel_size=8,
       optimizer_config={"zero_one_enabled": True, "grad_clipping": True, "max_grad_norm": 1.0},
   )

   # wrap model
   model = nxd.initialize_parallel_model(nxd_config, get_model)

   # wrap optimizer
   optimizer = nxd.initialize_parallel_optimizer(nxd_config, AdamW, model.parameters(), lr=1e-3)

   ...
   (training loop):
      loss = model.run_train(inputs)
      optimizer.step()

   ...
   # loading checkpoint (auto-resume)
   user_content = nxd.load_checkpoint(
       "ckpts",
       model=model,
       optimizer=optimizer,
       scheduler=scheduler,
   )
   ...
   # saving checkpoint
   nxd.save_checkpoint(
       "ckpts",
       nxd_config=nxd_config,
       model=model,
       optimizer=optimizer,
       scheduler=scheduler,
       user_content={"total_steps": total_steps},
   )

Modules:
^^^^^^^^

GQA-QKV Linear Module:
''''''''''''''''''''''

::

   class neuronx_distributed.modules.qkv_linear.GQAQKVColumnParallelLinear(
       input_size, output_size, bias=True, gather_output=True,
       sequence_parallel_enabled=False, dtype=torch.float32, device=None, kv_size_multiplier=1)

This module parallelizes the Q,K,V linear projections using ColumnParallelLinear layers. Instead of using 
3 different linear layers, we can replace it with a single QKV module. In case of GQA module, the number of 
Q attention heads are `N` times more than the number of K and V attention heads. The K and V attention heads 
are replicated after projection to match the number of Q attention heads. This helps to reduce the K and V 
weights and is useful especially during inference. However, in case of training these modules, it restricts 
the tensor-parallel degree that can be used, since the attention heads should be divisible by tensor-parallel 
degree. Hence, to mitigate this bottleneck, the `GQAQKVColumnParallelLinear` takes in a `kv_size_multiplier` 
argument. The module would replicate the K and V weights `kv_size_multiplier` times thereby allowing you to 
use higher tensor-parallel degree. Note: here instead of replicating the projection `N/tp_degree` times, we 
end of replicating the weights `kv_size_multiplier` times. This would produce the same result, allow you to use 
higher tp_degree degree, however, it would result in extra memory getting consumed.

.. _parameters-11:

Parameters:
           

-  ``input_size: (int)`` : First dimension of the weight matrix
-  ``output_sizes: (List[int])`` : A list of second dimension of the Q and K/V weight matrix
-  ``bias: (bool)``: If set to True, bias would be added
-  ``gather_output: (bool)`` : If true, call all-gather on output and
   make Y available to all Neuron devices, otherwise, every Neuron
   device will have its output which is Y_i = XA_i
- ``sequence_parallel_enabled: (bool)`` : When sequence-parallel is enabled, it would
   gather the inputs from the sequence parallel region and perform the forward and backward
   passes
-  ``init_method: (torch.nn.init)`` : Initialization function for the
   Q and K/V weights.
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU
- ``kv_size_multiplier: (int)``: Factor by which the K and V weights would be replicated along the first dimension

.. _nxd_tracing:

Model Trace:
^^^^^^^^^^^^

We can use the tensor parallel layers to perform large model inference
too. For performing inference, we can re-use the Parallel model built
above for training and then use the trace APIs provided by the
neuronx_distributed package to trace it for inference. One can use the
following set of APIs for running distributed inference:

::

   def neuronx_distributed.trace.parallel_model_trace(func, inputs, tp_degree=1)

This API would launch tensor parallel workers, where each worker would
trace its own model. These traced models would be wrapped with a single
TensorParallelModel module which can then be used like any other traced
model.

.. _parameters-9:

Parameters:


-  ``func : (Function)``: This is a function that returns a ``Model``
   object and a dictionary of states. The ``parallel_model_trace`` API would call this function
   inside each worker and run trace against them. Note: This differs
   from the ``torch_neuronx.trace`` where the ``torch_neuronx.trace``
   requires a model object to be passed.
-  ``inputs: (torch tensors)`` : The inputs that needs to be passed to
   the model.
-  ``tp_degree: (int)`` : How many devices to be used when performing
   tensor parallel sharding

Trace Model Save/Load:
^^^^^^^^^^^^^^^^^^^^^^

Save:
'''''

::

   def neuronx_distributed.trace.parallel_model_save(model, save_dir)

This API should save the traced model in save_dir . Each shard would be
saved in its respective directory inside the save_dir. Parameters:

-  ``model: (TensorParallelModel)`` : Traced model produced using the
   parallel_model_trace api.
-  ``save_dir: (str)`` : The directory where the model would be saved

Load:
'''''

::

   def neuronx_distributed.trace.parallel_model_load(load_dir)

This API will load the sharded traced model into ``TensorParallelModel``
for inference.

.. _parameters-10:

Parameters:
'''''''''''

-  ``load_dir: (str)`` : Directory which contains the traced model.


Neuron PyTorch-Lightning
^^^^^^^^^^^^^^^^^^^^^^^^
Neuron PyTorch-Lightning is currently based on Lightning version 2.1.0, and will eventually be upstreamed Lightning-AI code base

Neuron Lightning Module
'''''''''''''''''''''''

Inherited from `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`__
::

   class neuronx_distributed.lightning.NeuronLTModule(
      model_fn: Callable,
      nxd_config: Dict,
      opt_cls: Callable,
      scheduler_cls: Callable,
      model_args: Tuple = (),
      model_kwargs: Dict = {},
      opt_args: Tuple = (),
      opt_kwargs: Dict = {},
      scheduler_args: Tuple = (),
      scheduler_kwargs: Dict = {},
      grad_accum_steps: int = 1,
      manual_opt: bool = True,
   )

Parameters:

- ``model_fn``: Model function to create the actual model

- ``nxd_config``: Neuronx Distributed Config, output of neuronx_distributed.neuronx_distributed_config

- ``opt_cls``: Callable to create optimizer

- ``scheduler_cls``: Callable to create scheduler

- ``model_args``: Tuple of args fed to model callable

- ``model_kwargs``: Dict of keyworded args fed to model callable

- ``opt_args``: Tuple of args fed to optimizer callable

- ``opt_kwargs``: Dict of keyword args fed to optimizer callable

- ``scheduler_args``: Tuple of args fed to scheduler callable

- ``scheduler_args``: Dict of keyworded args fed to scheduler callable

- ``grad_accum_steps``: Grad accumulation steps

- ``manual_opt``: Whether to do manual optimization, note that currently NeuronLTModule doesn't support auto optimization so this should always set to True


Neuron XLA Strategy
'''''''''''''''''''

Inherited from `XLAStrategy <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.XLAStrategy.html>`__
::

   class neuronx_distributed.lightning.NeuronXLAStrategy(
      nxd_config: Dict = None,
      tensor_parallel_size: int = 1,
      pipeline_parallel_size: int = 1,
      save_load_xser: bool = True,
   )

Parameters:

- ``nxd_config``: Neuronx Distributed Config, output of neuronx_distributed.neuronx_distributed_config

- ``tensor_parallel_size``: Tensor parallel degree, only needed when nxd_config is not specified

- ``pipeline_parallel_size``: Pipeline parallel degree, only needed when nxd_config is not specified (Note that for now we only support TP with Neuron-PT-Lightning)

- ``save_load_xser``: Set to True will enable save/load with xla serialization, for more context check `Save Checkpoint <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#save-checkpoint>`__


Neuron XLA Precision Plugin
'''''''''''''''''''''''''''

Inherited from `XLAPrecisionPlugin <https://github.com/Lightning-AI/lightning/blob/2.1.0/src/lightning/pytorch/plugins/precision/xla.py>`__

::

   class neuronx_distributed.lightning.NeuronXLAPrecisionPlugin

Neuron TQDM Progress Bar
''''''''''''''''''''''''

Inherited from `TQDMProgressBar <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.TQDMProgressBar.html>`__

::

   class neuronx_distributed.lightning.NeuronTQDMProgressBar


Neuron TensorBoard Logger
'''''''''''''''''''''''''

Inherited from `TensorBoardLogger <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html>`__

::

   class neuronx_distributed.lightning.NeuronTensorBoardLogger(save_dir)

Parameters:

- ``save_dir``: Directory to save the log files