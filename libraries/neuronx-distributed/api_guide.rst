.. _api_guide:

Distributed Strategies APIs
===========================


NeuronX Distributed Core (NxD Core) is XLA based library for distributed training and inference on Neuron devices.
As part of this library, we support 3D parallelism: Tensor-Parallelism, Pipeline-Parallelism
and Data-Parallelism. We also support Zero1 optimizer to shard the optimizer weights.
To support tensor-parallelism on Neuron, we adopted the Apex Library
built for CUDA devices. We modified the implementations to work with
XLA. This document enlist the different APIs and modules provided by the library

.. contents:: Table of contents
   :local:
   :depth: 2


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

   class NxDPPModel(
        module: torch.nn.Module,
        transformer_layer_cls: Optional[Any] = None,
        num_microbatches: int = 1,
        virtual_pipeline_size: int = 1,
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
        auto_partition: Optional[bool] = False,
        deallocate_pipeline_outputs: bool = False,
   )

Parameters:

- ``module``: Module to be distributed with pipeline parallelism

- ``transformer_layer_cls``: The module class of transformer layers

- ``num_microbatches``: Number of pipeline microbatchs

- ``virtual_pipeline_size``: Virtual pipeline size if greater than 1 we will use the interleaved pipeline schedule.

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

- ``use_zero1_optimizer``: Whether to use the zero1 optimizer. When setting to True the gradient average will be handed over.

- ``auto_partition``:
      Boolean to indicate whether to use auto_partition for the model. When set to True, the pipeline
      cuts used as the pipeline stage boundaries to partition the model are automatically determined. When set to
      True, the pipeline_cuts parameter should not be set. The pipeline_cuts are chosen on the basis of the transformer layer names.

- ``deallocate_pipeline_outputs``: 
      Whether to deallocate the pipeline outputs after send. After send the output tensor is only useful for its 
      '.grad_fn' field, and not its '.data'.

Common used APIs
'''''''''''''''''

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

