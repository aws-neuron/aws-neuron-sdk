.. _tp_api_guide:

API Reference Guide for Tensor Parallelism (``neuronx-distributed`` )
======================================================================

To support tensor-parallelism on Neuron, we adopted the Apex Library
built for CUDA devices. We modified the implementations to work with
XLA. Here are the tensor-parallel APIs that can be used to enable tensor
parallelism:

Parallel Model State:
^^^^^^^^^^^^^^^^^^^^^

Initialize Model Parallelism:
'''''''''''''''''''''''''''''

::

   def neuronx_distributed.parallel_state.initialize_model_parallel(
           tensor_model_parallel_size=1)

This module would initialize the distributed model training and allows
users to set the number of tensor_parallel world size.

Parameters:
           

``tensor_model_parallel_size`` : This should set the number of tensor
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
       dtype=torch.float32, device=None)

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
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU

RowParallel Linear Layer:
'''''''''''''''''''''''''

::

   class neuronx_distributed.parallel_layers.RowParallelLinear(
       input_size, output_size, bias=True, input_is_parallel=False,
       dtype=torch.float32, device=False
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
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU

Checkpointing:
^^^^^^^^^^^^^^

These are set of APIs for saving and loading the checkpoint. These APIs
take care of saving and loading the shard depending the tensor parallel
rank of the worker.

Save Checkpoint:
''''''''''''''''

::

   def neuronx_distributed.parallel_layers.save(state_dict, save_dir)

This API will save the model from each tensor-parallel rank in the
save_dir . Only workers with data parallel rank equal to 0 would be
saving the checkpoints. Each tensor parallel rank would be creating a
``tp_rank_i`` folder inside ``save_dir`` and each ones saves its shard
in the ``tp_rank_i`` folder.

.. _parameters-4:

Parameters:
           

-  ``state_dict: (dict)`` : Model state dict. Its the same dict that you
   would save using torch.save
-  ``save_dir: (str)`` : Model save directory.

Load Checkpoint
'''''''''''''''

::

   def neuronx_distributed.parallel_layers.load(
       load_dir, model=None, model_key='model', sharded=True)

This API will automatically load checkpoint depending on the tensor
parallel rank. For large models, one should pass the model object to the
load API to load the weights directly into the model. This could avoid
host OOM, as the load API would load the checkpoints for one tensor
parallel rank at a time.

.. _parameters-5:

Parameters:
           

-  ``load_dir: (str)`` : Directory where the checkpoint is saved.
-  ``model``: (torch.nn.Module): Model object
-  ``model_key: (str)`` :The model key used when saving the model in the
   state_dict.
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

.. _parameters-7:

Parameters:
           

-  ``func : (Function)``: This is a function that returns a ``Model``
   object. The ``parallel_model_trace`` API would call this function
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

.. _parameters-8:

Parameters:
'''''''''''

-  ``load_dir: (str)`` : Directory which contains the traced model.