.. _nxd-core-model-builder-v2:

ModelBuilderV2 API Reference
==============================================

APIs
~~~~

- `neuronx_distributed.trace.model_builder.trace`_
- `neuronx_distributed.trace.model_builder.compile`_
- `neuronx_distributed.shard_checkpoint`_
- `neuronx_distributed.ModelBuilder`_
- `neuronx_distributed.ModelBuilder.trace`_
- `neuronx_distributed.ModelBuilder.compile`_
- `neuronx_distributed.trace.nxd_model.base_nxd_model.StateInitializer`_
- `neuronx_distributed.NxDModel`_
- `neuronx_distributed.NxDModel.add`_
- `neuronx_distributed.NxDModel.get_neff`_
- `neuronx_distributed.NxDModel.get_metaneff`_
- `neuronx_distributed.NxDModel.get_hlo`_
- `neuronx_distributed.NxDModel.set_weights`_
- `neuronx_distributed.NxDModel.to_neuron`_
- `neuronx_distributed.NxDModel.replace_weights`_
- `neuronx_distributed.NxDModel.read_from_neuron_buffer`_
- `neuronx_distributed.NxDModel.write_to_neuron_buffer`_
- `neuronx_distributed.NxDModel.forward`_
- `neuronx_distributed.NxDModel.save`_
- `neuronx_distributed.NxDModel.load`_

`Usage Notes`_

**Examples**

`Usage Examples`_

- `E2E with ModelBuilder APIs`_
- `E2E with Fundamental Units`_

neuronx_distributed.trace.model_builder.trace
=============================================

::

   neuronx_distributed.trace.model_builder.trace(
       model: Union[Callable, torch.nn.Module],
       args: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]] = None,
       kwargs: Optional[Dict[str, torch.Tensor]] = None,
       spmd: bool = True,
       preserve_parameters: bool = True,
   ) -> TraceArtifacts

The ``trace()`` function is a fundamental unit in the ModelBuilderV2
framework that handles the tracing of PyTorch models for execution on
Neuron devices. It processes example inputs as both positional and
keyword arguments, validates model parameters, and generates necessary
trace artifacts such as HLOs.

Parameters
~~~~~~~~~~

- **model: Union[Callable, torch.nn.Module]** — The PyTorch model or
  callable function to be traced. Must have explicitly defined
  parameters (no ``*args`` *or* ``**kwargs``). Must have at least one
  parameter.
- **args: Union[None, torch.Tensor, Tuple[torch.Tensor, …]] = None** —
  Example inputs as positional arguments. Can be None, a single tensor,
  or a tuple of tensors. Must match the model’s positional parameter
  requirements.
- **kwargs: Optional[Dict[str, torch.Tensor]] = None** — Example inputs
  as keyword arguments. Must be a dictionary mapping parameter names to
  tensor values. Cannot override parameters provided in args.
- **spmd: bool = True** — Whether to use SPMD (Single Program Multiple
  Data) for tracing. Currently only True is supported
- **preserve_parameters: bool = True** — Whether to preserve module
  buffers across multi-bucket trace.

Returns
~~~~~~~

Returns a ``TraceArtifacts`` object containing:

::

   neuronx_distributed.trace.model_builder_utils.TraceArtifacts(
       hlo: Any,                                 # HLO representation
       metaneff: Any,                            # Meta information for NEFF
       flattener: Any,                           # Function to flatten inputs
       packer: Any,                              # Function to pack outputs
       weight_name_to_idx: Dict[str, int],       # Maps weight names to indices
       weight_names_to_skip: Set,                # Weight names excluded from optimization
       provided_args: List[ProvidedArgInfo],     # Information about provided arguments
       model_params: List[ModelParamInfo],       # Information about model parameters
   )

``ProvidedArgInfo`` object contains:

::

   neuronx_distributed.trace.model_builder_utils.ProvidedArgInfo(
        param_name: str,       # Name of the parameter this argument corresponds to
        is_positional: bool,   # Whether this argument is positional (required) or keyword (optional)
        tensor: torch.Tensor,  # The tensor value provided for this argument
   )

``ModelParamInfo`` object contains:

::

   neuronx_distributed.trace.model_builder_utils.ModelParamInfo(
        param_name: str,      # Name of the parameter in the function signature
        is_positional: bool,  # Whether this parameter is positional (required) or keyword (optional)
   )

neuronx_distributed.trace.model_builder.compile
===============================================

::

   neuronx_distributed.trace.model_builder.compile(
       hlo_module: hlo_pb2.HloModuleProto,
       metaneff: Any,
       compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
       compiler_args: Optional[str] = None,
       key: Optional[str] = None
   ) -> CompilationArtifacts

The ``compile()`` function is a fundamental unit in the ModelBuilderV2
framework that compiles traced models using the Neuron Compiler, and
generates Neuron Executable File Format (NEFF) files. It handles
compiler configurations, workdir management, and produces compilation
artifacts.

.. _parameters-1:

Parameters
~~~~~~~~~~

- **hlo_module: hlo_pb2.HloModuleProto** — The HLO module representing
  the computational graph to be compiled. Generated from the ``trace()``
  function.
- **metaneff: Any** — Meta information for the Neuron Executable File
  Format (NEFF)
- **compiler_workdir: Optional[Union[str, pathlib.Path]] = None** —
  Directory path to store compiler artifacts. If None, uses a default
  path. Creates timestamped subdirectories (in UTC format) for each
  compilation.
- **compiler_args: Optional[str] = None** — Compiler flags for
  neuronx-cc. If None, uses default compiler
  flags. Can include optimization levels and other compiler options.
- **key: Optional[str] = None** — Key to tag the bucket with a
  meaningful name. If None, generates a hash from the HLO module. Used
  for logging and artifact organization

.. _returns-1:

Returns
~~~~~~~

Returns a ``CompilationArtifacts`` object containing:

::

   neuronx_distributed.trace.model_builder_utils.CompilationArtifacts(
       neff_filepath: str    # Path to the compiled NEFF file
   )

Default Compiler Flags
~~~~~~~~~~~~~~~~~~~~~~

If no ``compiler_args`` are provided, the following defaults are used:

::

   --enable-saturate-infinity --auto-cast=none --model-type=transformer -O1

Directory Structure
~~~~~~~~~~~~~~~~~~~

This creates the following directory structure:

::

   compiler_workdir/
   └── {key}/
       └── {timestamp}/
           ├── model/
           │   └── graph.hlo
           ├── graph.neff
           ├── metaneff.pb
           └── command.txt
           └── log-neuron-cc.txt

neuronx_distributed.shard_checkpoint
====================================

::

   neuronx_distributed.shard_checkpoint(
       checkpoint: Dict[str, torch.Tensor],
       model: torch.nn.Module,
       start_rank: Optional[int] = None,
       end_rank: Optional[int] = None,
       load_on_device: bool = False,
       serialize_path: Optional[str] = None
   ) -> List[Dict[str, torch.Tensor]]

The ``shard_checkpoint()`` function shards a model checkpoint across
tensor parallel ranks for distributed execution. It supports options for
serialization (pre-shard) and direct loading onto Neuron devices
(shard-on-load).

.. _parameters-2:

Parameters
~~~~~~~~~~

- **checkpoint: Dict[str, torch.Tensor]** — The model checkpoint
  dictionary. Maps parameter names to tensor values. Must contain all
  model parameters.
- **model: torch.nn.Module** — The PyTorch model to be sharded. Used for
  determining sharding strategy.
- **start_rank: Optional[int] = None** — Starting rank for sharding.
  Must be in range [0, tp_degree). Defaults to 0 if None.
- **end_rank: Optional[int] = None** — Ending rank for sharding. Must be
  in range [start_rank, tp_degree). Defaults to ``(tp_degree - 1)`` if
  None.
- **load_on_device: bool = False** — Whether to load sharded tensors
  onto Neuron devices. Requires running on supported Neuron instance.
  Defaults to False.
- **serialize_path: Optional[str] = None** — Path to save sharded
  checkpoints. If provided, saves as safetensors files. Creates
  directory if it doesn’t exist.

.. _returns-2:

Returns
~~~~~~~

Returns a ``List[Dict[str, torch.Tensor]]`` where:

- Each dictionary represents a sharded checkpoint for a rank
- Dictionary keys are parameter names
- Dictionary values are sharded tensor values
- List length is (end_rank - start_rank + 1)

neuronx_distributed.ModelBuilder
================================

::

   class ModelBuilderV2:
       def __init__(
           self,
           model: Union[Callable, torch.nn.Module],
       )

ModelBuilderV2 is a high-level class that provides a fluent interface
for tracing and compiling PyTorch models for Neuron devices. It supports
SPMD (Single Program Multiple Data) execution, and distributed model
execution.

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~

- **model: Union[Callable, torch.nn.Module]** — The PyTorch model to be
  traced and compiled. Can be a model class or callable function. Must
  have explicitly defined parameters (no ``*args`` *or* ``**kwargs``).
  Must have at least one argument.

neuronx_distributed.ModelBuilder.trace
======================================

::

   neuronx_distributed.ModelBuilder.trace(
       self,
       args: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]] = None,
       kwargs: Optional[Dict[str, torch.Tensor]] = None,
       tag: Optional[str] = None,
       spmd: bool = True,
   ) -> ModelBuilderV2

Traces the model with given inputs and stores trace artifacts. Leverages
`neuronx_distributed.trace.model_builder.trace`_
fundamental unit.

.. _parameters-3:

Parameters
~~~~~~~~~~

- **args: Union[None, torch.Tensor, Tuple[torch.Tensor, …]] = None** —
  Example inputs as positional arguments. Can be None, a single tensor,
  or a tuple of tensors. Must match the model’s positional parameter
  requirements.
- **kwargs: Optional[Dict[str, torch.Tensor]] = None** — Example inputs
  as keyword arguments
- **tag: Optional[str] = None** — Unique identifier for this trace.
  Corresponding bucket will be tagged with this name. If None, generates
  a hash from the HLO module.
- **spmd: bool = True** — Whether to use SPMD (Single Program Multiple
  Data) for tracing. Currently only True is supported

.. _returns-3:

Returns
~~~~~~~

Self reference for method chaining.

neuronx_distributed.ModelBuilder.compile
========================================

::

   neuronx_distributed.ModelBuilder.compile(
       self,
       priority_model_key: Optional[str] = None,
       compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
       compiler_args: Optional[Union[str, Dict[str, str]]] = None,
       max_workers: Optional[int] = None,
   ) -> NxDModel

Compiles traced models using the Neuron compiler. Leverages
`neuronx_distributed.trace.model_builder.compile`_
fundamental unit.

.. _parameters-4:

Parameters
~~~~~~~~~~

- **priority_model_key: Optional[str] = None** — Key of model to
  prioritize for WLO
- **compiler_workdir: Optional[Union[str, pathlib.Path]] = None** —
  Directory for compiler artifacts
- **compiler_args: Optional[Union[str, Dict[str, str]]] = None** —
  Compiler flags as string or dictionary mapping tags to flags.
- **max_workers: Optional[int] = None** — Maximum worker threads for
  parallel compilation. If None, uses the default value from
  ThreadPoolExecutor.

.. _returns-4:

Returns
~~~~~~~

A built and configured ``NxDModel`` instance.

neuronx_distributed.trace.nxd_model.base_nxd_model.StateInitializer
===================================================================

::

   class StateInitializer(torch.nn.Module):
       def __init__(
           self,
           shapes: Dict[str, List[int]],
           dtypes: Dict[str, torch.dtype],
           local_ranks_size: int
       ):

A TorchScript-compatible module to initialize state buffers onto Neuron.

.. _constructor-parameters-1:

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~

- **shapes: Dict[str, List[int]]** — Dict of shape lists associated with
  a specific stateful tensor by key
- **dtypes: Dict[str, torch.dtype]** — Dict of torch dtypes associated
  with a specific stateful tensor by key
- **local_ranks_size: int** — integer representing the number of ranks
  per instance in a distributed setting. Unless it’s a Multi Instance
  Data Parallel setup, it is usually just equal to the ``world_size``
  your model was compiled for.

neuronx_distributed.NxDModel
============================

::

   class NxDModel(torch.nn.Module, BaseNxDModel):
       def __init__(
           self,
           world_size: int,
           start_rank: Optional[int] = None,
           local_ranks_size: Optional[int] = None,
           state_initializer: Optional[StateInitializer] = None,
           layout_transformer: Optional[LayoutTransformerArtifacts] = None
       )

An executor class to run models compiled by either the ``ModelBuilder``
or ``trace()``, ``compile()`` fundamental units.

.. _constructor-parameters-2:

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~

- **world_size: int —** Total number of ranks/processes in the
  distributed setup.
- **start_rank: Optional[int], default=None —** Starting rank for this
  instance. If None, defaults to 0.
- **local_ranks_size: Optional[int], default=None —** Number of local
  ranks. Must be specified if start_rank is provided.
- **state_initializer: Optional[StateInitializer], default=None —**
  Initializer for model states. If not provided, stateful model tensors
  will be initialized with zeros.

neuronx_distributed.NxDModel.add
================================

::

   @torch.jit.unused
   def add(
       self,
       key: str,
       trace_artifacts: TraceArtifacts,
       compilation_artifacts: Union[CompilationArtifacts, WLOArtifacts],
   ) -> "NxDModel"

Add a compiled submodel to this ``NxDModel`` instance.

**Notes:**

- Creates a ``StateInitializer`` if state tensors are present in the
  metaneff, and none was provided in the ``NxDModel`` constructor
- Sets up ``SPMDModel`` instances and input/output processing components

.. _parameters-5:

Parameters
~~~~~~~~~~

- **key: str —** Unique identifier for this submodel within the
  ``NxDModel``
- **trace_artifacts: TraceArtifacts —** Artifacts produced from the
  ``trace()`` function
- **compilation_artifacts:** CompilationArtifacts — Artifacts produced
  from the ``compile()`` or ``compile_wlo()`` functions

.. _returns-5:

Returns
~~~~~~~

``NxDModel`` self reference, enabling builder-style method chaining.

neuronx_distributed.NxDModel.get_neff
=====================================

::

   @torch.jit.unused
   def get_neff(self, key: str) -> bytes

Retrieves the NEFF (Neuron Executable File Format) from the specified
model. Requires the associated model to already be added using the
``add()`` method.

.. _parameters-6:

Parameters
~~~~~~~~~~

- **key: str —** The identifier for the model whose NEFF should be
  retrieved.

.. _returns-6:

Returns
~~~~~~~

``bytes`` — The NEFF for the specified model

.. _raises-6:

Raises
~~~~~~

- ``KeyError``: If the specified key is not found in the available keys.
- ``RuntimeError``: If there is an error retrieving the NEFF.

neuronx_distributed.NxDModel.get_metaneff
=========================================

::

   @torch.jit.unused
   def get_metaneff(self, key: str) -> metaneff_pb2.MetaNeff

Retrieves the metaneff from the specified model. Requires the associated
model to already be added using the ``add()`` method.

.. _parameters-7:

Parameters
~~~~~~~~~~

- **key: str** — The identifier for the model whose metaneff should be
  retrieved.

.. _returns-7:

Returns
~~~~~~~

``metaneff_pb2.MetaNeff`` — The metaneff proto object for the specified
model.

.. _raises-7:

Raises
~~~~~~~

- ``KeyError``: If the specified key is not found in the available keys. 
- ``RuntimeError``: If there is an error retrieving the metaneff.

neuronx_distributed.NxDModel.get_hlo
====================================

::

   @torch.jit.unused
   def get_hlo(self, key: str) -> hlo_pb2.HloModuleProto

Retrieves the HLO from the specified model. Requires the associated
model to already be added using the ``add()`` method.

.. _parameters-8:

Parameters
~~~~~~~~~~

- **key: str** — The identifier for the model whose HLO should be
  retrieved.

.. _returns-8:

Returns
~~~~~~~

``hlo_pb2.HloModuleProto`` — The HLO module proto object for the
specified model.

.. _raises-8:

Raises
~~~~~~

- ``KeyError``: If the specified key is not found in the available keys.
- ``RuntimeError``: If there is an error retrieving the metaneff. 


neuronx_distributed.NxDModel.set_weights
========================================

::

   @torch.jit.export
   def set_weights(
       self,
       sharded_checkpoint: List[Dict[str, torch.Tensor]]
   )

Set the model’s weights from a sharded checkpoint.

This function initializes the model’s weights using a sharded
checkpoint. The checkpoint is processed and loaded using either a layout
transformer (if provided) or a direct parallel loading mechanism.

This function should only be called before the model is loaded onto a
Neuron device. Once the model is loaded, use the
``replace_weights()`` method to update the weights.

.. _parameters-9:

Parameters
~~~~~~~~~~

- **sharded_checkpoint: List[Dict[str, torch.Tensor]]** — \***\* A list
  of state dicts mapping parameter names to their corresponding tensor
  values for each rank.

.. _returns-9:

Returns
~~~~~~~

``None``

.. _raises-9:

Raises
~~~~~~

``ValueError``: If the model is already loaded on a Neuron device.

neuronx_distributed.NxDModel.to_neuron
======================================

::

   @torch.jit.export
   def to_neuron(self)

Loads the model onto Neuron Devices.

This function initializes the model onto Neuron Hardware. Must be called
before executing the model, otherwise the forward method will raise a
``RuntimeError``.

.. _returns-10:

Returns
~~~~~~~

``None``

neuronx_distributed.NxDModel.replace_weights
============================================

::

   @torch.jit.export
   def replace_weights(
       self,
       sharded_checkpoint: List[Dict[str, torch.Tensor]]
   )

Replace the model’s weights and reload onto Neuron devices.

This method should be used instead of ``set_weights()`` when the model
is already loaded on Neuron devices and weights need to be updated.

.. _parameters-10:

Parameters
~~~~~~~~~~

- **sharded_checkpoint: List[Dict[str, torch.Tensor]]** — \***\* A list
  of state dicts mapping parameter names to their corresponding tensor
  values for each rank.

.. _returns-11:

Returns
~~~~~~~

``None``

neuronx_distributed.NxDModel.read_from_neuron_buffer
====================================================

::

   @torch.jit.export
   def read_from_neuron_buffer(
       self,
       buffer_key: str,
       rank: int
   ) -> torch.Tensor

Reads a tensor value from a Neuron device buffer to CPU, based on given
key and rank.

.. _parameters-11:

Parameters
~~~~~~~~~~

- **buffer_key: str** — The key identifying the specific buffer
  to retrieve.
- **rank: int** — The rank from which to retrieve the buffer.

.. _returns-12:

Returns
~~~~~~~

``torch.Tensor``: The requested tensor buffer copied to Host memory.

.. _raises-12:

Raises
~~~~~~

- ``AssertionError``: If this method is called before to_neuron()
- ``KeyError``: If the specified state_buffer_key does not exist in the states for the given rank.

neuronx_distributed.NxDModel.write_to_neuron_buffer
===================================================

::

   @torch.jit.export
   def write_to_neuron_buffer(
       self,
       tensor: torch.Tensor,
       buffer_key: str,rank: int
   )

Write a tensor to a specific Neuron device buffer.

This function updates a state buffer on a Neuron device by copying
values from the provided tensor. The destination buffer must already
exist and have the same shape as the input tensor.

.. _parameters-12:

Parameters
~~~~~~~~~~

- **tensor: torch.Tensor** — The tensor containing the data to be
  written to the buffer.
- **buffer_key: str** — The key identifying the specific buffer
  to update.
- **rank: int** — The rank where the buffer is located.

.. _returns-13:

Returns
~~~~~~~

``None``

.. _raises-13:

Raises
~~~~~~~

- ``AssertionError``: If this method is called before ``to_neuron()``.
- ``KeyError``: If the specified ``state_buffer_key`` does not exist in the states for the given rank, or if the shapes of the input tensor and target buffer do not match.

neuronx_distributed.NxDModel.forward
====================================

::

   def forward(
       self,
       *args,
       model_name: Optional[str] = None,
       forward_mode='default',
       **kwargs
   ):

The forward method of the NxDModel class, which will take in inputs and
run the respective NEFF.

.. _parameters-13:

Parameters
~~~~~~~~~~

- **args: Union[torch.Tensor, List[torch.Tensor]]** — Positional
  tensor inputs to model. List form must be used if
  ``forward_mode != 'default'``.
- **model_name: Optional[str]** — Parameter to pass in a specific
  key to execute. This must be used in cases of ambiguous routing.
- **forward_mode: str, default=‘default’** — There are 3
  supported modes: default, ranked, async.

  - **default**: This takes in inputs, replicates them across ranks,
    executes the model, and only returns the outputs from rank 0
  - **ranked:** This takes in inputs in ranked form, meaning each
    individual tensor input (ie each ``arg`` in ``*args``) must be a list
    of tensors whose length is equal to the world size of the compiled
    model. The model will execute, and return a ranked output, which is
    a ``List`` of all outputs by rank (ie a
    ``List[List[torch.Tensor]]``.
  - **async:** Like ranked, this takes in inputs and returns outputs in
    ranked form, except the major difference is that the outputs will be
    returned instantly, and will be references to buffers where the
    model will write the output once the NEFF is done executing. To
    block on the NEFF call, you must call ``.cpu()`` for each tensor in
    the output.

- ****kwargs (torch.Tensor, List[torch.Tensor])** — Keyword arguments
  corresponding to specific input tensors to the model. List form must
  be used if ``forward_mode != 'default'``.

.. _returns-14:

Returns
~~~~~~~

It depends on the ``forward_mode`` setting: 

- **default:** Expected format of tensor outputs based on what was originally traced.
- **ranked or async:** ``List[List[torch.Tensor]]`` of shape (num_out_tensors, world_size).

neuronx_distributed.NxDModel.save
=================================

::

   def save(self, path_to_save: str, save_weights: bool = False)

Saves the model as a TorchScript module to the specified path. The saved
artifact can be loaded with ``NxDModel.load`` or ``torch.jit.load``
(``NxDModel.load`` is preferrable).

.. _parameters-14:

Parameters
~~~~~~~~~~

- **path_to_save: str** — The file path where the TorchScript
  model should be saved.
- **save_weights: Optional[bool], default=False** — If ``True``,
  preserves the weights within the TorchScript model. It is ``False`` by
  default.

.. _returns-15:

Returns
~~~~~~~

``None``

neuronx_distributed.NxDModel.load
=================================

::

   @classmethod
   def load(
       cls,
       path_to_model: str,
       start_rank: Optional[int] = None,
       local_ranks_size: Optional[int] = None
   ) -> Union["NxDModel", torch.jit.ScriptModule]

Attempts to load and restore an ``NxDModel`` from a saved TorchScript
model.

This classmethod tries to reconstruct an NxDModel instance from a
previously saved TorchScript model. If the restoration process fails, it
returns the loaded TorchScript model instead, as backwards compatibility
is not guaranteed across different versions of NxD.

.. _parameters-15:

Parameters
~~~~~~~~~~

- **path_to_model: str** — Path to the saved TorchScript model
  file.
- **start_rank: Optional[int], default=None** — Starting rank for
  distributed processing. If ``None``, and ``local_ranks_size`` is set,
  an ``AssertionError`` will be raised. Defaults to ``None``
- **local_ranks_size: Optional[int], default=None** — Size of
  local_ranks for distribtued processing. Must be set if ``start_rank``
  is provided. Defaults to ``None``

.. _returns-16:

Returns
~~~~~~~

``Union[NxDModel, torch.jit.ScriptModule]``: Either the restored
``NxdModel`` instance, or the loaded TorchScript model if restoration
fails.

.. _raises-16:

Raises
~~~~~~~

- ``ValueError``: If the provided model was not originally saved using ``NxDModel.save()``.
- ``AssertionError``: If ``start_rank``/``local_ranks_size`` parameters are inconsistently set.

Usage Notes
===========

In-place buffer updates
~~~~~~~~~~~~~~~~~~~~~~~

Description
~~~~~~~~~~~

ModelBuilderV2 enables users to update model buffers in-place during
their model’s ``forward`` pass. In-place updates enable users to
efficiently utilize memory when caching values during the ``forward``
pass. An example use case for in-place updates is the population of a
model’s KV Cache.

Under the hood, ModelBuilderV2 detects when buffers are mutated during
``forward`` while tracing a model, and uses `XLA’s
aliasing <https://openxla.org/xla/aliasing>`__ to ensure that buffers
are mutated in-place.

Supported Usage
~~~~~~~~~~~~~~~

In-place updates are currently supported for the following combinations
of ``torch.Tensor`` subclasses and torch operations:

+-----------------------+-----------------------+-----------------------+
| Tensor class          | Out of place torch    | In place torch        |
|                       | operation             | operation             |
+=======================+=======================+=======================+
| torch.nn.Buffer,      | Supported             | Not Supported         |
| persistent=True       |                       |                       |
+-----------------------+-----------------------+-----------------------+
| torch.nn.Buffer,      | Supported             | Not Supported         |
| persistent=False      |                       |                       |
+-----------------------+-----------------------+-----------------------+
| torch.nn.Parameter    | Not Supported         | Not Supported         |
+-----------------------+-----------------------+-----------------------+

Additionally, the following forms of updates are not supported, because
these mutations change the memory utilization or memory layout of the
mutated tensor:

- Updating the ``dtype`` of a buffer or parameter during ``forward``.
- Updating the ``shape`` of a buffer or parameter during ``forward``.

.. _supported-usage-1:

Supported Usage:
~~~~~~~~~~~~~~~~

::

   import torch
   import torch.nn as nn

   class ExampleModel(nn.Module):
       def __init__(self):
           super().__init__()
           
           self.register_buffer("buffer_persistent", torch.zeros(10), dtype=torch.bfloat16, persistent=True)
           self.register_buffer("buffer_nonpersistent", torch.zeros(10), dtype=torch.bfloat16, persistent=False)
           self.parameter = nn.Parameter(torch.zeros(10), dtype=torch.bfloat16)
           
       def forward(self, x, dim_tensor, index, src):
           # supported: buffers with out of place torch operations
           self.buffer_persistent = self.buffer_persistent + 1
           self.buffer_nonpersistent = torch.scatter(self.buffer_persistent, dim_tensor, index, src)
           
           # not supported: buffers with inplace torch operations
           self.buffer_persistent.scatter_(dim_tensor, index, src)
           self.buffer_nonpersistent.index_copy_(dim_tensor, index, src)
           
           # not supported: parameters
           self.parameter = torch.scatter(self.paramter, dim_tensor, index, src)
           self.parameter.scatter_(dim_tensor, index, src)
           
           # not supported: dtype updates
           self.buffer_persistent = self.buffer_persistent.to(torch.float32)
           
           # not supported: shape changes
           self.buffer_persistent = torch.reshape(self.buffer_persistent.reshape, (2, 5))

Usage Examples
==============

E2E with ModelBuilder APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Build and run callable with ModelBuilder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import torch
   import torch.nn as nn
   from neuronx_distributed import ModelBuilder

   torch.manual_seed(0)

   def func(a, b):
       return a + b

   nxd_model = ModelBuilder(func) \
       .trace(kwargs={'a': torch.rand(2,2), 'b': torch.rand(2,2)}, tag="key1") \
       .compile()

   nxd_model.to_neuron()
   input = (torch.rand(2, 2), torch.rand(2, 2))
   cpu_out = func(a=input[0], b=input[1])
   neuron_out = nxd_model(a=input[0], b=input[1])

   torch.testing.assert_close(cpu_out, neuron_out)

Example: Build and run torch module with ModelBuilder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import torch
   import torch.nn as nn
   from neuronx_distributed.utils.model_utils import init_on_device
   from neuronx_distributed import NxDParallelState, shard_checkpoint, ModelBuilder
   from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear

   torch.manual_seed(0)

   class Model(nn.Module):
       def __init__(self, is_distributed=True):
           super().__init__()
           if is_distributed:
               self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
               self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)
           else:
               self.layer1 = nn.Linear(1024, 1024)
               self.layer2 = nn.Linear(1024, 1024)
       def forward(self, x):
           x = self.layer1(x)
           return self.layer2(x)

   cpu_model = Model(is_distributed=False)
   model_checkpoint = cpu_model.state_dict()

   with NxDParallelState(world_size=32, tensor_model_parallel_size=32):
       model = Model()

       example_inputs = torch.rand(32, 1024)

       nxd_model = ModelBuilder(model) \
           .trace(args=example_inputs, tag="key1") \
           .compile()

   with NxDParallelState(world_size=32, tensor_model_parallel_size=32), init_on_device(torch.device("meta")):
       sharded_checkpoint = shard_checkpoint(
           checkpoint=model_checkpoint,
           model=Model()
       )

   nxd_model.set_weights(sharded_checkpoint)
   nxd_model.to_neuron()

   input = torch.ones(32, 1024)
   cpu_out = cpu_model(input)
   neuron_out = nxd_model(x=input)

Example: Multi-bucket trace with ModelBuilder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import torch
   import torch.nn as nn
   from neuronx_distributed.utils.model_utils import init_on_device
   from neuronx_distributed import NxDParallelState, shard_checkpoint, ModelBuilder
   from neuronx_distributed.parallel_layers import ColumnParallelLinear

   torch.manual_seed(0)

   class Model(nn.Module):
       def __init__(self, is_distributed=True):
           super().__init__()
           if is_distributed:
               self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=True)
               self.layer2 = ColumnParallelLinear(1024, 1024, gather_output=True)
           else:
               self.layer1 = nn.Linear(1024, 1024)
               self.layer2 = nn.Linear(1024, 1024)
       def forward(self, x):
           x = self.layer1(x)
           return self.layer2(x)

   cpu_model = Model(is_distributed=False)
   model_checkpoint = cpu_model.state_dict()

   with NxDParallelState(world_size=32, tensor_model_parallel_size=32):
       model = Model()

       example_inputs1 = torch.rand(32, 1024)
       example_inputs2 = torch.rand(16, 1024)
       
       nxd_model = ModelBuilder(model) \
           .trace(args=example_inputs1, tag="bucket1") \
           .trace(args=example_inputs2, tag="bucket2") \
           .compile()


   with NxDParallelState(world_size=32, tensor_model_parallel_size=32), init_on_device(torch.device("meta")):
       sharded_checkpoint = shard_checkpoint(
           checkpoint=model_checkpoint,
           model=Model()
       )

   nxd_model.set_weights(sharded_checkpoint)
   nxd_model.to_neuron()

   input1 = torch.rand(32, 1024)
   input2 = torch.rand(16, 1024)

   for input in [input1, input2]:
       cpu_out = cpu_model(input)
       neuron_out = nxd_model(input)
       torch.testing.assert_close(cpu_out, neuron_out)

Example: Build and run torch module with ModelBuilder where example inputs are supplied as kwargs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import torch
   import torch.nn as nn
   from neuronx_distributed.utils.model_utils import init_on_device
   from neuronx_distributed import NxDParallelState, shard_checkpoint, ModelBuilder
   from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear

   torch.manual_seed(0)

   class Model(nn.Module):
       def __init__(self, is_distributed=True):
           super().__init__()
           if is_distributed:
               self.layer1 = ColumnParallelLinear(5, 10, gather_output=True)
               self.layer2 = ColumnParallelLinear(20, 10, gather_output=True)
           else:
               self.layer1 = nn.Linear(5, 10)
               self.layer2 = nn.Linear(20, 10)

       def forward(self, x, y):
           return self.layer1(x) + self.layer2(y)

   cpu_model = Model(is_distributed=False)
   model_checkpoint = cpu_model.state_dict()

   with NxDParallelState(world_size=2, tensor_model_parallel_size=2):
       model = Model()

       example_inputs1 = {'x': torch.rand(10, 5), 'y': torch.rand(10, 20)}
       example_inputs2 = {'x': torch.rand(50, 5), 'y': torch.rand(50, 20)}
       
       nxd_model = ModelBuilder(model) \
           .trace(kwargs=example_inputs1, tag="bucket1") \
           .trace(kwargs=example_inputs2, tag="bucket2") \
           .compile()


   with NxDParallelState(world_size=2, tensor_model_parallel_size=2), init_on_device(torch.device("meta")):
       sharded_checkpoint = shard_checkpoint(
           checkpoint=model_checkpoint,
           model=Model()
       )

   nxd_model.set_weights(sharded_checkpoint)
   nxd_model.to_neuron()

   input1 = (torch.rand(10, 5), torch.rand(10, 20))
   input2 =  (torch.rand(50, 5), torch.rand(50, 20))

   for input in [input1, input2]:
       cpu_out = cpu_model(input[0], input[1])
       neuron_out = nxd_model(x=input[0], y=input[1])
       torch.testing.assert_close(cpu_out, neuron_out)

Example: Build and run torch module with in-place buffer updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import torch
   from neuronx_distributed import ModelBuilder

   torch.manual_seed(0)

   class Model(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.register_buffer('cache', torch.tensor([0], dtype=torch.float32), persistent=True)

       def forward(self, x, update_value):
           self.cache = torch.add(self.cache, update_value)
           return x + self.cache

   cpu_model = Model()

   model = Model()

   example_inputs1 = {'x': torch.zeros(1, dtype=torch.float32), 'update_value': torch.zeros(1, dtype=torch.float32)}

   nxd_model = ModelBuilder(model) \
       .trace(kwargs=example_inputs1, tag="bucket1") \
       .compile()

   state_dict = [
       {
           "cache": torch.tensor([0])
       }
   ]
   nxd_model.set_weights(state_dict)
   nxd_model.to_neuron()

   input1 = (torch.tensor([1], dtype=torch.float32), torch.tensor([5], dtype=torch.float32))
   input2 =  (torch.tensor([2], dtype=torch.float32), torch.tensor([10], dtype=torch.float32))

   model_iteration = 0
   for input in [input1, input2]:
       cpu_out = cpu_model(input[0], input[1])
       neuron_out = nxd_model(x=input[0], update_value=input[1])
       
       torch.testing.assert_close(cpu_out, neuron_out)
       model_iteration += 1
       print(f"Iteration {model_iteration} matches!")

E2E with Fundamental Units
~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Build and run Callable with Fundamental Units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import torch
   from neuronx_distributed import NxDModel
   from neuronx_distributed.trace.model_builder import trace, compile

   torch.manual_seed(0)

   def func(a,b):
       return a + b

   trace_artifacts = trace(func, kwargs={'a': torch.rand(2,2), 'b': torch.rand(2,2)})
   compilation_artifacts = compile(trace_artifacts.hlo, trace_artifacts.metaneff)

   nxd_model = NxDModel(world_size=1)
   nxd_model.add('func', trace_artifacts, compilation_artifacts)
   nxd_model.to_neuron()

   cpu_out = func(torch.ones(2, 2), torch.ones(2, 2))
   neuron_out = nxd_model(torch.ones(2,2), torch.ones(2,2))
   torch.testing.assert_close(cpu_out, neuron_out)

Example: Build and run torch module with Fundamental Units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import os
   import shutil
   import torch
   import torch.nn as nn

   from neuronx_distributed.utils.model_utils import init_on_device
   from neuronx_distributed import NxDParallelState, shard_checkpoint, ModelBuilder, NxDModel
   from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
   from neuronx_distributed.trace.model_builder_utils import ModelBuilderConstants
   from neuronx_distributed.trace.model_builder import (
       trace,
       compile,
   ) 

   torch.manual_seed(0)

   class Model(nn.Module):
       def __init__(self, is_distributed=True):
           super().__init__()
           if is_distributed:
               self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
               self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)
           else:
               self.layer1 = nn.Linear(1024, 1024)
               self.layer2 = nn.Linear(1024, 1024)
       def forward(self, x):
           x = self.layer1(x)
           return self.layer2(x)

   cpu_model = Model(is_distributed=False)
   model_checkpoint = cpu_model.state_dict()

   with NxDParallelState(world_size=32, tensor_model_parallel_size=32):
       model = Model()

       example_inputs = torch.rand(32, 1024)

       trace_artifacts = {
           "bucket1": trace(model, args=example_inputs),
       }

       compilation_artifacts_priority = compile(
           hlo_module=trace_artifacts["bucket1"].hlo,
           metaneff=trace_artifacts["bucket1"].metaneff,
           key="bucket1"
       )

   with NxDParallelState(world_size=32, tensor_model_parallel_size=32), init_on_device(torch.device("meta")):
       sharded_checkpoint = shard_checkpoint(
           checkpoint=model_checkpoint,
           model=Model()
       )

   nxd_model = NxDModel(world_size=32)
   nxd_model.add(key="bucket1", trace_artifacts=trace_artifacts["bucket1"], compilation_artifacts=compilation_artifacts_priority)

   nxd_model.set_weights(sharded_checkpoint)
   nxd_model.to_neuron()

   input = torch.rand(32, 1024)

   cpu_out = cpu_model(input)
   neuron_out = nxd_model(input)
   torch.testing.assert_close(cpu_out, neuron_out)

Example: Multi-bucket trace with Fundamental Units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import os
   import shutil
   import torch
   import torch.nn as nn

   from neuronx_distributed.utils.model_utils import init_on_device
   from neuronx_distributed import NxDParallelState, shard_checkpoint, ModelBuilder, NxDModel
   from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
   from neuronx_distributed.trace.model_builder_utils import ModelBuilderConstants
   from neuronx_distributed.trace.model_builder import (
       trace,
       compile,
   ) 

   torch.manual_seed(0)

   class Model(nn.Module):
       def __init__(self, is_distributed=True):
           super().__init__()
           if is_distributed:
               self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
               self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)
           else:
               self.layer1 = nn.Linear(1024, 1024)
               self.layer2 = nn.Linear(1024, 1024)
       def forward(self, x):
           x = self.layer1(x)
           return self.layer2(x)

   cpu_model = Model(is_distributed=False)
   model_checkpoint = cpu_model.state_dict()

   with NxDParallelState(world_size=32, tensor_model_parallel_size=32):
       model = Model()

       example_inputs1 = torch.rand(32, 1024)
       example_inputs2 = torch.rand(16, 1024)

       trace_artifacts = {
           "bucket1": trace(model, args=example_inputs1),
           "bucket2": trace(model, args=example_inputs2),
       }

       compilation_artifacts_bucket1 = compile(
           hlo_module=trace_artifacts["bucket1"].hlo,
           metaneff=trace_artifacts["bucket1"].metaneff,
           key="bucket1"
       )
       compilation_artifacts_bucket2 = compile(
           hlo_module=trace_artifacts["bucket2"].hlo,
           metaneff=trace_artifacts["bucket2"].metaneff,
           key="bucket2"
       )

   with NxDParallelState(world_size=32, tensor_model_parallel_size=32), init_on_device(torch.device("meta")):
       sharded_checkpoint = shard_checkpoint(
           checkpoint=model_checkpoint,
           model=Model()
       )

   nxd_model = NxDModel(world_size=32)
   nxd_model.add(key="bucket1", trace_artifacts=trace_artifacts["bucket1"], compilation_artifacts=compilation_artifacts_bucket1)
   nxd_model.add(key="bucket2", trace_artifacts=trace_artifacts["bucket2"], compilation_artifacts=compilation_artifacts_bucket2)

   nxd_model.set_weights(sharded_checkpoint)
   nxd_model.to_neuron()

   input1 = torch.rand(32, 1024)
   input2 = torch.rand(16, 1024)

   for input in [input1, input2]:
       cpu_out = cpu_model(input)
       neuron_out = nxd_model(input)
       torch.testing.assert_close(cpu_out, neuron_out)

Example: Build and run torch module with Fundamental Units where example inputs are supplied as kwargs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import os
   import shutil
   import torch
   import torch.nn as nn

   from neuronx_distributed.utils.model_utils import init_on_device
   from neuronx_distributed import NxDParallelState, shard_checkpoint, ModelBuilder, NxDModel
   from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
   from neuronx_distributed.trace.model_builder_utils import ModelBuilderConstants
   from neuronx_distributed.trace.model_builder import (
       trace,
       compile,
   ) 

   torch.manual_seed(0)

   class Model(nn.Module):
       def __init__(self, is_distributed=True):
           super().__init__()
           if is_distributed:
               self.linear1 = ColumnParallelLinear(5, 10, gather_output=True)
               self.linear2 = ColumnParallelLinear(20, 10, gather_output=True)
           else:
               self.linear1 = nn.Linear(5, 10)
               self.linear2 = nn.Linear(20, 10)

       def forward(self, x, y):
           return self.linear1(x) + self.linear2(y)

   cpu_model = Model(is_distributed=False)
   model_checkpoint = cpu_model.state_dict()

   with NxDParallelState(world_size=2, tensor_model_parallel_size=2):
       model = Model()

       example_inputs1 = {'x': torch.rand(10, 5), 'y': torch.rand(10, 20)}
       example_inputs2 = {'x': torch.rand(50, 5), 'y': torch.rand(50, 20)}

       trace_artifacts = {
           "bucket1": trace(model, kwargs=example_inputs1),
           "bucket2": trace(model, kwargs=example_inputs2),
       }

       compilation_artifacts_bucket1 = compile(
           hlo_module=trace_artifacts["bucket1"].hlo,
           metaneff=trace_artifacts["bucket1"].metaneff,
           key="bucket1"
       )
       compilation_artifacts_bucket2 = compile(
           hlo_module=trace_artifacts["bucket2"].hlo,
           metaneff=trace_artifacts["bucket2"].metaneff,
           key="bucket2"
       )

   with NxDParallelState(world_size=2, tensor_model_parallel_size=2), init_on_device(torch.device("meta")):
       sharded_checkpoint = shard_checkpoint(
           checkpoint=model_checkpoint,
           model=Model()
       )

   nxd_model = NxDModel(world_size=2)
   nxd_model.add(key="bucket1", trace_artifacts=trace_artifacts["bucket1"], compilation_artifacts=compilation_artifacts_bucket1)
   nxd_model.add(key="bucket2", trace_artifacts=trace_artifacts["bucket2"], compilation_artifacts=compilation_artifacts_bucket2)

   nxd_model.set_weights(sharded_checkpoint)
   nxd_model.to_neuron()

   input1 = (torch.rand(10, 5), torch.rand(10, 20))
   input2 =  (torch.rand(50, 5), torch.rand(50, 20))

   for input in [input1, input2]:
       cpu_out = cpu_model(input[0], input[1])
       neuron_out = nxd_model(x=input[0], y=input[1])
       torch.testing.assert_close(cpu_out, neuron_out)

Example: Build and run torch module with in-place buffer updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   import torch

   from neuronx_distributed import NxDModel
   from neuronx_distributed.trace.model_builder import (
       trace,
       compile,
   ) 

   torch.manual_seed(0)

   class Model(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.register_buffer('cache', torch.tensor([0], dtype=torch.float32), persistent=True)

       def forward(self, x, update_value):
           self.cache = torch.add(self.cache, update_value)
           return x + self.cache

   cpu_model = Model()

   model = Model()

   example_inputs1 = {'x': torch.zeros(1, dtype=torch.float32), 'update_value': torch.zeros(1, dtype=torch.float32)}

   trace_artifacts = {
       "bucket1": trace(model, kwargs=example_inputs1),
   }

   compilation_artifacts_bucket1 = compile(
       hlo_module=trace_artifacts["bucket1"].hlo,
       metaneff=trace_artifacts["bucket1"].metaneff,
       key="bucket1"
   )


   nxd_model = NxDModel(world_size=1)
   nxd_model.add(key="bucket1", trace_artifacts=trace_artifacts["bucket1"], compilation_artifacts=compilation_artifacts_bucket1)

   state_dict = [
       {
           "cache": torch.tensor([0], dtype=torch.float32)
       }
   ]
   nxd_model.set_weights(state_dict)
   nxd_model.to_neuron()

   input1 = (torch.tensor([1], dtype=torch.float32), torch.tensor([5], dtype=torch.float32))
   input2 =  (torch.tensor([2], dtype=torch.float32), torch.tensor([10], dtype=torch.float32))

   model_iteration = 0
   for input in [input1, input2]:
       cpu_out = cpu_model(input[0], input[1])
       neuron_out = nxd_model(x=input[0], update_value=input[1])
       
       torch.testing.assert_close(cpu_out, neuron_out)
       model_iteration += 1
       print(f"Iteration {model_iteration} matches!")
