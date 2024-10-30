.. _api_guide_nxd_inference:

Inference APIs
==============

.. contents:: Table of contents
   :local:
   :depth: 2


.. _nxd_tracing:

Model Trace:
^^^^^^^^^^^^

We can use the tensor parallel layers to perform large model inference
too. For performing inference, we can re-use the Parallel model built
above for training and then use the trace APIs provided by the
neuronx_distributed package to trace it for inference. One can use the
following set of APIs for running distributed inference:

::

   def neuronx_distributed.trace.parallel_model_trace(func, example_inputs, compiler_workdir=None, compiler_args=None, inline_weights_to_neff=True, bucket_config=None, tp_degree=1, max_parallel_compilations=None)

This API would launch tensor parallel workers, where each worker would
trace its own model. These traced models would be wrapped with a single
TensorParallelModel module which can then be used like any other traced
model.

.. _parameters-9:

Parameters:


-  ``func : Callable``: This is a function that returns a ``Model``
   object and a dictionary of states. The ``parallel_model_trace`` API would call this function
   inside each worker and run trace against them. Note: This differs
   from the ``torch_neuronx.trace`` where the ``torch_neuronx.trace``
   requires a model object to be passed.

-  ``example_inputs: (torch.Tensor like)`` : The inputs that needs to be passed to
   the model. If you are using ``bucket_config``, then this must be a list of inputs for
   each bucket model. This configuration is similar to :func:`torch_neuronx.bucket_model_trace`

-  ``compiler_workdir: Optional[str,pathlib.Path]`` : Work directory used by
   |neuronx-cc|. This can be useful for debugging and inspecting
   intermediary |neuronx-cc| outputs.

-  ``compiler_args: Optional[Union[List[str],str]]`` : List of strings representing
   |neuronx-cc| compiler arguments. See :ref:`neuron-compiler-cli-reference-guide`
   for more information about compiler options.

-  ``inline_weights_to_neff: bool`` : A boolean indicating whether the weights should be
   inlined to the NEFF. If set to False, weights will be separated from the NEFF.
   The default is ``True``.

-  ``bucket_config: torch_neuronx.BucketModelConfig`` : The config object that defines
   bucket selection behavior. See :func:`torch_neuronx.BucketModelConfig` for more details.

-  ``tp_degree: (int)`` : How many devices to be used when performing
   tensor parallel sharding

-  ``max_parallel_compilations: Optional[int]`` : If specified, this function will only trace these numbers
   of models in parallel, which can be necessary to prevent OOMs while tracing. The default
   is None, which means the number of parallel compilations is equal to the ``tp_degree``.




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
