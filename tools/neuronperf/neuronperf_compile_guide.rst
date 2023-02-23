.. _neuronperf_compile_guide:

========================
NeuronPerf Compile Guide
========================

If you wish to compile multiple configurations at once, NeuronPerf provides a simplified and uniform API across frameworks. The output is a :ref:`neuronperf_model_index` that tracks the artifacts produces, and can be passed directly to the :ref:`benchmark <neuronperf_api_benchmark>` routine for a streamlined end-to-end process. This may be useful if you wish to test multiple configurations of your model on Neuron hardware.

You can manually specify the model index filename by passing ``filename``, or let NeuronPerf generate one and return it for you. Compiled artifacts will be placed in a local ``models`` directory.

How does ``compile`` know which instance type to compile for?
-------------------------------------------------------------

NeuronPerf will assume that the instance type your are currently on is also the compile target. However, you may compile on a non-Neuron instance or choose to target a different instance type. In the case, you can pass ``compiler_target`` to the ``compile`` call.

For example:

.. code:: python

   import neuronperf as npf
   import neuronperf.torch

   npf.torch.compile(model, inputs)  # compile for current instance type
   npf.torch.compile(model, inputs, compiler_target="inf2")  # compile for inf2



Compiling multiple variants
---------------------------

If you provide multople pipeline sizes, batch sizes, and/or cast modes, NeuronPerf will compile all of them.

.. code:: python

   # Select a few batch sizes and pipeline configurations to test
   batch_sizes = [1, 5, 10]
   pipeline_sizes = [1, 2, 4]

   # Construct example inputs
   example_inputs = [torch.zeros([batch_size, 3, 224, 224], dtype=torch.float16) for batch_size in batch_sizes]

   # Compile all configurations
   index = npf.torch.compile(
      model,
      example_inputs,
      batch_sizes=batch_sizes,
      pipeline_sizes=pipeline_sizes,
   )


If you wished to benchmark specific subsets of configurations, you could compile the specific configurations independently and later combine the results into a single index, as shown below.

.. code:: python

   # Compile with pipeline size 1 and vary batch dimension
   batch_index = npf.torch.compile(
      model,
      example_inputs,
      batch_sizes=batch_sizes,
      pipeline_sizes=1,
   )

   # Compile with batch size 1 and vary pipeline dimension
   pipeline_index = npf.torch.compile(
      model,
      example_inputs[0],
      batch_sizes=1,
      pipeline_sizes=pipeline_sizes,
   )

   index = npf.model_index.append(batch_index, pipeline_index)
   npf.model_index.save(index, 'model_index.json')

The ``compile`` function supports ``batch_sizes``, ``pipeline_sizes``, ``cast_modes``, and custom ``compiler_args``. If there is an error during compilation for a requested configuration, it will be logged and compilation will continue onward without terminating. (This is to support long-running compile jobs with many configurations.)

