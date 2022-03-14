.. _neuronperf_terminology:

NeuronPerf Terminology
======================

  * Model Inputs
    - An individual input or ``list`` of inputs
    - Example: ``inputs = [(torch.ones((batch_size, 5))) for batch_size in batch_sizes]``
    - Each input is associated with the ``batch_sizes`` specified, in the same order
    - Each input is fed individually to a corresponding model
    - If an input is provided as a ``tuple``, it will be destructured to ``model(*input)`` to support multiple args
    - See :ref:`neuronperf_framework_notes` for framework-specific requirements
  * Latency
  	- Time to execute a single ``model(input)``
  	- Typically measured in milliseconds
  * Model
   	- Your data model; varies by framework. See :ref:`neuronperf_framework_notes`
  	- Models may be wrapped by submodules (``torch``, ``tensorflow``, ``mxnet``) as callables
  * Model Index
  	- A JSON file that tracks compiled model artifacts
  * Model Inputs
  	- A ``tuple`` of inputs passed to a model, i.e. a single complete example
  	- Example: ``input = (torch.ones((5, 3, 224, 224)),)``
  * Throughput
  	- Inferences / second