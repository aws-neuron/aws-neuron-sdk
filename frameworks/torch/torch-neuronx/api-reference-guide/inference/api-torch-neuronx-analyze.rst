.. _torch_neuronx_analyze_api:

PyTorch NeuronX Analyze API for Inference
============================================================

.. py:function:: torch_neuronx.analyze(func, example_inputs, compiler_workdir=None)

   Checks the support of the operations in the ``func`` by checking each operator against neuronx-cc.

   :arg ~torch.nn.Module,callable func: The function/module that that will be
      run using the ``example_inputs`` arguments in order to record the
      computation graph.
    
   :arg ~torch.Tensor,tuple[~torch.Tensor] example_inputs: A tuple of example
      inputs that will be passed to the ``func`` while tracing.

   :keyword str compiler_workdir: Work directory used by
      |neuronx-cc|. This can be useful for debugging and/or inspecting
      intermediary |neuronx-cc| outputs
   
   :keyword set additional_ignored_ops: A set of aten operators to not analyze. Default is an empty set.
   
   :keyword int max_workers: The max number of workers threads to spawn.
      The default is ``4``.
   
   :keyword bool is_hf_transformers: If the model is a huggingface transformers model,
      it is recommended to enable this option to prevent deadlocks. Default is ``False``.
   
   :keyword bool cleanup: Specifies whether to delete the compiler artifact directories
      generated after running analyze. Default is ``False``.
   

   :returns: A JSON like :class:`~Dict` with the supported operators and their count, and unsupported
      operators with the failure mode and location of the operator in the python code.
    
   :rtype: :class:`~Dict`


   .. rubric:: Notes

      This function is meant to be used as a way to evaluate operator support for the model that is intended to be traced.
      The information can be used to modify operators that are unsupported to ones that are supported, or custom partitioning
      of the model.

      Note that this API does not return a traced model.
      
      Just like torch.neuronx.trace, this API can be used on any EC2 machine with sufficient memory and compute resources.


   .. rubric:: Examples

   *Fully supported model*

   .. code-block:: python

      import json

      import torch
      import torch.nn as nn
      import torch_neuronx

      class MLP(nn.Module):
         def __init__(self, input_size=28*28, output_size=10, layers=[120,84]):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, layers[0])
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(layers[0], layers[1])
         def forward(self, x):
            f1 = self.fc1(x)
            r1 = self.relu(f1)
            f2 = self.fc2(r1)
            r2 = self.relu(f2)
            f3 = self.fc3(r2)
            return torch.log_softmax(f3, dim=1)
    
      model = MLP()
      ex_input = torch.rand([32,784])

      model_support = torch_neuronx.analyze(model,ex_input)
      print(json.dumps(model_support,indent=4))

   .. code-block::

     {
         "torch_neuronx_version": "1.13.0.1.5.0",
         "neuronx_cc_version": "2.0.0.11796a0+24a26e112",
         "support_percentage": "100.00%",
         "supported_operators": {
            "aten::linear": 3,
         "aten::relu": 2,
         "aten::log_softmax": 1
         },
         "unsupported_operators": []
      }
   
   *Unsupported Model/Operator*

   .. code-block:: python

      import json
      import torch
      import torch_neuronx

      def fft(x):
         return torch.fft.fft(x)

      model = fft
      ex_input = torch.arange(4)

      model_support = torch_neuronx.analyze(model,ex_input)
      print(json.dumps(model_support,indent=4))

   .. code-block::

      {
         "torch_neuronx_version": "1.13.0.1.5.0",
         "neuronx_cc_version": "2.0.0.11796a0+24a26e112",
         "support_percentage": "0.00%",
         "supported_operators": {},
         "unsupported_operators": [
            {
               "kind": "aten::fft_fft",
               "failureAt": "neuronx-cc",
               "call": "test.py(6): fft\n/home/ubuntu/testdir/venv/lib/python3.8/site-packages/torch_neuronx/xla_impl/analyze.py(35): forward\n/home/ubuntu/testdir/venv/lib/python3.8/site-packages/torch/nn/modules/module.py(1182): _slow_forward\n/home/ubuntu/testdir/venv/lib/python3.8/site-packages/torch/nn/modules/module.py(1194): _call_impl\n/home/ubuntu/testdir/venv/lib/python3.8/site-packages/torch/jit/_trace.py(976): trace_module\n/home/ubuntu/testdir/venv/lib/python3.8/site-packages/torch/jit/_trace.py(759): trace\n/home/ubuntu/testdir/venv/lib/python3.8/site-packages/torch_neuronx/xla_impl/analyze.py(302): analyze\ntest.py(11): <module>\n",
               "opGraph": "graph(%x : Long(4, strides=[1], requires_grad=0, device=cpu),\n      %neuron_4 : NoneType,\n      %neuron_5 : int,\n      %neuron_6 : NoneType):\n  %neuron_7 : ComplexFloat(4, strides=[1], requires_grad=0, device=cpu) = aten::fft_fft(%x, %neuron_4, %neuron_5, %neuron_6)\n  return (%neuron_7)\n"
            }
         ]
      }
   
   **Note:** the ``failureAt`` field can either be "neuronx-cc" or "Lowering to HLO". If the field is "neuronx-cc", then it indicates that the provided operator configuration failed to be compiled with ``neuronx-cc``. This could either indicate that the operator configuration is unsupported, or there is a bug with that operator configuration.

.. |neuronx-cc| replace:: :ref:`neuronx-cc <neuron-compiler-cli-reference-guide>`
