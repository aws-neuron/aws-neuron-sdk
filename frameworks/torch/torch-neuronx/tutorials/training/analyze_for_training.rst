.. _torch-analyze-for-training-tutorial:

Analyze for Training Tutorial
==============================

This tutorial explains how to analyze a model for training support using via ``torch-neuronx``.

.. note::
    For analyzing models for inference support via ``torch-neuronx``, please refer to :ref:`torch_neuronx.analyze() <torch_neuronx_analyze_api>`

Setup
-----

For this tutorial we'll be using two scripts: ``supported.py`` and ``unsupported.py``. Create these files by copy pasting the below code to their respective files.

``supported.py``

.. code:: ipython3

    import torch
    import torch_xla.core.xla_model as xm

    class NN(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = torch.nn.Linear(4,4)
            self.nl1 = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(4,2)
            self.nl2 = torch.nn.Tanh()

        def forward(self, x):
            x = self.nl1(self.layer1(x))
            return self.nl2(self.layer2(x))
    
    
    def main():
        device = xm.xla_device()

        model = NN().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        inp = torch.rand(4)
        target = torch.tensor([1,0])

        model.train()
        for epoch in range(2):
            optimizer.zero_grad()
            inp = inp.to(device)
            target = target.to(device)
            output = model(inp)
            loss = loss_fn(output,target)
            loss.backward()
            optimizer.step()
            xm.mark_step()
    
    if __name__ == '__main__':
        main()


``unsupported.py``

.. code:: ipython3

    import torch
    import torch_xla.core.xla_model as xm

    class UnsupportedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y =  torch.fft.fft(x)
            x = x + 10
            return x * y
    
    
    def main():
        device = xm.xla_device()

        model = UnsupportedModel().to(device)

        inp = torch.rand(4)

        model.train()
        for epoch in range(1):
            inp = inp.to(device)
            output = model(inp)

            xm.mark_step()
    
    if __name__ == '__main__':
        main()

Running ``analyze`` via ``neuron_parallel_compile``
---------------------------------------------------

To analyze a model, we supply the training script to the ``analyze`` command, which is shipped with ``neuron_parallel_compile``.
The command is:

.. code:: shell

    neuron_parallel_compile --command analyze python supported.py

This will generate a lot of output showing a lot of compilation statuses.
Here's a snippet of the output when running the above command. 

.. code:: shell

    .2023-05-25 00:43:43.000394:  776642  INFO ||ANALYZE||: Compiling /tmp/model_analyis_graphs/compare_7841189860629745939_23.hlo.pb using following command: neuronx-cc compile --target=trn1 --framework XLA /tmp/model_analyis_graphs/compare_7841189860629745939_23.hlo.pb --verbose=35 --query-compute-placement 
    2023-05-25 00:43:43.000418:  776642  INFO ||ANALYZE||: Compiling /tmp/model_analyis_graphs/multiply_15640857564712679356_53.hlo.pb using following command: neuronx-cc compile --target=trn1 --framework XLA /tmp/model_analyis_graphs/multiply_15640857564712679356_53.hlo.pb --verbose=35 --query-compute-placement 
    .
    Compiler status PASS
    2023-05-25 00:43:43.000549:  776642  INFO ||ANALYZE||: Compiling /tmp/model_analyis_graphs/subtract_1927104012014828209_49.hlo.pb using following command: neuronx-cc compile --target=trn1 --framework XLA /tmp/model_analyis_graphs/subtract_1927104012014828209_49.hlo.pb --verbose=35 --query-compute-placement 
    ...
    Compiler status PASS


The analysis report will be generated as a JSON file.
The location of the report is shown as the last log entry:

.. code:: shell

    2023-05-25 00:43:49.000252:  776642  INFO ||ANALYZE||: Removing existing report /home/ubuntu/analyze_for_training/model_analysis_result/result.json
    2023-05-25 00:43:49.000252:  776642  INFO ||ANALYZE||: Model analysis completed. Report - /home/ubuntu/analyze_for_training/model_analysis_result/result.json

.. note::

    Note that if a report is already present in the specified path, ``analyze`` will remove/overwrite it.

The report generated running the above command looks like:

.. code:: json

    {
        "torch_neuronx_version": "1.13.0.1.6.1",
        "neuronx_cc_version": "2.5.0.28+1be23f232",
        "support_percentage": "100.00%",
        "supported_operators": {
            "aten": {
                "aten::permute": 8,
                "aten::add": 8,
                "aten::mul": 8,
                "aten::expand": 18,
                "aten::mm": 10,
                "aten::mse_loss_backward": 12,
                "aten::relu": 3,
                "aten::threshold_backward": 4,
                "aten::squeeze": 4,
                "aten::view": 4,
                "aten::pow": 2,
                "aten::mse_loss": 2,
                "aten::tanh": 2
            }
        },
        "unsupported_operators": {
            "aten": []
        }
    }

.. note::

    Note that the ``torch_neuronx`` and ``neuronx_cc`` versions may be different from this example

Understanding ``analyze`` report for Unsupported Models
-------------------------------------------------------

Default Verbosity
~~~~~~~~~~~~~~~~~

Let's run ``analyze`` for ``unsupported.py``

.. code:: shell

    neuron_parallel_compile --command analyze python unsupported.py

Here is the report generated by the above command:

.. code:: json

    {
        "torch_neuronx_version": "1.13.0.1.6.1",
        "neuronx_cc_version": "2.5.0.28+1be23f232",
        "support_percentage": "60.00%",
        "supported_operators": {
            "aten": {
                "aten::add": 2,
                "aten::mul": 1
            }
        },
        "unsupported_operators": {
            "aten": [
                {
                    "kind": "aten::mul",
                    "failureAt": "neuronx-cc",
                    "call": "test2_unsup.py 24"
                }
            ]
        }
    }

In the list of unsupported operators we are provided the specific aten op that failed, and where that operator is in the training script.

One thing to notice is that the ``support_percentage`` doesn't exactly add up. This is because the ``support_percentage`` is calculated based on the supported number of XLA/HLO instructions (explained more in the next section). To see the specific XLA/HLO op lowerings, use the flag ``--analyze-verbosity 1``, as the default is ``2``.

The last thing is that a specific aten operator can be supported and unsupported simultaneously. In our example, this can be seen with ``aten::mul``. This is due to the configuration of the aten op. The below section will describe what went wrong with the ``aten::mul`` op.

Lower Level Verbosity
~~~~~~~~~~~~~~~~~~~~~

Let's run again with lower verbosity level:

.. code:: shell

    neuron_parallel_compile --command analyze --analyze-verbosity 1 python unsupported.py

The report looks like:

.. code:: json

    {
        "torch_neuronx_version": "1.13.0.1.6.1",
        "neuronx_cc_version": "2.5.0.28+1be23f232",
        "support_percentage": "60.00%",
        "supported_operators": {
            "aten": {
                "aten::mul": 1,
                "aten::add": 2
            },
            "xla": [
                "f32[] multiply(f32[], f32[])",
                "f32[4]{0} broadcast(f32[]), dimensions={}",
                "f32[4]{0} add(f32[4]{0}, f32[4]{0})"
            ]
        },
        "unsupported_operators": {
            "aten": [
                {
                    "kind": "aten::mul",
                    "failureAt": "neuronx-cc",
                    "call": "test2_unsup.py 24"
                }
            ],
            "xla": [
                {
                    "hlo_instruction": "c64[4]{0} convert(f32[4]{0})",
                    "aten_op": "aten::mul"
                },
                {
                    "hlo_instruction": "c64[4]{0} multiply(c64[4]{0}, c64[4]{0})",
                    "aten_op": "aten::mul"
                }
            ]
        }
    }

This report provides both the aten operator and the failed XLA/HLO instructions. There will be more HLO instructions than aten ops since an aten op generally lowers to multiple HLO instructions. As a result, the ``support_percentage`` field doesn't exactly line up with the aten operator count, but does line up the XLA/HLO instruction count. This level of verbosity is intended for use when you have the ability to modify the model's HLO lowering, or generally have insight into the HLO lowering.

As mentioned before, the ``aten::mul`` op appears to be both supported and unsupported. This is because the compiler does not support a specific configuration of ``aten::mul``, which can be seen more clearly with the HLO lowering. In the above example, the ``aten::mul`` operator is unsupported since at least one parameter provided was a complex type (``C64``), which is unsupported by ``neuronx-cc``.

This concludes the tutorial. The API for ``analyze`` can be found within :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`
