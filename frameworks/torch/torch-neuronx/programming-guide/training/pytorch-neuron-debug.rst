.. _pytorch-neuronx-debug:

How to debug models in PyTorch Neuron (``torch-neuronx``)
=========================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Torch-XLA evaluates operations lazily, which means it builds a symbolic
graph in the background and the graph is executed in hardware only when
the users request (print) for the output or a mark_step is encountered.
To effectively debug training scripts with torch-xla, please use one of
the approaches mentioned below:

**Printing Metrics**
~~~~~~~~~~~~~~~~~~~~

Torch-xla provides a utility that records metrics of different sections
of the code. These metrics can help figure out things like: How much
time is spent in compilation? How much time is spent in execution? To
check the metrics:

1. Import metrics: ``import torch_xla.debug.metrics as met``
2. Print metrics at the end of the step: ``print(met.metrics_report())``

Printing metrics should produce an output that looks like this:

.. code:: bash

   Metric: CompileTime
     TotalSamples: 1
     Accumulator: 09s969ms486.408us
     Percentiles: 1%=09s969ms486.408us; 5%=09s969ms486.408us; 10%=09s969ms486.408us; 20%=09s969ms486.408us; 50%=09s969ms486.408us; 80%=09s969ms486.408us; 90%=09s969ms486.408us; 95%=09s969ms486.408us; 99%=09s969ms486.408us
   .....
   Metric: ExecuteTime
     TotalSamples: 1
     Accumulator: 186ms062.970us
     Percentiles: 1%=186ms062.970us; 5%=186ms062.970us; 10%=186ms062.970us; 20%=186ms062.970us; 50%=186ms062.970us; 80%=186ms062.970us; 90%=186ms062.970us; 95%=186ms062.970us; 99%=186ms062.970us
   ....
   Metric: TensorsGraphSize
     TotalSamples: 1
     Accumulator: 9.00
     Percentiles: 1%=9.00; 5%=9.00; 10%=9.00; 20%=9.00; 50%=9.00; 80%=9.00; 90%=9.00; 95%=9.00; 99%=9.00
   Metric: TransferFromServerTime
     TotalSamples: 2
     Accumulator: 010ms130.597us
     ValueRate: 549ms937.108us / second
     Rate: 108.372 / second
     Percentiles: 1%=004ms948.602us; 5%=004ms948.602us; 10%=004ms948.602us; 20%=004ms948.602us; 50%=006ms181.995us; 80%=006ms181.995us; 90%=006ms181.995us; 95%=006ms181.995us; 99%=006ms181.995us
   Metric: TransferToServerTime
     TotalSamples: 6
     Accumulator: 061ms698.791us
     ValueRate: 007ms731.182us / second
     Rate: 0.665369 / second
     Percentiles: 1%=006ms848.579us; 5%=006ms848.579us; 10%=006ms848.579us; 20%=007ms129.666us; 50%=008ms940.718us; 80%=008ms496.166us; 90%=024ms636.413us; 95%=024ms636.413us; 99%=024ms636.413us
   Metric: TransferToServerTransformTime
     TotalSamples: 6
     Accumulator: 011ms835.717us
     ValueRate: 001ms200.844us / second
     Rate: 0.664936 / second
     Percentiles: 1%=108.403us; 5%=108.403us; 10%=108.403us; 20%=115.676us; 50%=167.399us; 80%=516.659us; 90%=010ms790.400us; 95%=010ms790.400us; 99%=010ms790.400us
   .....
   Counter: xla::_copy_from
     Value: 7
   Counter: xla::addmm
     Value: 2
   Counter: xla::empty
     Value: 5
   Counter: xla::t
     Value: 2
   ....
   Metric: XrtCompile
     TotalSamples: 1
     Accumulator: 09s946ms607.609us
     Mean: 09s946ms607.609us
     StdDev: 000.000us
     Percentiles: 25%=09s946ms607.609us; 50%=09s946ms607.609us; 80%=09s946ms607.609us; 90%=09s946ms607.609us; 95%=09s946ms607.609us; 99%=09s946ms607.609us
   Metric: XrtExecute
     TotalSamples: 1
     Accumulator: 176ms932.067us
     Mean: 176ms932.067us
     StdDev: 000.000us
     Percentiles: 25%=176ms932.067us; 50%=176ms932.067us; 80%=176ms932.067us; 90%=176ms932.067us; 95%=176ms932.067us; 99%=176ms932.067us
   Metric: XrtReadLiteral
     TotalSamples: 2
     Accumulator: 608.578us
     Mean: 304.289us
     StdDev: 067.464us
     Rate: 106.899 / second
     Percentiles: 25%=236.825us; 50%=371.753us; 80%=371.753us; 90%=371.753us; 95%=371.753us; 99%=371.753us

As seen, you can get useful information about graph compile
times/execution times. You can also know which operators are present in
the graph, which operators are run on the CPU and which operators are run on an XLA device.
For example, operators that have a prefix ``aten::`` would run on the CPU, since they do not have
xla lowering. All operators with prefix ``xla::`` would run on an XLA device. Note: aten operators
that do not have xla lowering would result in a graph fragmentation and might end up slowing down the
entire execution. If you encounter such operators, create a request for operator support.

**Printing Tensors**
~~~~~~~~~~~~~~~~~~~~

Users can print tensors in their script as below:

.. code:: python

   import os
   import torch
   import torch_xla
   import torch_xla.core.xla_model as xm

   device = xm.xla_device()
   input1 = torch.randn(2,10).to(device)
   # Defining 2 linear layers
   linear1 = torch.nn.Linear(10,30).to(device)
   linear2 = torch.nn.Linear(30,20).to(device)

   # Running forward
   output1 = linear1(input1)
   output2 = linear2(output1)
   print(output2)

Since torch-xla evaluates operations lazily, when you try to print
``output2`` , the graph associated with the tensor would be evaluated.
When a graph is evaluated, it is first compiled for the device and executed on
the selected device. Note: Each tensor would have a graph associated
with it and can result in graph compilations and executions. For
example, in the above script, if you try to print ``output1`` , a new
graph is cut and you would see another evaluation. To avoid multiple evaluations, you can make use of ``mark_step`` (next section).

**Use mark_step**
~~~~~~~~~~~~~~~~~

Torch-XLA provides an api called ``mark_step`` which evaluates a graph
collected upto that point. While this is similar to printing of an output tensor
wherein a graph is also evaluated, there is a difference. When 
an output tensor is printed, only the graph associated with that specific tensor is
evaluated, whereas mark_step enables all the output tensors up to ``mark_step`` call to be evaluated
in a single graph. Hence, any tensor print after ``mark_step`` would be
effectively free of cost as the tensor values are already evaluated.
Consider the example below:

.. code:: python

   import os
   import torch
   import torch_xla
   import torch_xla.core.xla_model as xm
   import torch_xla.debug.metrics as met

   device = xm.xla_device()
   input1 = torch.randn(2,10).to(device)
   # Defining 2 linear layers
   linear1 = torch.nn.Linear(10,30).to(device)
   linear2 = torch.nn.Linear(30,20).to(device)

   # Running forward
   output1 = linear1(input1)
   output2 = linear2(output1)
   xm.mark_step()
   print(output2)
   print(output1)
   # Printing the metrics to check if compilation and execution occurred
   print(met.metrics_report())

In the printed metrics, the number of compiles and
executions is only 1, even though 2 tensors are printed.
Hence, to avoid multiple graph evaluations, it is recommended that you
visualize tensors after a ``mark_step`` . You can also make use of the
`add_step_closure <https://pytorch.org/xla/release/1.9/index.html#torch_xla.core.xla_model.add_step_closure>`__
api for this purpose. With this api, you pass in the tensors that needs to
be visualized/printed. The added tensors would then be preserved in the
graph and can be printed as part of the callback function passed to the
api. Here is a sample usage:
https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py#L133

**Note:** Graph compilations can take time as the compiler optimizes the graph to run on device.

**Using Eager Debug Mode**
~~~~~~~~~~~~~~~~~~~~~~~~~~

Eager debug mode provides a convenient utility to step through the code and evaluate operators one by one for correctness. Eager debug mode is useful to inspect your models the way you would do in eager-mode frameworks like PyTorch and Tensorflow. With Eager Debug Mode operations are executed eagerly. As soon as an operation is registered with torch-xla, it would be sent for compilation and
execution. Since compiling a single operation, the time spent
would be minimal. Moreover, the chances of hitting the framework compilation cache
increases as models would have repeated operations throughout. 
Consider example 1 below:

.. code:: python

   # Example 1

   import os
   # You need to set this env variable before importing torch-xla
   # to run in eager debug mode.
   os.environ["NEURON_USE_EAGER_DEBUG_MODE"] = "1"

   import torch
   import torch_xla
   import torch_xla.core.xla_model as xm
   import torch_xla.debug.metrics as met

   device = xm.xla_device()
   input1 = torch.randn(2,10).to(device)
   # Defining 2 linear layers
   linear1 = torch.nn.Linear(10,30).to(device)
   linear2 = torch.nn.Linear(30,20).to(device)

   # Running forward
   output1 = linear1(input1)
   output2 = linear2(output1)

   # Printing the metrics to check if compilation and execution occurred
   # Here, in the metrics you should notice that the XRTCompile and XRTExecute
   # value is non-zero, even though no tensor is printed. This is because, each
   # operation is executed eagerly.
   print(met.metrics_report())

   print(output2)
   print(output1)
   # Printing the metrics to check if compilation and execution occurred.
   # Here the XRTCompile count should be same as the previous count.
   # In other words, printing tensors did not incur any extra compile
   # and execution of the graph
   print(met.metrics_report())

As seen from the above scripts, each operator is evaluated eagerly and
there is no extra compilation when output tensors are printed. Moreover, together with
the on-disk Neuron persistent cache, eager debug mode only incurs one time
compilation cost when the ops is first run. When the script is run again, the compiled ops will be
pulled from the persistent cache. Any changes you make to the
training script would result in the re-compilation of only the newly
inserted operations. This is because each operation is compiled
independently. Consider example 2 below:

.. code:: python

   # Example 2

   import os
   # You need to set this env variable before importing torch-xla
   # to run in eager debug mode.
   os.environ["NEURON_USE_EAGER_DEBUG_MODE"] = "1"

   import torch
   import torch_xla
   import torch_xla.core.xla_model as xm
   import torch_xla.debug.metrics as met

   os.environ['NEURON_CC_FLAGS'] = "--log_level=INFO"

   device = xm.xla_device()
   input1 = torch.randn(2,10).to(device)
   # Defining 2 linear layers
   linear1 = torch.nn.Linear(10,30).to(device)
   linear2 = torch.nn.Linear(30,20).to(device)
   linear3 = torch.nn.Linear(20,30).to(device)
   linear4 = torch.nn.Linear(30,20).to(device)

   # Running forward
   output1 = linear1(input1)
   output2 = linear2(output1)
   output3 = linear3(output2)

   # Note the number of compiles at this point and compare
   # with the compiles in the next metrics print
   print(met.metrics_report())

   output4 = linear4(output3)
   print(met.metrics_report())

Running the above example 2 script after running example 1 script, you may notice that from the start until the statement ``output2 = linear2(output1)`` ,
all the graphs would hit the persistent cache. Executing the line
``output3 = linear3(output2)`` would result in a new compilation for ``linear3`` layer only because the layer configuration is new.
Now, when we run
``output4 = linear4(output3)`` , you would observe no new compilation
happens. This is because the graph for ``linear4`` is same as the graph for
``linear2`` and hence the compiled graph for ``linear2`` is reused for ``linear4`` by the framework's internal cache.

Eager debug mode avoids the wait times involved with tensor printing because of larger graph compilation.
It is designed only for debugging purposes, so when the training script is ready, please remove the ``NEURON_USE_EAGER_DEBUG_MODE`` environment
variable from the script in order to obtain optimal performance.

By default, in eager debug mode the
logging level in the Neuron compiler is set to error mode. Hence, no
logs would be generated unless there is an error. Before your first
print, if there are many operations that needs to be compiled, there
might be a small delay. In case you want to check the logs, switch on
the ``INFO`` logs for compiler using:

.. code:: python

   os.environ["NEURON_CC_FLAGS"] = "--log_level=INFO"

**Profiling Model Run**
~~~~~~~~~~~~~~~~~~~~~~~

Profiling model run can help to identify different bottlenecks and
resolve issues faster. You can profile different sections of the code to
see which block is the slowest. To profile model run, you can follow the
steps below:

1. Add: ``import torch_xla.debug.profiler as xp``

2. Start server. This can be done by adding the following line after
   creating xla device: ``server = xp.start_server(9012)``

3. In a separate terminal, start tensorboard. The logdir should be in
   the same directory from which you run the script.

   .. image:: /images/tensorboard.png
      :alt: Image: tensorboard.png

   Open the tensorboard on a browser. Go to profile section in the top
   right. Note: you may have to install the profile plugin using:
   ``pip install tensorboard-plugin-profile``

4. When you click on the profile, it should give an option to capture
   profile. Clicking on capture profile produces the following pop-up.

   .. image:: /images/popup.png
      :alt: Image: popup.png

   In the URL enter: ``localhost:9012`` . Port in this URL should
   be same as the one you gave when starting the server in the script.

5. Once done, click capture and it should automatically load the
   following page:

   .. image:: /images/./tb_1.png
      :alt: Image: tb_1.png

6. To check the profile for different blocks of code, head to
   ``trace_viewer`` under ``Tools`` (on the left column).

   .. image:: /images/./options.png
      :alt: Image: options.png

7. It should show a profile that looks like this:

   .. image:: /images/./profile_large.png
      :alt: Image: profile_large.png

Note: By default, torch-xla would time different blocks of code inside
the library. However, you can also profile block of code in your
scripts. This can be done by adding the code within a ``xp.Trace``
context as follows:

.. code:: python

   ....
   for epoch in range(total_epochs):
       inputs = torch.randn(1,10).to(device)
       labels = torch.tensor([1]).to(device)
       with xp.Trace("model_build"):
           loss = model(inputs, labels)
       with xp.Trace("loss_backward"):
           loss.backward()
   ....

It should produce a profile that has the ``model_build`` and
``loss_backward`` section timed. This way you can time any block of
script for debugging.

.. image:: /images/./profile_zoom.png
   :alt: Image: Screen profile_zoom.png

Note: If you are running your training script in a docker container, to view the
tensorboard, you should launch the docker container using flag: ``--network host``
eg. ``docker run --network host my_image:my_tag``
