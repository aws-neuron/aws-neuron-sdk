.. _neuronx-plugin-tensorboard:

Neuron Plugin for TensorBoard (Trn1)
====================================

.. contents:: Table of Contents
  :local:
  :depth: 2


Overview
--------

This guide is for developers who want to better understand how their
model is executed using Neuron SDK through TensorBoard.

The Neuron plugin for TensorBoard provides metrics to the performance of machine learning tasks accelerated using the Neuron SDK. It is
compatible with TensorBoard versions 1.15 and higher. It provides visualizations and profiling results for graphs executed on NeuronCores.

.. note::

    The following information is compatible with Neuron SDK for Trn1.  For a walkthrough on Inf1, please check out the guide
    :ref:`neuron-plugin-tensorboard`.


Enable profiling on Trn1
------------------------

.. note::

   Profiling is currently only supported with PyTorch Neuron (``torch-neuronx``).

Please refer to the following guides:

- PyTorch-Neuron
    - :ref:`torch-neuronx-profiling-with-tb`


Launch TensorBoard
------------------

In this step, we will process the Neuron profile data and launch TensorBoard.

1. Install the Neuron plugin for Tensorboard on your EC2 instance.

.. code:: bash

    python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"

    pip install tensorboard-plugin-neuronx

.. note::

   If using TensorBoard >= 2.5, please use the ``--load_fast=false`` option when launching.
   ``tensorboard --logdir results --load_fast=false``

2. After you see the following message, TensorBoard is ready to use.  By default,
TensorBoard will be launched at ``localhost:6006``.

::

   ...
   Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
   TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)


View results in TensorBoard
---------------------------

In this step, we will view the Neuron plugin for TensorBoard from a browser on your local
development machine.

1. Connect to the EC2 instance where TensorBoard is running while enabling port forwarding.
In this example, we assume TensorBoard has been launched using the default address ``localhost:6006``.

.. code:: bash

   # if Ubuntu-based AMI
   ssh -i <PEM key file> ubuntu@<instance DNS> -L 6006:localhost:6006

   # if AL2-based AMI
   ssh -i <PEM key file> ec2-user@<instance DNS> -L 6006:localhost:6006

2. In a browser, visit |tensorboard_address|.

3. In the top navigation bar, switch from ``Graphs`` to ``Neuron``.  If it does not show up,
please wait a while and refresh the page while the plugin loads.  If the issue persists, check
the ``Inactive`` dropdown list on the right and check for ``Neuron``.

|image1|

4. If TensorBoard failed to find the generated logs, you will see the following message:

|image2|


In this case, please make sure the version of the ``aws-neuronx-tools``
package and the Neuron framework package is from Neuron release 2.6 or newer.


Neuron Trace View
-----------------

|image3|

The trace view gives a high level timeline of execution by aligning Neuron events, such as Neuron Device execution,
data transfers, and Collective Compute synchronization (if applicable), with other events from the XLA profiler.

Use this view to better understand bottlenecks during the run, and potentially experiment with how execution changes
by moving the ``mark_step()`` call which will execute the graph.


Neuron Operator View
--------------------

|image4|

The operator view can show timing information for both the framework operators and HLO operators by selecting
the ``operator-framework`` and ``operator-hlo`` tools respectively.  The pie charts show breakdowns of the time taken
by device, as well as per operator on a single device.  The table below lists out the operators and can be sorted by clicking
on the columnn headers.  For fused operations, hover over the ``?`` to see which operators are being executed.

For a quick glance at the most time consuming operators, click the ``Time %`` column in the table to sort by the relative
time spent on this type of operation compared to the rest of the model.


Neuron Operator Timeline View
-----------------------------

|image5|

The operator timeline view is a detailed look into a single execution with Neuron.  A high level overview at the top breaks
down the execution into categories, including Neuron Runtime setup time, as well as NeuronCore compute engine and DMA engine busyness.
Activity on the compute and DMA engines are further categorized as compute, control, and data transfer intervals which are
shown as separate processes, with each showing a hierarchical view of the framework operators and their corresponding
HLO operation.  The fused operations can be a result of compiler optimizations or are operations that are running in
parallel on the device.  Each bar can be clicked to show information regarding which operators are overlapped.

This view can give better insight into how operators translate to Neuron, as well as how certain Neuron compiler options
may improve performance.



.. |image1| image:: /images/Neuron_Profiler_Tensorboard_Dropdown.jpg
.. |image2| image:: /images/tb-plugin-img12.png
  :height: 2914
  :width: 5344
  :scale: 10%
.. |image3| image:: /images/Neuron_Profiler_Runtime_Trace_Original.jpg
.. |image4| image:: /images/Neuron_Profiler_T1_Op_Framework_View.png
.. |image5| image:: /images/TB_Operator_Timeline_2-10.png
.. |tensorboard_address| raw:: html

   <a href="http://localhost:6006" target="_blank">localhost:6006</a>
