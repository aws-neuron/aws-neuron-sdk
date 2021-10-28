Neuron runtime FAQ
==================

.. contents::
   :local:
   :depth: 1


Where can I find information about Neuron Runtime 2.x (``libnrt.so``)
---------------------------------------------------------------------

See :ref:`introduce-libnrt` for detailed information about Neuron Runtime 2.x (``libnrt.so``).

What will happen if I will upgrade Neuron Framework without upgrading latest kernel mode driver?
------------------------------------------------------------------------------------------------

Application start would fail with the following error message:
.. code:: bash

    2021-Aug-11 19:18:21.0661 24616:24616 ERROR   NRT:nrt_init      This runtime requires Neuron Driver version 2.0 or greater. Please upgrade aws-neuron-dkms package.


Do I need to recompile my model to use the Runtime Library?
-----------------------------------------------------------
No. Runtime 2.x supports all the models compiled with Neuron Compiler 1.x.


Do I need to change my application launch command?
--------------------------------------------------
No.

How do I restart/start/stop the Neuron Runtime?
-----------------------------------------------
Since Neuron Runtime is a library, starting/stopping application would result in starting/stopping the Neuron Runtime.


How do I know which runtimes are associated with which Neuron Device(s)?
------------------------------------------------------------------------
`neuron-ls` and `neuron-top` can be used to find out applications using Neuron Devices.


What about RedHat or other versions of Linux and Windows?
--------------------------------------------------------

We dont officially support it yet.

How can I use Neuron in a container based environment? Does Neuron work with ECS and EKS?
-----------------------------------------------------------------------------------------

ECS and EKS support is coming soon. Containers can be configured as
shown :ref:`here <neuron-containers>`.

How can I take advantage of multiple NeuronCores to run multipleinferences in parallel?
---------------------------------------------------------------------------------------

Examples of this for TensorFlow are found
:ref:`here <tensorflow-tutorials>`
:ref:`here <mxnet-tutorials>`