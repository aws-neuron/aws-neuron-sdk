.. _neuron-runtime-faq:

NeuronX runtime FAQ
==================

.. contents:: Table of Contents
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

How do I restart/start/stop the NeuronX Runtime?
-----------------------------------------------
Since Neuron Runtime is a library, starting/stopping application would result in starting/stopping the Neuron Runtime.


How do I know which runtimes are associated with which Neuron Device(s)?
------------------------------------------------------------------------
`neuron-ls` and `neuron-top` can be used to find out applications using Neuron Devices.


What about RedHat or other versions of Linux and Windows?
--------------------------------------------------------

We don't officially support it yet.


How can I take advantage of multiple NeuronCores to run multipleinferences in parallel?
---------------------------------------------------------------------------------------

Examples of this for TensorFlow and MXNet are found
:ref:`here <tensorflow-tutorials>` and :ref:`here <mxnet-tutorials>`.
