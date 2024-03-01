.. _neuronx_compiler_faq:

Neuron Compiler FAQ (``neuronx-cc``)
====================================

.. contents:: Table of contents
   :local:
   :depth: 1

Where can I compile to Neuron?
---------------------------------

The one-time compilation step from the standard framework-level model to
NEFF binary may be performed on any EC2 instance or even
on-premises.

We recommend using a high-performance compute server of choice (C5 or
z1d instance types), for the fastest compile times and ease of use with
a prebuilt `DLAMI <https://aws.amazon.com/machine-learning/amis/>`__.
Developers can also install Neuron in their own environments; this
approach may work well for example when building a large fleet for
inference, allowing the model creation, training and compilation to be
done in the training fleet, with the NEFF files being distributed by a
configuration management application to the inference fleet.

.. _neuron-vs-neuronx:

What is the difference between ``neuron-cc`` and ``neuronx-cc``?
----------------------------------------------------------------

* ``neuron-cc`` is the Neuron Compiler with TVM front-end, ``neuron-cc`` supports only :ref:`neuroncores-v1-arch`.
* ``neuronx-cc`` is the Neuron Compiler with XLA front-end, ``neuronx-cc`` currently supports 
  :ref:`neuroncores-v2-arch`, ``neuronx-cc`` support of :ref:`neuroncores-v1-arch` is currently a 
  :ref:`Roadmap Item <neuron_roadmap>`.

Should I use ``neuron-cc`` or ``neuronx-cc``?
---------------------------------------------

See :ref:`neuron-vs-neuronx`

My current neural network is based on FP32, how can I use it with Neuron?
-------------------------------------------------------------------------

Developers who want to train their models in FP32 for best accuracy can
compile and deploy them with Neuron. The Neuron compiler automatically converts
FP32 to internally supported datatypes, such as FP16 or BF16.
You can find more details about FP32 data type support
and performance and accuracy tuning
in :ref:`neuronx-cc-training-mixed-precision` or :ref:`neuron-cc-training-mixed-precision`.
The Neuron compiler preserves the application interface - FP32 inputs and outputs.
Transferring such large tensors may become a bottleneck for your application.
Therefore, you can improve execution time by casting the inputs and outputs to
FP16 or BF16 in the ML framework prior to compilation.

Which operators does Neuron support?
---------------------------------------

You can use the ``neuronx-cc list-operators`` command on the cli to list the operators. See :ref:`neuron-compiler-cli-reference-guide`.

To request support for new operators, open an issue on our `GitHub forum <https://github.com/aws/aws-neuron-sdk/issues/new>`_.

Any operators that Neuron Compiler doesn't support?
---------------------------------------------------

Models with control-flow and dynamic shapes are not supported now. You will
need to partition the model using the framework prior to compilation.

.. note::

  Starting with :ref:`neuroncores-v2-arch` Neuron supports control-flow and dynamic shapes.

  Stay tuned and follow the :ref:`Neuron Roadmap <neuron_roadmap>`.

Will I need to recompile again if I updated runtime/driver version?
----------------------------------------------------------------------

The compiler and runtime are committed to maintaining compatibility for
major version releases with each other. The versioning is defined as
major.minor, with compatibility for all versions with the same major
number. If the versions mismatch, an error notification is logged and
the load will fail. This will then require the model to be recompiled.

I have a NEFF binary, how can I tell which compiler version generated it?
-------------------------------------------------------------------------
 ** We will bring a utility out to help with this soon.

How long does it take to compile?
------------------------------------

It depends on the model and its size and complexity, but this generally
takes a few minutes.

Why is my model producing different results compared to CPU/GPU?
----------------------------------------------------------------

:ref:`neuroncores-v2-arch` supports multiple casting modes for floating point numbers, each with
associated implications for performance and accuracy. The default casting mode
is a pragmatic balance between performance and accuracy, however on some models
it may result in loss of precision.

See the :option:`--auto-cast` and :option:`--auto-cast-type` options in :ref:`neuron-compiler-cli-reference-guide` for details on how to adjust the casting mode.

Do you support model *<insert model type>*?
-------------------------------------------

``neuronx-cc`` has explicit support for select model families using the :option:`--model-type` option, though many other model types are supported. You can also inspect supported operators using the :option:`list-operators` sub-command. See th :ref:`neuron-compiler-cli-reference-guide` for details.
More generally, support for new operators and models is continually being added. See our :ref:`neuron_roadmap` for details.
