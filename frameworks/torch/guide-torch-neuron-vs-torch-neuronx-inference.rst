.. _torch-neuron_vs_torch-neuronx:

Comparison of |torch-neuron| (|Inf1|) versus |torch-neuronx| (|Inf2| & |Trn1|) for Inference
============================================================================================

Neuron now supports multiple instance types for inference. The choice of
instance should be motivated primarily by the performance needs of the
application, the instance pricing, and model compatibility.

In prior releases, |torch-neuron| *only supported inference* and
|torch-neuronx| *only supported training*. While |torch-neuron| will never
be updated to support training, |torch-neuronx| now supports both *inference and
training*.

.. note::

    **Recommendation**: Continue using |torch-neuron| (|Inf1|) for existing
    inference applications.

    |torch-neuronx| (|Inf2| & |Trn1|) should be used for inference applications that
    require very low latency, distributed inference, and large models that would
    not otherwise work with |Inf1|. See: :ref:`benchmark`.


Framework Comparison
--------------------

Example
~~~~~~~

The following scripts are identical model compilations performed using each
framework. The lines that are changed are highlighted to show where the
differences occur.


.. tab-set::

    .. tab-item:: torch-neuron

        .. code-block:: python
            :emphasize-lines: 3, 8

            import torch
            import torchvision
            import torch_neuron

            model = torchvision.models.resnet50(pretrained=True).eval()
            image = torch.rand(1, 3, 224, 224)

            trace = torch_neuron.trace(model, image)

    .. tab-item:: torch-neuronx

        .. code-block:: python
            :emphasize-lines: 3, 8

            import torch
            import torchvision
            import torch_neuronx

            model = torchvision.models.resnet50(pretrained=True).eval()
            image = torch.rand(1, 3, 224, 224)

            trace = torch_neuronx.trace(model, image)


Hardware Features
~~~~~~~~~~~~~~~~~

The |torch-neuron| framework supports |Inf1| instances and the |torch-neuronx|
framework supports |Inf2| & |Trn1| instances. These instances have different
|architectures|, networking configurations, and capabilities due to the
NeuronCore versions used.

Models compiled with |torch-neuron| produce artifacts which are *only*
compatible with |NeuronCore-v1|. Models compiled with |torch-neuronx| produce
artifacts which are *only* compatible with |NeuronCore-v2|. This also means
that models that were previously compiled with |torch-neuron| for |Inf1| are
not forwards compatible with |Inf2| & |Trn1| instances. Likewise, models compiled
with |torch-neuronx| for |Inf2| & |Trn1| are not backwards compatible with |Inf1|.

|NeuronCore-v2| is capable of higher throughput and lower latency than
|NeuronCore-v1| due to more powerful compute engines and improved memory
bandwidth. |NeuronCore-v2| can also support larger models since more
memory is available per NeuronCore. The hardware differences between
NeuronCore versions means that models compiled with |torch-neuronx| will
usually outperform models compiled with |torch-neuron|.

In cases where throughput may be similar across instance-types, instances using
|NeuronCore-v2| tend to achieve *significantly lower* latency than instances
using |NeuronCore-v1|. This can enable applications that require extremely fast
response time.

See the :ref:`benchmark` page for the most up-to-date performance metrics.

Besides performance benefits, |NeuronCore-v2| also has more hardware
capabilities compared to |NeuronCore-v1|. For example, |NeuronCore-v2|
supports a greater variety of data types and introduces a new fully programmable
GPSIMD-Engine.

Note that ``Trn`` instance-types are optimized for training purposes. Some
``Trn`` features (such as inter-chip networking) may be unnecessary
for inference applications that do not require distribution across multiple
NeuronCores.


Software Features
~~~~~~~~~~~~~~~~~

The |torch-neuron| framework uses :func:`torch_neuron.trace` to
create a TensorFlow GraphDef protobuf intermediate representation (IR) of the
model compute graph. This is compiled to a binary Neuron Executable File Format
(NEFF) with the |neuron-cc| compiler.

The |torch-neuronx| framework uses :func:`torch_neuronx.trace` with
torch-xla_ to create a HloModule protobuf IR of the model compute graph. This is
compiled to a binary executable NEFF with the |neuronx-cc| compiler.

The use of different compiler versions means that separate flags are supported
by each framework. For example:

- :ref:`neuroncore-pipeline` is supported in |neuron-cc| but is not supported
  in |neuronx-cc|. However, this feature is much less useful when using the
  |NeuronCore-v2| architecture due to significant memory improvements.
- Mixed precision flags will differ across the compilers. |neuronx-cc| improves
  the flags by making the behavior more explicit and streamlined:

  - |neuron-cc-mixed-precision|
  - |neuronx-cc-mixed-precision|

Since the python graph recording methods used by the frameworks are much
different, this may lead to different levels of model support. To view the
models which are known to be working, many compilation samples are provided for
each framework:

- `torch-neuron Samples`_
- `torch-neuronx Samples`_

Framework model support may also be affected by the graph partitioning feature.
In |torch-neuron|, the :func:`torch_neuron.trace` API provides the ability to
fall back to CPU for operations that are not supported directly by Neuron. This
fallback behavior is currently not supported by :func:`torch_neuronx.trace`,
however, certain operations that were previously not well-supported
in |torch-neuron| are now supported in |torch-neuronx| by default (e.g.
:class:`torch.nn.Embedding`).


Feature Summary
~~~~~~~~~~~~~~~

+-----------------------+-----------------------------+-----------------------------+
|                       | `torch-neuron`              | `torch-neuronx`             |
+=======================+=============================+=============================+
| Supported Instances   | |Inf1|                      | |Inf2| & |Trn1|             |
+-----------------------+-----------------------------+-----------------------------+
| Inference Support     | Yes                         | Yes                         |
+-----------------------+-----------------------------+-----------------------------+
| Training Support      | No                          | Yes                         |
+-----------------------+-----------------------------+-----------------------------+
| Architecture          | |NeuronCore-v1|             | |NeuronCore-v2|             |
+-----------------------+-----------------------------+-----------------------------+
| Model Support         | |model-support-v1|          | |model-support-v2|          |
+-----------------------+-----------------------------+-----------------------------+
| Trace API             | :func:`torch_neuron.trace`  | :func:`torch_neuronx.trace` |
+-----------------------+-----------------------------+-----------------------------+
| NeuronCore Pipeline   | Yes                         | No                          |
+-----------------------+-----------------------------+-----------------------------+
| Partitioning          | Yes                         | No                          |
+-----------------------+-----------------------------+-----------------------------+
| IR                    | GraphDef                    | HLO                         |
+-----------------------+-----------------------------+-----------------------------+
| Compiler              | |neuron-cc|                 | |neuronx-cc|                |
+-----------------------+-----------------------------+-----------------------------+
| Samples               | `torch-neuron Samples`_     | `torch-neuronx Samples`_    |
+-----------------------+-----------------------------+-----------------------------+


References
----------

To determine if a model is already supported in a given framework, it is
recommended to check the existing documentation for specific models. In order
of reference quality, the following pages can be checked prior to compiling a
model:

1. :ref:`benchmark`: Models that are available here have been optimized to
   maximize throughput and/or latency. These metrics are updated frequently as
   improvements are made. Since metrics are published for different instance
   types, this can provide a direct performance comparison between instances.
   Note that the exact models and configurations may differ across instances.
2. `Neuron GitHub Samples`_: Provides simple examples of compiling and executing
   models. Compared to the benchmarks, this reference is only
   intended to show *how* to run a particular model on Neuron. This only
   validates if a framework supports a given model.
3. :ref:`model_architecture_fit`: If the a model is not listed on the prior
   pages, it may be that the model has not been tested or may not be
   well-supported. The architecture fit page provides high-level guidelines for
   which kinds of models will work well based on the hardware capabilities.

If a model does not appear in any of these references, the last option is
to attempt to compile the model to see how it performs. In the case that an
error occurs during compilation, please file a ticket in the
`Neuron SDK Github Issues`_.


.. |neuron-cc-mixed-precision| replace:: :ref:`neuron-cc-training-mixed-precision`
.. |neuronx-cc-mixed-precision| replace:: :ref:`neuronx-cc-training-mixed-precision`
.. |Inf1| replace:: :ref:`Inf1 <aws-inf1-arch>`
.. |Trn1| replace:: :ref:`Trn1 <aws-trn1-arch>`
.. |Inf2| replace:: :ref:`Inf2 <aws-inf2-arch>`
.. |architectures| replace:: :ref:`architectures <neuroncores-arch>`
.. |NeuronCore-v1| replace:: :ref:`NeuronCore-v1 <neuroncores-v1-arch>`
.. |NeuronCore-v2| replace:: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`
.. |neuron-cc| replace:: :ref:`neuron-cc <neuron-compiler-cli-reference>`
.. |neuronx-cc| replace:: :ref:`neuronx-cc <neuron-compiler-cli-reference-guide>`
.. |torch-neuron| replace:: :ref:`torch-neuron <inference-torch-neuron>`
.. |torch-neuronx| replace:: :ref:`torch-neuronx <inference-torch-neuronx>`
.. |model-support-v1| replace:: :ref:`Architecture Fit NeuronCore-v1 <model-architecture-fit-neuroncore-v1>`
.. |model-support-v2| replace:: :ref:`Architecture Fit NeuronCore-v2 <model-architecture-fit-neuroncore-v2>`

.. _Neuron GitHub Samples: https://github.com/aws-neuron/aws-neuron-samples
.. _torch-neuron Samples: https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuron
.. _torch-neuronx Samples: https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx
.. _torch-xla: https://github.com/pytorch/xla
.. _Neuron SDK Github Issues: https://github.com/aws-neuron/aws-neuron-sdk/issues