.. _torch_neuronx_core_placement_guide:

NeuronCore Allocation and Model Placement for Inference (|torch-neuronx|)
=========================================================================

This programming guide describes the how to allocate NeuronCores to processes
and place models onto specific NeuronCores. The models in this guide are
expected to have been traced with with :func:`torch_neuronx.trace`.

.. warning::

    This guide is **not** applicable to NeuronCore placement using XLA
    LazyTensor device execution. See: :ref:`trace-vs-xla-lazytensor`

In order of precedence, the recommendation is to use the following placement
techniques:

1. For nearly all regular models, default core placement should be used to take
   control of all cores for a single process.
2. For applications using multiple processes, default core placement should be
   used in conjunction with ``NEURON_RT_NUM_CORES`` (:ref:`torch_neuronx_placement_default`)
3. For more granular control, then the beta explicit placement APIs may
   be used (:ref:`torch_neuronx_placement_explicit`).

.. contents:: Table of Contents
    :depth: 3

The following guide will assume a machine with 8 NeuronCores:

- NeuronCores will use the notation ``nc0``, ``nc1``, etc.
- Models will use the notation ``m0``, ``m1`` etc.

NeuronCores and  model allocations will be displayed in the following format:

.. raw:: html
    :file: images/0-0-legend-neuronx.svg

The actual cores that are visible to the process can be adjusted according to
the :ref:`nrt-configuration`.

Unlike |torch-neuron| (with |neuron-cc|) instances, |torch-neuronx| (with
|neuronx-cc|) does not support :ref:`neuroncore-pipeline`. This simplifies
model core allocations since it means that model pipelines will likely not span
across multiple NeuronCores.

.. _torch_neuronx_placement_default:

Default Core Allocation & Placement
-----------------------------------

The most basic requirement of an inference application is to be able to place a
single model on a single NeuronCore. More complex applications may use multiple
NeuronCores or even multiple processes each executing different models. The
important thing to note about designing an inference application is that a
single NeuronCore will always be allocated to a single process. *Processes do
not share NeuronCores*. Different configurations can be used to ensure that
an application process has enough NeuronCores allocated to execute its model(s):

- Default: A process will attempt to take ownership of **all NeuronCores**
  visible on the instance. This should be used when an instance is only running
  a single inference process since no other process will be allowed to take
  ownership of any NeuronCores.
- ``NEURON_RT_NUM_CORES``: Specify the **number of NeuronCores** to allocate
  to the process. This places no restrictions on which NeuronCores will be used,
  however, the resulting NeuronCores will always be contiguous. This should be
  used in multi-process applications where each process should only use a subset
  of NeuronCores.
- ``NEURON_RT_VISIBLE_CORES``: Specifies exactly **which NeuronCores** are
  allocated to the process by index. Similar to ``NEURON_RT_NUM_CORES``, this
  can be used in multi-process applications where each process should only use a
  subset of NeuronCores. This provides more fined-grained controls over the
  exact NeuronCores that are allocated to a given process.

See the :ref:`nrt-configuration` for more environment variable details.

Example: Default
^^^^^^^^^^^^^^^^

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuronx

    m0 = torch.jit.load('model.pt')  # Loads to nc0
    m1 = torch.jit.load('model.pt')  # Loads to nc1


.. raw:: html
    :file: images/0-1-default-2.svg

With no environment configuration, the process will take ownership of all
NeuronCores. In this example, only two of the NeuronCores are used by the
process and the remaining are allocated but left idle.


Example: ``NEURON_RT_NUM_CORES``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_NUM_CORES = '2'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuronx

    m0 = torch.jit.load('model.pt')  # Loads to nc0
    m1 = torch.jit.load('model.pt')  # Loads to nc1

.. raw:: html
    :file: images/0-2-default-rt-num-cores.svg

Since there is no other process on the instance, only the first 2 NeuronCores
will be acquired by the process. Models load in a simple linear order to the
least used NeuronCores.


Example: ``NEURON_RT_VISIBLE_CORES``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_VISIBLE_CORES = '4-5'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuronx

    m0 = torch.jit.load('model.pt')  # Loads to nc4
    m1 = torch.jit.load('model.pt')  # Loads to nc5


.. raw:: html
    :file: images/0-3-default-rt-visible-cores.svg

Unlike ``NEURON_RT_NUM_CORES``, setting the visible NeuronCores allows the
process to take control of a specific contiguous set. This allows an application
to have a more fine-grained control of where models will be placed.


Example: Multiple Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_NUM_CORES = '2'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuronx

    m0 = torch.jit.load('model.pt')  # Loads to nc0
    m1 = torch.jit.load('model.pt')  # Loads to nc1


In this example, if the script is run **twice**, the following allocations
will be made:

.. raw:: html
    :file: images/0-5-default-multiprocess.svg

Note that each process will take ownership of as many NeuronCores as is
specified by the ``NEURON_RT_NUM_CORES`` configuration.


.. _torch_neuronx_placement_explicit:

Explicit Core Placement [Beta]
-------------------------------------

The ``torch_neuronx`` framework allows can be found in the
:ref:`torch_neuronx_core_placement_api` documentation.


Example: Manual Core Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most direct usage of the placement APIs is to manually select the
start NeuronCore that each model is loaded to.

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_NUM_CORES = '4'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuronx

    # NOTE: Order of loads does NOT matter
    with torch_neuronx.experimental.neuron_cores_context(start_nc=3):
        m0 = torch.jit.load('model.pt')  # Loads to nc3

    with torch_neuronx.experimental.neuron_cores_context(start_nc=0, nc_count=2):
        m1 = torch.jit.load('model.pt')  # Loads replicas to nc0 and nc1

    example = torch.rand(1, 3, 224, 224)

    m1(example)  # Executes on nc3
    m1(example)  # Executes on nc3

    m0(example)  # Executes on nc0
    m0(example)  # Executes on nc1
    m0(example)  # Executes on nc0


.. raw:: html
    :file: images/8-models-m0-3-m1-1-2.svg


Example: Automatic Multicore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using explicit core placement it is possible to replicate a model to multiple
NeuronCores simultaneously. This means that a single model object within python
can utilize all available NeuronCores (or NeuronCores allocated to the process).

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_NUM_CORES = '8'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuronx

    with torch_neuronx.experimental.multicore_context():
        m0 = torch.jit.load('model.pt')  # Loads replications to nc0-nc7

    example = torch.rand(1, 3, 224, 224)

    m0(example)  # Executes on nc0
    m0(example)  # Executes on nc1

.. raw:: html
    :file: images/6-multicore.svg

To make full use of a model that has been loaded to multiple NeuronCores,
multiple threads should be used to run inferences in parallel.


.. |neuron-cc| replace:: :ref:`neuron-cc <neuron-compiler-cli-reference>`
.. |neuronx-cc| replace:: :ref:`neuronx-cc <neuron-compiler-cli-reference-guide>`
.. |torch-neuron| replace:: :ref:`torch-neuron <inference-torch-neuron>`
.. |torch-neuronx| replace:: :ref:`torch-neuronx <inference-torch-neuronx>`
.. |Inf1| replace:: :ref:`Inf1 <aws-inf1-arch>`
.. |Trn1| replace:: :ref:`Trn1 <aws-trn1-arch>`
