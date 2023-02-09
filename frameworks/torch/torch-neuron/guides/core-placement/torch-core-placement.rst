.. _torch_neuron_core_placement_guide:

PyTorch Neuron (``torch-neuron``) Core Placement
================================================

This programming guide describes the available techniques and APIs to be able
to allocate NeuronCores to a process and place models onto specific NeuronCores.
In order of precedence, the current recommendation is to use the following
placement techniques:

1. For most regular models, default core placement should be used in
   conjunction with ``NEURON_RT_NUM_CORES`` (:ref:`torch_placement_default`)
2. For more specific core placement for NeuronCore Pipelined models, then
   ``NEURONCORE_GROUP_SIZES`` should be used (:ref:`torch_placement_ncg`).
3. Finally, for even more granular control, then the experimental
   explicit placement APIs may be used (:ref:`torch_placement_explicit`).

.. contents:: Table of Contents
    :depth: 3

The following guide will assume a machine with 8 NeuronCores:

- NeuronCores will use the notation ``nc0``, ``nc1``, etc.
- NeuronCore Groups will use the notation ``ncg0``, ``ncg1`` etc.
- Models will use the notation ``m0``, ``m1`` etc.

NeuronCores, NeuronCore Groups, and model allocations will be displayed in
the following format:

.. raw:: html
    :file: images/0-0-legend.svg

Note that the actual cores that are visible to the process can be adjusted
according to the :ref:`nrt-configuration`.

NeuronCore Pipeline
-------------------

A key concept to understand the intent behind certain core placement strategies
is NeuronCore Pipelining (See :ref:`neuroncore-pipeline`). NeuronCore Pipelining
allows a model to be automatically split into pieces and executed on different
NeuronCores.

For most models only 1 NeuronCore will be required for execution. A model will
**only** require more than one NeuronCore when using NeuronCore Pipeline.
When model pipelining is enabled, the model is split between multiple
NeuronCores and data is transferred between them. For example, if the compiler
flag ``--neuroncore-pipeline-cores 4`` is used, this splits the model into
4 pieces to be executed on 4 separate NeuronCores.

.. _torch_placement_default:

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
- ``NEURONCORE_GROUP_SIZES``: Specifies a number of **NeuronCore Groups** which
  are allocated to the process. This is described in more detail in the
  :ref:`torch_placement_ncg` section.

See the :ref:`nrt-configuration` for more environment variable details.

Example: Default
^^^^^^^^^^^^^^^^

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
    m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1


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
    import torch_neuron

    m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
    m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1

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
    import torch_neuron

    m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc4
    m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc5


.. raw:: html
    :file: images/0-3-default-rt-visible-cores.svg

Unlike ``NEURON_RT_NUM_CORES``, setting the visible NeuronCores allows the
process to take control of a specific contiguous set. This allows an application
to have a more fine-grained control of where models will be placed.


Example: Overlapping Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_VISIBLE_CORES = '0-1'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
    m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1
    m2 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1

.. raw:: html
    :file: images/0-4-default-overlap-model-2.svg

.. raw:: html
    :file: images/0-4-default-overlap.svg

This shows how models may share NeuronCores but the default model placement
will attempt to evenly distribute NeuronCore usage rather than overlapping all
models on a single NeuronCore.


Example: Multiple Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_NUM_CORES = '2'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
    m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1


In this example, if the script is run **twice**, the following allocations
will be made:

.. raw:: html
    :file: images/0-5-default-multiprocess.svg

Note that each process will take ownership of as many NeuronCores as is
specified by the ``NEURON_RT_NUM_CORES`` configuration.


.. _torch_placement_ncg:

NEURONCORE_GROUP_SIZES
----------------------

.. important::

    The use of explicit core placement should only be used when a specific
    performance goal is required. By default ``torch-neuron`` places models on
    the **least used** NeuronCores. This should be optimal for most
    applications.

    Secondly, ``NEURONCORE_GROUP_SIZES`` is being deprecated in a future
    release and should be avoided in favor of newer placement methods.
    Use ``NEURON_RT_NUM_CORES`` or ``NEURON_RT_VISIBLE_CORES`` with default
    placement if possible (See :ref:`torch_placement_default`)


In the current release of NeuronSDK, the most well-supported method of placing
models onto specific NeuronCores is to use the ``NEURONCORE_GROUP_SIZES``
environment variable. This will define a set of "NeuronCore Groups" for the
application process.

NeuronCore Groups are *contiguous sets of NeuronCores* that are allocated to
a given process. Creating groups allows an application to ensure that a
model has a defined set of NeuronCores that will always be allocated to it.

Note that NeuronCore Groups *can* be used to allocate non-pipelined models
(those requiring exactly 1 NeuronCore) to specific NeuronCores but this is
not the primary intended use. The intended use of NeuronCore Groups is to
ensure pipelined models (those requiring >1 NeuronCore) have exclusive access
to a specific set of contiguous NeuronCores.

In the cases where models are being used *without* NeuronCore Pipeline, the
general recommendation is to use default placement
(See :ref:`torch_placement_default`).

The following section demonstrates how ``NEURONCORE_GROUP_SIZES`` can be used
and the issues that may arise.

Example: Single NeuronCore Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example where one model requires 4 NeuronCores, the correct environment
configuration would be:

**Environment Setup**:

.. code-block:: bash

    export NEURONCORE_GROUP_SIZES = '4'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    m0 = torch.jit.load('model-with-4-neuron-pipeline-cores.pt')  # Loads to nc0-nc3


.. raw:: html
    :file: images/1-ncg-4.svg

This is the most basic usage of a NeuronCore Group. The environment setup
causes the process to take control of 4 NeuronCores and then the script loads
a model compiled with a NeuronCore Pipeline size of 4 to the first group.


Example: Multiple NeuronCore Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With more complicated configurations, the intended use of
``NEURONCORE_GROUP_SIZES`` is to create 1 Group per model with the correct size
to ensure that the models are placed on the intended NeuronCores. Similarly, the
environment would need to be configured to create a NeuronCore Group for each
model:

**Environment Setup**:

.. code-block:: bash

    export NEURONCORE_GROUP_SIZES = '3,4,1'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    m0 = torch.jit.load('model-with-3-neuron-pipeline-cores.pt')  # Loads to nc0-nc2
    m1 = torch.jit.load('model-with-4-neuron-pipeline-cores.pt')  # Loads to nc3-nc6
    m2 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc7




.. raw:: html
    :file: images/2-ncg-3-4-1.svg


Issue: Overlapping Models with Differing Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When multiple models are loaded to a single NeuronCore Group, this can cause
unintended inefficiencies. A single model is only intended to span a single
NeuronCore Group. Applications with many models of varying sizes can be
restricted by NeuronCore Group configurations since the most optimal model
layout may require more fine-grained controls.

**Environment Setup**:

.. code-block:: bash

    export NEURONCORE_GROUP_SIZES = '2,2'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1
    m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc2-nc3
    m2 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
    m3 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc2
    m4 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0


.. raw:: html
    :file: images/3-models-m4-0-warning.svg

.. raw:: html
    :file: images/3-models-m2-0-m3-2.svg

.. raw:: html
    :file: images/3-ncg-2-2.svg


Here the ``NEURONCORE_GROUP_SIZES`` does not generate an optimal layout
because placement strictly follows the layout of NeuronCore Groups. A
potentially more optimal layout would be to place ``m4`` onto ``nc1``. In this
case, since a pipelined model will not be able to have exclusive access to a set
of NeuronCores, the default NeuronCore placement (no NeuronCore Groups
specified) would more evenly distribute the models.

Also note here that this is an example of where the order of model loads
affects which model is assigned to which NeuronCore Group. If the order of the
load statements is changed, models may be assigned to different NeuronCore
Groups.


Issue: Incompatible Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another problem occurs when attempting to place a model which does not evenly
fit into a single group:

**Environment Setup**:

.. code-block:: bash

    export NEURONCORE_GROUP_SIZES = '2,2'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1
    m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc2-nc3
    m2 = torch.jit.load('model-with-3-neuron-pipeline-cores.pt')  # Loads to nc0-nc2


.. raw:: html
    :file: images/4-models-m2-0-2-warning.svg

.. raw:: html
    :file: images/3-ncg-2-2.svg


The model will be placed *across* NeuronCore Groups since there is no obvious
group to assign the model to according to the environment variable
configuration. Depending on the individual model and application requirements,
the placement here may not be optimal.


Issue: Multiple Model Copies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is common in inference serving applications to use multiple replicas of a
single model across different NeuronCores. This allows the hardware to be fully
utilized to maximize throughput. In this scenario, when using NeuronCore
Groups, the only way to replicate a model on multiple NeuronCores is to create a
*new model* object. In the example below, 4 models loads are performed to place
a model in each NeuronCore Group.

**Environment Setup**:

.. code-block:: bash

    export NEURONCORE_GROUP_SIZES = '2,2,2,2'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    models = list()
    for _ in range(4):
        model = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')
        models.append(model)


.. raw:: html
    :file: images/3-ncg-2-2-2-2-copies.svg


The largest consequence of this type of model allocation is that the application
code is responsible for routing inference requests to models. There are a
variety of ways to implement the inference switching but in all cases routing
logic needs to be implemented in the application code.


Issue Summary
^^^^^^^^^^^^^

The use of ``NEURONCORE_GROUP_SIZES`` has the following problems:

- **Variable Sized Models**: Models which require crossing NeuronCore Group
  boundaries may be placed poorly. This means group configuration limits the
  size of which models can be loaded.
- **Model Load Order**: Models are loaded to NeuronCore Groups greedily. This
  means that the order of model loads can potentially negatively affect
  application performance by causing unintentional overlap.
- **Implicit Placement**: NeuronCore Groups cannot be explicitly chosen in the
  application code.
- **Manual Replication**: When loading multiple copies of a model to different
  NeuronCore Groups, this requires that multiple model handles are used.


.. _torch_placement_explicit:

Experimental: Explicit Core Placement
-------------------------------------

To address the limitations of ``NEURONCORE_GROUP_SIZES``, a new set of APIs has
been added which allows specific NeuronCores to be chosen by the application
code. These can be found in the :ref:`torch_neuron_core_placement_api` documentation.


Example: Manual Core Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most direct usage of the placement APIs is to manually select the
start NeuronCore that each model is loaded to. This will automatically use as
many NeuronCores as is necessary for that model (1 for most models, >1 for
NeuronCore Pipelines models).

**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_NUM_CORES = '4'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    # NOTE: Order of loads does NOT matter

    with torch_neuron.experimental.neuron_cores_context(2):
        m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc2-nc3

    with torch_neuron.experimental.neuron_cores_context(0):
        m2 = torch.jit.load('model-with-3-neuron-pipeline-cores.pt')  # Loads to nc0-nc2

    with torch_neuron.experimental.neuron_cores_context(0):
        m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1

    with torch_neuron.experimental.neuron_cores_context(3):
        m3 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc3


.. raw:: html
    :file: images/5-models-m2-0-2-m3-3.svg

.. raw:: html
    :file: images/5-placement.svg


Note that this directly solves the ``NEURONCORE_GROUP_SIZES`` issues of:

- **Variable Sized Models**: Now since models are directly placed on the
  NeuronCores requested by the application, there is no disconnect
  between the model sizes and NeuronCore Group sizes.
- **Model Load Order**: Since the NeuronCores are explicitly selected, there is
  no need to be careful about the order in which models are loaded since they
  can be placed deterministically regardless of the load order.
- **Implicit Placement**: Similarly, explicit placement means there is no chance
  that a model will end up being allocated to an incorrect NeuronCore Group.


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
    import torch_neuron

    with torch_neuron.experimental.multicore_context():
        m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads replications to nc0-nc7


.. raw:: html
    :file: images/6-multicore.svg


This addresses the last ``NEURONCORE_GROUP_SIZES`` issue of:

- **Manual Replication**: Since models can be automatically replicated to
  multiple NeuronCores, this means that applications no longer need to implement
  routing logic and perform multiple loads.

This API has a secondary benefit that the exact same loading logic can be used
on an ``inf1.xlarge`` or an ``inf1.6xlarge``. In either case, it will use all
of the NeuronCores that are visible to the process. This means that no special
logic needs to be coded for different instance types.


Example: Explicit Replication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replication is also possible with the
:func:`~torch_neuron.experimental.neuron_cores_context` API. The number of
replications is chosen by ``replications = floor(nc_count / cores_per_model)``.


**Environment Setup**:

.. code-block:: bash

    export NEURON_RT_NUM_CORES = '8'

**Python Script**:

.. code-block:: python

    import torch
    import torch_neuron

    with torch_neuron.experimental.neuron_cores_context(start_nc=2, nc_count=4):
        m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads replications to nc2-nc5


.. raw:: html
    :file: images/7-replication.svg
