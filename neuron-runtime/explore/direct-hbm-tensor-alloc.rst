.. _direct-hbm-tensor-alloc:

.. meta::
   :description: Guide on Direct HBM Tensor Allocation with Neuron
   :date_updated: 12/02/2024

Direct HBM Tensor Allocation with Neuron
========================================

This topic provides an overview and usage examples for directly allocating tensors into High Bandwidth Memory (HBM) on AWS Neuron devices using the Neuron Runtime with PyTorch.

Overview
---------

* Device identifier: On Trainium/Inferentia instances, Neuron devices are identified in PyTorch through the names: ``privateuseone`` or ``neuron``. These names can be used interchangeably
* Direct HBM allocation: Allows tensors to be allocated directly into High Bandwidth Memory (HBM) on Neuron devices  
* Performance optimization: Eliminates memory transfer overhead between CPU and device memory

Background
-----------

* PyTorch has many different devices which it dispatches ops (like add, matmul, to) to, ``privateuseone`` is one of these devices, we utilize this and register our backend using this PyTorch interface, and we rename it as ``neuron``. If a tensor is created or moved to a device, PyTorch will dispatch the allocation operation to that device. For instance, if a tensor is created on ``neuron:0`` specifically, the Neuron Runtime will handle the allocation, and will allocate the result on device instead of CPU.

* *Diagram 1: Device registration and allocation flow*

  .. image:: /neuron-runtime/img/device-allocation-flow.png
     :align: center
     :width: 80%

* *Diagram 2: Tensor allocation behaviour*

  .. image:: /neuron-runtime/img/tensor-allocation-behavior.png
     :align: center
     :width: 80%

Device Placement Behavior
--------------------------

Critical Rule
~~~~~~~~~~~~~~

* All-or-nothing: ALL inputs must be on ``neuron:0`` for outputs to remain on device  
* CPU fallback: Any CPU input causes ALL outputs to move to CPU

Why This Matters
~~~~~~~~~~~~~~~~~

* Chained operations: Enables efficient multi-model pipelines without CPU roundtrips  
* Reduced latency: Eliminates expensive device-to-CPU transfers  
* Memory efficiency: Better utilization of 32GB (trn1) / 96GB (trn2) HBM available on Trainium instances

Usage Examples
----------------

Basic Usage - All Inputs on Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    traced_model = '{your-model-here}'
    torch_neuronx.move_trace_to_device(traced_model, 0)

    # Single input
    input_tensor = torch.rand([1, 3, 224, 224], device="neuron:0")
    output = traced_model(input_tensor)
    print(output.device)  # device(type='neuron', index=0)

    # Multiple inputs
    a = torch.rand([2, 2], device="neuron:0")
    b = torch.rand([2, 2], device="neuron:0")
    output = traced_model(a, b)
    print(output.device)  # device(type='neuron', index=0)


Mixed Device Inputs - Shows Fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    a = torch.rand([2, 2], device="neuron:0")
    b = torch.rand([2, 2], device="cpu")  # One CPU tensor
    output = traced_model(a, b)
    print(output.device)  # device(type='cpu') - falls back to CPU


Efficient Model Chaining
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    input_data = torch.rand([1, 256], device="neuron:0")
    intermediate = traced_model1(input_data)    # stays on device
    final_output = traced_model2(intermediate)  # stays on device


Best Practices
----------------

* Keep all tensors on same device: Ensure all inputs are on ``neuron:0`` to avoid CPU fallback  
* Monitor HBM usage: Be aware of HBM limits on Trainium instances (32GB for trn1, 96GB for trn2)
* Verify device placement: Check ``tensor.device`` to confirm expected placement

Compatibility
--------------

* Works with: All ``torch_neuronx.trace`` models, dynamic batching, ``move_trace_to_device``
* Limited by: Available HBM memory
