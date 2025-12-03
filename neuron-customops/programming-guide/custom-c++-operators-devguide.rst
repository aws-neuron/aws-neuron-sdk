.. _feature-custom-operators-devguide:

Neuron Custom C++ Operators Developer Guide [Beta]
==================================================

This document gives an overview of the Neuron Custom C++ Operator feature and APIs . Currently, CustomOp support is limited to the PyTorch framework.  

Please refer to the following documents for further information regarding Neuron Custom C++ Operators:

* :ref:`neuronx-customop-mlp-tutorial`
* :ref:`neuronx-customop-mlp-perf`
* :ref:`custom-ops-api-ref-guide`

.. contents:: Table of contents
   :local:
   :depth: 1

Setup & Installation
--------------------

.. note::
   The name of ``aws-neuronx-gpsimd-customop`` has been changed to ``aws-neuronx-gpsimd-customop-lib`` as of the neuron 2.10 release.

We provide tooling and library packages (RPM and DEB) that can be installed on TRN1 and INF2 instances:
::

   aws-neuronx-gpsimd-tools-0.3
   aws-neuronx-gpsimd-customop-lib-0.3

For AL2023 only, the following packages need be installed as dependencies:

::
   sudo dnf install libnsl
   sudo dnf install libxcrypt-compat

On AL2023, they can be installed with the following commands:

::
   sudo dnf remove python3-devel -y
   sudo dnf remove aws-neuronx-gpsimd-tools-0.* -y
   sudo dnf remove aws-neuronx-gpsimd-customop-lib-0.* -y
   
   sudo dnf install python3-devel -y
   sudo dnf install aws-neuronx-gpsimd-tools-0.* -y 
   sudo dnf install aws-neuronx-gpsimd-customop-lib-0.* -y

On Ubuntu, they can be installed with the following commands:

::
   sudo apt-get remove python3-dev -y
   sudo apt-get remove aws-neuronx-gpsimd-tools=0.* -y
   sudo apt-get remove aws-neuronx-gpsimd-customop-lib=0.* -y  
   
   sudo apt-get install python3-dev -y
   sudo apt-get install aws-neuronx-gpsimd-tools=0.* -y
   sudo apt-get install aws-neuronx-gpsimd-customop-lib=0.* -y 


Implementing an operator in C++
-------------------------------

Custom operators require a function that defines the custom computation. We define this as the **kernel function**. Neuron Custom C++ Operators also contain a **shape function** separate from the normal compute code. This *shape function* defines the shapes of output tensors for a given set of inputs to the operator. This is needed because PyTorch Neuron (torch-neuronx) is based on the PyTorch/XLA software package and uses a Just-In-Time (JIT) compilation strategy. At runtime the operators in the model will be compiled into a binary to be executed on the NeuronCore. During compilation the shapes of the input and output tensors to operators are computed. The **shape function** is executed on the host, whereas the **kernel function** is executed on the NeuronCore. 

Kernel Function
^^^^^^^^^^^^^^^

The kernel function contains the C++ implementation of the CustomOp, as shown in the example below.  By including torch.h in the source, the developer has access to a NeuronCore-ported subset of the torch C++ api  (https://pytorch.org/cppdocs/).  The port contains everything required for CustomOp development and model integration, specifically Tensor and Scalar classes in c10, and a subset of aTen operators.
::

   #include <stdint.h>
   #include <stdlib.h>
   #include <torch/torch.h>

   torch::Tensor tensor_negate_compute(const torch::Tensor& t_in) {
      size_t num_elem = t_in.numel();
      torch::Tensor t_out = torch::zeros({num_elem}, torch::kFloat);

      auto t_in_acc = t_in.accessor<float, 1>();
      auto t_out_acc = t_out.accessor<float, 1>();
      for (size_t i = 0; i < num_elem; i++) {
         t_out_acc[i] = -1 * t_in_acc[i];
      }
      return t_out;
   }

The kernel function is the main computational code for the operator. We support a subset of the input types usable by regular PyTorch Custom Operators: ``torch::Tensor``, ``torch::Scalar``, ``double``, and ``int64_t``. However we do not support ``std::vector`` or ``std::tuple`` of these types at this time. When passing in scalars, ``double`` is the only supported dtype, no other integral types such as ``int``, ``short``, ``int64_t`` or ``long`` are supported. The return value must be a ``torch::Tensor``.

.. warning::
   Tensors passed into and returned from CustomOp functions can either have up to 8 dimensions where the maximum size of each dimension is 65535, or up to 4 dimensions where the maximum size of each dimension is 4294967295.

The body of the kernel function may exercise C/C++ libraries, ``torch::Tensor`` classes, and select aTen operators, as is customary for Torch programming.  For high performance, feature offerings provide faster memory access, via new Tensor Accessor classes and stack management compiler flags. Additionally, higher performance can be obtained by parallelizing execution of the kernel over multiple GPSIMD cores. See the :ref:`custom-ops-api-ref-guide` for more details.

Finally, because the kernel is specially compiled for and run by the NeuronCore target, its tooling, libraries, and environment differ from the host pytorch installation. For example, while the host may run Pytorch 1.13 and a C++17 compatible compiler in a linux environment, the NeuronCore may run a port of Pytorch 1.12 (c10) and LLVM’s libc++ C++14 version 10.0.1 without linux.  Developers must develop for the compiler, torch version, and environment of their targeted NeuronCore.  See the :ref:`custom-ops-api-ref-guide` for more details.


Shape Function
^^^^^^^^^^^^^^

The shape function has the same function signature as the kernel function, but does not perform any computations. Rather, it only defines the shape of the output tensor but not the actual values. 
::

   #include <stdint.h>
   #include <stdlib.h>
   #include <torch/torch.h>

   torch::Tensor tensor_negate_shape(torch::Tensor t1) {
      size_t num_elem = t1.numel();
      torch::Tensor t_out = torch::zeros({num_elem}, torch::kFloat);

      return t_out;
   }

The body of the shape function may exercise C/C++ libraries or ``torch::Tensor`` classes. The body may not access the data of input tensors since these are XLA Tensors and do not have any data storage allocated yet. However, any of the functions that access shape information such as *numel* (to get the number of elements) may be used. 


Building and executing operators
--------------------------------

Once you have the kernel and shape functions for your operators you can build them into a library to use them from PyTorch in your model. Just like regular PyTorch Custom Operators, Neuron Custom C++ Operators use a registration macro to associate the kernel and shape functions with the name of the operator that will be called from Python.

Similar to PyTorch, Neuron Custom C++ Operators are grouped into libraries defined within the ``NEURON_LIBRARY(<lib_name>, m)`` scope, where lib_name is the name of your library of custom operators. Within this scope, calls to ``m.def(<op_name>, <shape_fcn>, <kernel_fcn>)`` define each operator in your library. The ``op_name`` is the name to call the operator with in the model (i.e. ``torch.ops.lib_name.op_name()``). The ``shape_fcn`` is a function pointer to the shape function to call during compilation. Finally the ``kernel_fcn`` is the name of the function to be executed on the NeuronCore at runtime. 
::

   #include <stdint.h>
   #include <stdlib.h>
   #include <torch/torch.h>
   #include "torchneuron/register.h"

   torch::Tensor tensor_negate_shape(torch::Tensor t1) {
      size_t num_elem = t1.numel();
      torch::Tensor t_out = torch::zeros({num_elem}, torch::kFloat);

      return t_out;
   }

   NEURON_LIBRARY(my_ops, m) {
      m.def("tensor_negate", &tensor_negate_shape, "tensor_negate_compute");
   }

Notice that the ``NEURON_LIBRARY`` macro is used in the same C++ file as the shape function. This is because the registration is loaded on the host.

.. warning::
   Each model can only have one CustomOp library, and the library can have 10 functions registered. However, models using ``torch.sort`` cannot have any CustomOps.

The custom op library is built by calling the ``load`` API in Python like:
::

   import torch_neuronx
   from torch_neuronx.xla_impl import custom_op

   custom_op.load(
      name='my_ops',
      compute_srcs=['kernel.cpp'],
      shape_srcs=['shape.cpp'],
      multicore=False
   )

In the example above, name refers to the name of the library file to be created (i.e. ``libmy_ops.so``) and the ``compute_srcs`` and ``shape_srcs`` are lists of files to be compiled. After the ``load`` API completes, the library will have been compiled and loaded into the current PyTorch process. 

.. warning::
   The library file name should not be "builtin" as it is a reserved keyword.

CustomOp also supports multicore execution mode. If you want to the library to run in multicore mode, pass the flag ``multicore=True`` into the ``load`` API. Notice that the execution mode is specified at the library level, so all the functions in the library run in the same mode. For more details of multicore CustomOp, please refer to `Using multiple GPSIMD cores` section in :ref:`custom-ops-api-ref-guide`.

Similar to PyTorch, the Neuron custom op will be available at ``torch.ops.<lib_name>.<op_name>`` where ``lib_name`` is defined in the ``NEURON_LIBRARY`` macro, and ``op_name`` is defined in the call to ``m.def``.
::

   import torch

   out_tensor = torch.ops.my_ops.tensor_negate(in_tensor)


Loading a previously built library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library can also be built ahead of time or in a separate process and loaded later. In the ``load`` API, specify the ``build_directory`` argument and the library will be written to that location on disk.
::

   import torch_neuronx
   from torch_neuronx.xla_impl import custom_op

   custom_op.load(
      name='my_ops',
      compute_srcs=['kernel.cpp'],
      shape_srcs=['shape.cpp'],
      build_directory*=*os.getcwd(),
   )

Then, later, this library can be loaded by calling the ``load_library`` API and using the ops in the exact same way.
::

   import torch
   import torch_neuronx
   from torch_neuronx.xla_impl import custom_op

   custom_op.load_library('/home/user/libmy_ops.so')

   out_tensor = torch.ops.my_ops.tensor_negate(in_tensor)

Note: The ``load_library`` API does not need to be called in the same process where the library is built with the load API. Similar to regular PyTorch Custom Operators, Neuron Custom C++ Operators are built and loaded at the same time when the ``load`` API is called.  


Performance Guidance
--------------------

When possible, it is recommended that operators supported by the designated framework with supported compilation onto Neuron devices are used. These operators have been have been highly optimized for the Neuron architecture. However, for other scenarios where Custom C++ operators are the required solution, the following recommendations can be followed to improve performance:

* Use the provided memory management accessors (streaming and tcm accessor). Both of these accessors improve data fetch overhead. See the :ref:`custom-ops-api-ref-guide` for more information.
* You can optionally specify the estimated amount of stack space (in bytes) used in your Custom C++ operator via the ``extra_cflags`` argument in the call to ``custom_op.load()``. For instance, if you anticipate your operator using ~20KB of stack space, include the argument ``extra_cflags=['-DSTACK_SIZE=20000']`` in the call to custom_op.load(). **This is necessary only if you anticipate the stack to grow beyond ~8KB.** This flag is used to decide whether to place the stack in faster local memory, which significantly improves performance, or if we will need to place the stack in larger NeuronCore memory with longer access latency. If you do not specify this flag, or the estimate you provide is small enough (less than ~8KB), the stack will go in local memory. Note, when placed in local memory, the stack space will not be restricted by your estimate, but if your stack grows beyond ~8KB, there's a risk of a stack overflow, and you will be notified with an error message from GPSIMD should such a case occur. If you do specify a stack size, the maximum supported stack size is 400KB.
* Use multiple GPSIMD cores when possible to parallelize (and hence improve performance) of Custom C++ operator, refer to `Using multiple GPSIMD cores`  section in :ref:`custom-ops-api-ref-guide` for more information.

Functional Debug
----------------

Custom C++ operators support the use of the C++ language's ``printf()``. For functional debug, the recommended approach is using ``printf()`` to print input, intermediate, and final values. Consult the :ref:`custom-ops-api-ref-guide` for more information.


