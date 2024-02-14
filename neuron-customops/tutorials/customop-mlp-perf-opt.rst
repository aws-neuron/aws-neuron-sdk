.. _neuronx-customop-mlp-perf:

Neuron Custom C++ Operators Performance Optimization
====================================================

In this tutorial, we will build on the small MLP model shown in :ref:`neuronx-customop-mlp-tutorial` and demonstrate methods to optimize the performance of a custom C++ operator. We will be taking advantage of the TCM accessor as well as the usage of multiple GPSIMD cores to enhance performance.

This tutorial assumes the reader has read and set up an environment described in :ref:`neuronx-customop-mlp-tutorial`.

.. contents:: Table of Contents
    :local:
    :depth: 2

Download Examples
-----------------

To download the source code for this tutorial, do:

.. code:: bash

    git clone https://github.com/aws-neuron/aws-neuron-samples.git
    cd aws-neuron-samples/torch-neuronx/inference/customop_mlp

.. note:: 
    We will be using an inference example in this tutorial in order to adhere to certain Custom C++ operator restrictions when using multiple GPSIMD cores (see :ref:`custom-ops-api-ref-guide`  for details on current restrictions).

.. note::

    Custom C++ Operators are supported as of Neuron SDK Version 2.7 as a beta feature. As such this feature is not installed by default, additional tooling and library packages (RPM and DEB) are required. 

    For AL2023 only, the following packages need be installed as dependencies:
    ::
      sudo yum install libnsl
      sudo yum install libxcrypt-compat
    
    On AL2 and AL2023, they can be installed with the following commands:
    ::
      sudo yum remove python3-devel -y
      sudo yum remove aws-neuronx-gpsimd-tools-0.* -y
      sudo yum remove aws-neuronx-gpsimd-customop-lib-0.* -y
      
      sudo yum install python3-devel -y
      sudo yum install aws-neuronx-gpsimd-tools-0.* -y 
      sudo yum install aws-neuronx-gpsimd-customop-lib-0.* -y

    On Ubuntu, they can be installed with the following commands:
    ::
      sudo apt-get remove python3-dev -y
      sudo apt-get remove aws-neuronx-gpsimd-tools=0.* -y
      sudo apt-get remove aws-neuronx-gpsimd-customop-lib=0.* -y  
      
      sudo apt-get install python3-dev -y
      sudo apt-get install aws-neuronx-gpsimd-tools=0.* -y
      sudo apt-get install aws-neuronx-gpsimd-customop-lib=0.* -y  

Activate the virtual environment created in :ref:`neuronx-customop-mlp-tutorial`,

.. code:: shell

    source ~/aws_neuron_venv_pytorch/bin/activate

As a reminder, ``ninja`` should be already installed in the virtual environment. If not, install it for PyTorch Custom Extensions in your environment by running:

.. code:: bash

    pip install regex
    pip install ninja

Model Configuration Adjustment
------------------------------

For this tutorial, we will enlarge the size of the hidden layer from ``[120, 84]`` to ``[4096, 2048]`` in ``model.py``.

.. code-block:: python
    :emphasize-lines: 8

    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import my_ops

    # Declare 3-layer MLP for MNIST dataset                                                                
    class MLP(nn.Module):
        def __init__(self, input_size = 28 * 28, output_size = 10, layers = [4096, 2048]):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, layers[0])
            self.fc2 = nn.Linear(layers[0], layers[1])
            self.fc3 = nn.Linear(layers[1], output_size)

        def forward(self, x):
            f1 = self.fc1(x)
            r1 = my_ops.Relu.apply(f1)
            f2 = self.fc2(r1)
            r2 = my_ops.Relu.apply(f2)
            f3 = self.fc3(r2)
            return torch.log_softmax(f3, dim=1)

Performance with Element-wise Accessor
---------------------------------------

The ``neuron`` directory contains the same code shown in :ref:`neuronx-customop-mlp-tutorial`, where the ``relu_forward`` is implemented with element-wise accessor. Go to ``neuron`` directory, run ``build.py`` then ``inference.py``, the expected output on a trn1 instance is,

.. code-block:: bash

    Inf throughput (iter/sec): 8.098649744235592
    ----------End Inference ---------------

Performance with TCM Accessor
-----------------------------
Now we switch to ``neuron-tcm`` folder. As mentioned in :ref:`custom-ops-api-ref-guide`, TCM accessors provide faster read and write performance. We implement the ``relu_forward`` using TCM accessor in ``relu.cpp``:

.. code-block:: c++

    torch::Tensor relu_forward(const torch::Tensor& t_in) {
        size_t num_elem = t_in.numel();
        torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat); 

        static constexpr size_t buffer_size = 1024;
        float *tcm_buffer = (float*)torch::neuron::tcm_malloc(sizeof(float) * buffer_size);

        if (tcm_buffer != nullptr) {
            auto t_in_tcm_acc = t_in.tcm_accessor();
            auto t_out_tcm_acc = t_out.tcm_accessor();

            for (size_t i = 0; i < num_elem; i += buffer_size) {
            size_t remaining_elem = num_elem - i;
            size_t copy_size = (remaining_elem > buffer_size) ? buffer_size : remaining_elem;

            t_in_tcm_acc.tensor_to_tcm<float>(tcm_buffer, i, copy_size);
            for (size_t j = 0; j < copy_size; j++) {
                tcm_buffer[j] = tcm_buffer[j] > 0.0 ? tcm_buffer[j] : 0.0;
            }
            t_out_tcm_acc.tcm_to_tensor<float>(tcm_buffer, i, copy_size);
            }
        }
        torch::neuron::tcm_free(tcm_buffer);
        return t_out;
    }

Run ``build.py`` then ``inference.py``, the expected output on a trn1 instance is:

.. code-block:: bash

    Inf throughput (iter/sec): 220.73800131604054
    ----------End Inference ---------------

Extending the example to utilize multiple GPSIMD cores
------------------------------------------------------

Now we switch to the ``neuron-multicore`` folder. We first enable the usage of multiple GPSIMD cores by ``multicore=True`` in the ``build.py``. 

.. code-block:: python

    custom_op.load(
        name='relu',
        compute_srcs=['relu.cpp'],
        shape_srcs=['shape.cpp'],
        build_directory=os.getcwd(),
        multicore=True,
        verbose=True
    )

After passing the flag, the kernel function ``relu_forward`` defined in ``relu.cpp`` will execute on all GPSIMD cores. Thus we need to use ``cpu_id`` to partition the workload among all cores. 

.. code-block:: c++

    torch::Tensor relu_forward(const torch::Tensor& t_in) {
        size_t num_elem = t_in.numel();
        torch::Tensor t_out = get_dst_tensor();

        uint32_t cpu_id = get_cpu_id();
        uint32_t cpu_count = get_cpu_count();
        uint32_t partition = num_elem / cpu_count;
        if (cpu_id == cpu_count - 1) {
            partition = num_elem - partition * (cpu_count - 1);
        }

        static constexpr size_t buffer_size = 1024;
        float *tcm_buffer = (float*)torch::neuron::tcm_malloc(sizeof(float) * buffer_size);

        if (tcm_buffer != nullptr) {
            auto t_in_tcm_acc = t_in.tcm_accessor();
            auto t_out_tcm_acc = t_out.tcm_accessor();

            for (size_t i = 0; i < partition; i += buffer_size) {
            size_t remaining_elem = partition - i;
            size_t copy_size = (remaining_elem > buffer_size) ? buffer_size : remaining_elem;

            t_in_tcm_acc.tensor_to_tcm<float>(tcm_buffer, partition *cpu_id + i, copy_size);
            for (size_t j = 0; j < copy_size; j++) {
                tcm_buffer[j] = tcm_buffer[j] > 0.0 ? tcm_buffer[j] : 0.0;
            }
            t_out_tcm_acc.tcm_to_tensor<float>(tcm_buffer, partition *cpu_id + i, copy_size);
            }
        }
        torch::neuron::tcm_free(tcm_buffer);
        return t_out;
    }

There are two things noteworthy in the code:

1. We use ``cpu_id`` and ``cpu_count`` to distribute the workload among all cores. Particularly, each cores performs ``relu`` on a partition of the tensor, the offset is computed based on ``cpu_id``.
2. The output of the operator is directly written to the tensor from ``get_dst_tensor()``. The ``return t_out;`` statement is ignored during execution.

Run ``build.py`` then ``inference.py``, the expected output on a trn1 instance is:

.. code-block:: bash

    Inf throughput (iter/sec): 269.936119707143
    ----------End Inference ---------------

Details of the API used in the sample here can be found in :ref:`custom-ops-api-ref-guide`. 

