.. _neuronx-customop-mlp-tutorial:

Neuron Custom C++ Operators in MLP Training 
===========================================

In this tutorial we’ll demonstrate how to prepare a PyTorch model that contains a custom operator (ie. CppExtension) for Neuron compilation to run on Trainium EC2 instances. To learn more about Neuron CustomOps see :ref:`neuron_c++customops`. For a deeper dive on MNIST or Multi-Layer Perceptron models, see the :ref:`neuronx-mlp-training-tutorial`. This tutorial assumes the reader is familiar with `PyTorch Custom Extensions <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_.

.. contents:: Table of Contents
   :local:
   :depth: 2

Setup Environment and Download Examples
---------------------------------------

Before running the tutorial please follow the installation instructions at:

* :ref:`pytorch-neuronx-install` on Trn1

.. note::
    The name of ``aws-neuronx-gpsimd-customop`` has been changed to ``aws-neuronx-gpsimd-customop-lib`` as of the neuron 2.10 release.

.. note::

    Custom C++ Operators are supported as of Neuron SDK Version 2.7 as a beta feature. As such this feature is not installed by default, additional tooling and library packages (RPM and DEB) are required. 

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

  
For all the commands below, make sure you are in the virtual environment that you have created above before you run the commands:

.. code:: shell

    source ~/aws_neuron_venv_pytorch/bin/activate

Install dependencies for PyTorch Custom Extensions in your environment by running:

.. literalinclude:: tutorial_source_code/custom_c_mlp_training/custom_c_mlp_training_code.sh
   :language: bash
   :lines: 5-6

The ``ninja`` package is only needed for the reference CPU example. It is not needed by Neuron to run on Trainium instances.
    
To download the source code for this tutorial, do:

.. code:: bash

    git clone https://github.com/aws-neuron/aws-neuron-samples.git
    cd aws-neuron-samples/torch-neuronx/training/customop_mlp

In the ``customop_mlp`` directory there are two subdirectories. The ``pytorch`` directory contains an example model and training script using a custom operator that runs using the cpu device with standard PyTorch APIs and libraries (ie. not specific to AWS/Neuron). The ``neuron`` directory contains a version of the same model and training script with the custom operator ported to Neuron to run on trn1 using the XLA device. 

Basic PyTorch Custom Relu Operator
----------------------------------

For the next few sections we’ll review the example model in the ``pytorch`` directory. This is a condensed and simplified explanation of PyTorch C++ Extensions, for more details see the `PyTorch documentation <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_. In ``my_ops.py`` we implement a custom relu activation op as a torch autograd function so that we can use it in a training loop:

.. code-block:: python

    import torch

    torch.ops.load_library('librelu.so')

    class Relu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.ops.my_ops.relu_forward(input)

        @staticmethod
        def backward(ctx, grad):
            input, = ctx.saved_tensors
            return torch.ops.my_ops.relu_backward(grad, input), None

Notice that here we first load ``librelu.so`` using the ``load_library`` API. And then call the ``relu_forward`` and ``relu_backward`` functions from our library within the relevant static methods. 

We implemented these two library functions in the ``relu.cpp`` file:

.. code-block:: c++

    torch::Tensor relu_forward(const torch::Tensor& t_in) {
        ...
        t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_in_acc[i][j] : 0.0;
        ...
    }

    torch::Tensor relu_backward(const torch::Tensor& t_grad, const torch::Tensor& t_in) {
        ...
        t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_grad_acc[i][j] : 0.0;
        ...
    }

    TORCH_LIBRARY(my_ops, m) {
        m.def("relu_forward", &relu_forward);
        m.def("relu_backward", &relu_backward);
    }

And then built them into a library using the PyTorch Cpp Extension APIs in the ``build.py`` script:

.. code-block:: python

    torch.utils.cpp_extension.load(
        name='librelu',
        sources=['relu.cpp'],
        is_python_module=False,
        build_directory=os.getcwd()
    )

Run ``python build.py`` to produce the ``librelu.so`` library.
    
Multi-layer perceptron MNIST model
----------------------------------

In ``model.py``, we define the multi-layer perceptron (MLP) MNIST model with 3 linear layers and a custom ReLU activation, followed by a log-softmax layer. Highlighted below are the relevant custom changes in the ``model.py`` file:

.. code-block:: python
    :emphasize-lines: 4, 16, 18

    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import my_ops

    # Declare 3-layer MLP for MNIST dataset                                                                
    class MLP(nn.Module):
        def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
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

Training the MLP model on CPU
-----------------------------

In the ``train_cpu.py`` script we load the MNIST train dataset, instantiate the MLP model, and use ``device='cpu'`` to execute on the host CPU. Expected CPU output:

.. code:: bash

    ----------Training ---------------
    Train throughput *(*iter/sec*)*: *286*.96994718801335
    Final loss is *0*.1040
    ----------End Training ---------------

Neuron Relu CustomOp
--------------------

Now switch over into the ``neuron`` directory. To migrate our PyTorch customOp to Neuron, we have to make a few small changes. First, we create a new ``shape.cpp`` file to implement our shape function as required by XLA (see :ref:`feature-custom-operators-devguide` for details). We also replace the ``TORCH_LIBRARY`` API with ``NEURON_LIBRARY``.

.. code-block:: c++

    torch::Tensor relu_fwd_shape(torch::Tensor t_in) {
        torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat);
        return t_out;
    }

    torch::Tensor relu_bwd_shape(torch::Tensor t_grad, torch::Tensor t_in) {
        torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat);
        return t_out;
    }

    NEURON_LIBRARY(my_ops, m) {
        m.def("relu_forward", &relu_fwd_shape, "relu_forward");
        m.def("relu_backward", &relu_bwd_shape, "relu_backward");
    }

And then we build it using the ``torch_neuronx`` package in ``build.py``:

.. code-block:: python

    from torch_neuronx.xla_impl import custom_op

    custom_op.load(
        name='relu',
        compute_srcs=['relu.cpp'],
        shape_srcs=['shape.cpp'],
        build_directory=os.getcwd()
    )

Notice that here we specify both the ``relu.cpp`` and ``shape.cpp`` files separately. This is because the shape functions will be compiled with an x86 compiler and run on the host during the XLA compilation, and the compute functions will be compiled for the NeuronCore accelerator and executed during the training loop. Running ``build.py`` produces the same ``librelu.so`` as in the CPU example, but compiles the source code to execute on the NeuronCore.

In our ``my_ops.py`` file we just use the ``torch_neuronx`` API to load our new library and execute our customOp exactly the same way we did before:

.. code-block:: python

    import torch
    import torch_neuronx
    from torch_neuronx.xla_impl import custom_op

    custom_op.load_library('librelu.so')

    class Relu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.ops.my_ops.relu_forward(input)

        @staticmethod
        def backward(ctx, grad):
            input, = ctx.saved_tensors
            return torch.ops.my_ops.relu_backward(grad, input), None

Training the MLP model on Trainium
----------------------------------

In the ``train.py`` script we modify the CPU training script ``train_cpu.py`` to run with PyTorch Neuron torch_xla. Expected output on a trn1 instance:

.. code:: bash

    ----------Training ---------------
    2023-02-02 22 (tel:2023020222):46:58.000299: INFO ||NCC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/USER_neuroncc-2.0.0.8683a0+c94c3936c/MODULE_4447837791278761679/MODULE_0_SyncTensorsGraph.329_4447837791278761679_ip-172-31-38-167.us-west-2.compute.internal-49ad7ade-14011-5f3bf523d8788/1650ba41-bcfd-4d15-9038-16d391c4a57c/MODULE_0_SyncTensorsGraph.329_4447837791278761679_ip-172-31-38-167.us-west-2.compute.internal-49ad7ade-14011-5f3bf523d8788.neff. Exiting with a successfully compiled graph
    2023-02-02 22 (tel:2023020222):46:58.000433: INFO ||NCC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/USER_neuroncc-2.0.0.8683a0+c94c3936c/MODULE_16964505026440903899/MODULE_1_SyncTensorsGraph.401_16964505026440903899_ip-172-31-38-167.us-west-2.compute.internal-4d0cabba-14011-5f3bf529794a3/23d74230-59dd-4347-b247-fa98aed416bd/MODULE_1_SyncTensorsGraph.401_16964505026440903899_ip-172-31-38-167.us-west-2.compute.internal-4d0cabba-14011-5f3bf529794a3.neff. Exiting with a successfully compiled graph
    Train throughput (iter/sec): 117.47151142662648
    Final loss is 0.1970
    ----------End Training ---------------
