.. _error-code-evr016:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVR016.

NCC_EVRF016
===========

The NCC_EVRF016 error is raised when the Neuron compiler detects that you are trying to use an integer or boolean type with one of the restricted reduction functions.

**Error message**: The scatter-reduce operation cannot perform reduction logic if the data being scattered or the destination tensor is using an integer or boolean data type.

The hardware instructions used on the Neuron device for these specific scatter-and-reduce functions are optimized for and limited to floating-point arithmetic. When the compiler detects that you are trying to use an integer or boolean type with one of the restricted reduction functions, it stops the compilation process to prevent a hardware crash or incorrect calculation.

**Example of the error**

The following example shows the **NCC\_EVRF016** error because the :code:`input_tensor` is defined using an integer data type (:code:`torch.int32`) while being used with a reduction function (:code:`reduce='sum'`) in the :code:`scatter_reduce_` operation.

.. code-block:: python

    def forward(self, input_tensor, indices_tensor, src_tensor):
        output = input_tensor.clone()
        
        output.scatter_reduce_(
            dim=1,
            index=indices_tensor,
            src=src_tensor,
            reduce='sum',
        )
        return output

    # ERROR: using integer dtype with scatter-reduce
    input_tensor = torch.zeros(BATCH_SIZE, DIM_SIZE, dtype=torch.int32)
    ...

**How to fix**

To fix this error, you must cast your input and source tensors to a floating-point data type (e.g., torch.float32 or torch.bfloat16).

.. code-block:: python

    def forward(self, input_tensor, indices_tensor, src_tensor):
        output = input_tensor.clone()
        
        output.scatter_reduce_(
            dim=1,
            index=indices_tensor,  
            src=src_tensor,        
            reduce='sum',
        )
        return output

    # FIXED: changed to float32
    # now works with scatter-reduce
    input_tensor = torch.zeros(BATCH_SIZE, DIM_SIZE, dtype=torch.float32)
    ...
