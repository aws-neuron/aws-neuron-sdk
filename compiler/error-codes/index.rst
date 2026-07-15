.. meta::
    :description: "Neuron Compiler error code documentation home."
    :date-modified: 07/14/2026

.. _ncc-errors-home:

Neuron Compiler Error Codes
============================

This page lists the error codes you can encounter while developing with the Neuron Compiler. For more details on any individual error, click the link for that error code in the table below.

.. list-table::
   :header-rows: 1

   * - Error Code
     - Error Message
     - Recommendation
   * - :ref:`NCC_EARG001 <error-code-earg001>`
     - Unsupported Logical Neuron Core (LNC) configuration.
     - You attempted to use a Logical Neuron Core configuration that is not supported by the target Neuron architecture.
   * - :ref:`NCC_EBIR023 <error-code-ebir023>`
     - MLP kernel intermediate size exceeds the maximum supported value of 4096.
     - Consider tiling large intermediate tensors in your kernel to stay within the supported limit, or increase tensor parallelism to shard the intermediate dimension across more cores.
   * - :ref:`NCC_EBVF030 <error-code-ebvf030>`
     - The number of instructions generated exceeds the limit.
     - Consider applying model parallelism as partitioning the model will help break large computational graphs into smaller subgraphs.
   * - :ref:`NCC_EHCA005 <error-code-ehca005>`
     - The compiler encountered a custom call instruction with a target name that is not recognized.
     - Use a supported custom call target from the list of recognized targets.
   * - :ref:`NCC_EMOD025 <error-code-emod025>`
     - Dynamic shape is not supported. The Neuron compiler requires all tensor dimensions to be statically sized.
     - Recompile the model with fully static input shapes. When compiling via PyTorch torch.compile, set dynamic=False to disable dynamic shape specialization.
   * - :ref:`NCC_EOOM001 <error-code-eoom001>`
     - The combined memory needed for the model's activation tensors exceeds the high-bandwidth memory limit.
     - You may need to reduce batch/tensor size or utilize pipeline/tensor parallelism via neuronx-distributed.
   * - :ref:`NCC_EOOM002 <error-code-eoom002>`
     - The combined memory needed for the model's activation tensors exceeds the high-bandwidth memory limit.
     - You may need to reduce batch/tensor size or utilize pipeline/tensor parallelism via neuronx-distributed.
   * - :ref:`NCC_ESFH002 <error-code-esfh002>`
     - The compiler encountered a unsigned 64-bit integer constant with a value that cannot be safely converted to 32-bit representation.
     - Try to use uint32 for constants when possible and restructure code to avoid large constants.
   * - :ref:`NCC_ESPP004 <error-code-espp004>`
     - The compiler encountered a data type that is not supported for code generation.
     - Use a supported data type as listed in the Neuron documentation.
   * - :ref:`NCC_ESPP047 <error-code-espp047>`
     - Unsupported 8-bit floating-point data type.
     - The compiler found usage of an unsupported 8-bit floating-point data type. Convert to a supported type like torch.float16.
   * - :ref:`NCC_EUOC002 <error-code-euoc002>`
     - An unsupported operator was used.
     - Try using alternative operators from the full list of supported operators via `neuronx-cc list-operators --framework XLA` to workaround the limitation.
   * - :ref:`NCC_EVRF001 <error-code-evrf001>`
     - An unsupported operator was used.
     - Try using alternative operators from the full list of supported operators to workaround the limitation.
   * - :ref:`NCC_EVRF004 <error-code-evrf004>`
     - Complex data types are not supported on the Neuron device.
     - You cannot use complex data types (such as ``complex64``, ``complex128``, and others) on the Neuron device directly.
   * - :ref:`NCC_EVRF005 <error-code-evrf005>`
     - Unsupported F8E4M3FNUZ, F8E4M3B11FNUZ, or F8E5M2FNUZ data type.
     - The compiler found usage of unsupported 8-bit floating-point data types. Convert to a supported type like torch.float16.
   * - :ref:`NCC_EVRF006 <error-code-evrf006>`
     - The compiler encountered a RNGBitGenerator operation using a random number generation algorithm other than RNG_DEFAULT.
     - Ensure that you are using standard JAX/PyTorch random APIs and not explicitly specifying an RNG algorithm.
   * - :ref:`NCC_EVRF007 <error-code-evrf007>`
     - The number of instructions generated exceeds the limit.
     - Consider applying model parallelism as partitioning the model will help break large computational graphs into smaller subgraphs.
   * - :ref:`NCC_EVRF009 <error-code-evrf009>`
     - The combined memory needed for the model's activation tensors exceeds the high-bandwidth memory limit.
     - You may need to reduce batch/tensor size or utilize pipeline/tensor parallelism via neuronx-distributed.
   * - :ref:`NCC_EVRF010 <error-code-evrf010>`
     - The compiler encountered simultaneous use of input and kernel dilation, which is not supported.
     - If possible, use only input or kernel dilation, not both simultaneously.
   * - :ref:`NCC_EVRF011 <error-code-evrf011>`
     - The compiler encountered strided convolution combined with dilated input, which is not supported.
     - If possible, remove stride or input dilation, or apply upsampling and downsampling separately.
   * - :ref:`NCC_EVRF013 <error-code-evrf013>`
     - TopK does not support integer input tensors (int32, int64).
     - The TopK operation cannot be performed on integer data types.
   * - :ref:`NCC_EVRF015 <error-code-evrf015>`
     - The compiler encountered a custom call instruction with a target name that is not recognized.
     - Use a supported custom call target from the list of recognized targets.
   * - :ref:`NCC_EVRF016 <error-code-evr016>`
     - The scatter-reduce operation cannot perform reduction logic if the data being scattered or the destination tensor is using an integer or boolean data type.
     - Cast your input and source tensors to a floating-point data type (e.g., torch.float32 or torch.bfloat16).
   * - :ref:`NCC_EVRF017 <error-code-evrf017>`
     - Reduce-window operation with base dilation greater than 1 is not supported.
     - Change base dilation to be all 1s or consider manual dilation if necessary.
   * - :ref:`NCC_EVRF018 <error-code-evrf018>`
     - Reduce-window operation with window dilation greater than 1 is not supported.
     - Remove window_dilation or change values to be all 1s, or consider manual dilation if necessary.
   * - :ref:`NCC_EVRF019 <error-code-evrf019>`
     - The compiler encountered a reduce-window operation with more or less than 2 operands.
     - If possible, split multi-operand reduce_window with multiple single-operand reduce_window operations.
   * - :ref:`NCC_EVRF022 <error-code-evrf022>`
     - Shift-right-arithmetic operation on non 32-bit inputs is not supported. Cast the first argument's data type to be S32, U32, or F32.
     - You need to use 32-bit data types for shift operations. Cast inputs to int32, uint32, or float32.
   * - :ref:`NCC_EVRF031 <error-code-evrf031>`
     - The compiler encountered a scatter out-of-bounds error.
     - Ensure that the iota size matches the operand dimension size.
   * - :ref:`NCC_EVRF036 <error-code-evrf036>`
     - QuantizeMX custom call has invalid backend_config JSON.
     - Provide a valid JSON object in backend_config.
   * - :ref:`NCC_EVRF037 <error-code-evrf037>`
     - QuantizeMX custom call operand count must be exactly 1 (input tensor).
     - Pass exactly one input tensor as the operand to QuantizeMX.
   * - :ref:`NCC_EVRF038 <error-code-evrf038>`
     - QuantizeMX custom call dim is invalid for input tensor rank.
     - Use the last dimension, or the second-to-last dimension for inputs with rank 2 or greater.
   * - :ref:`NCC_EVRF039 <error-code-evrf039>`
     - QuantizeMX custom call block_size must be 32.
     - Use block_size=32 as required by the OCP MXFP specification.
   * - :ref:`NCC_EVRF040 <error-code-evrf040>`
     - QuantizeMX custom call scale_method is unsupported.
     - Use "EMAX", the only supported scale method.
   * - :ref:`NCC_EVRF041 <error-code-evrf041>`
     - QuantizeMX custom call input type is unsupported.
     - Cast input tensor to BF16 or F16 before quantization.
   * - :ref:`NCC_EVRF042 <error-code-evrf042>`
     - QuantizeMX custom call is malformed.
     - Use a supported logical FP8 dtype and a correctly shaped, U32-packed quantized_data output.
   * - :ref:`NCC_EVRF043 <error-code-evrf043>`
     - ScaledMatmul custom call must have exactly 4 operands.
     - Pass all 4 operands: lhs, rhs, lhs_scale, rhs_scale.
   * - :ref:`NCC_EVRF044 <error-code-evrf044>`
     - ScaledMatmul custom call LHS input type is unsupported.
     - Use the packed U32 quantized data tensor returned by QuantizeMX.
   * - :ref:`NCC_EVRF045 <error-code-evrf045>`
     - ScaledMatmul custom call output type is unsupported.
     - Declare the result as F32 or BF16.
   * - :ref:`NCC_EVRF046 <error-code-evrf046>`
     - ScaledMatmul custom call LHS tensor must have rank >= 2.
     - Reshape the LHS to have at least 2 dimensions.
   * - :ref:`NCC_EVRF047 <error-code-evrf047>`
     - ScaledMatmul custom call RHS tensor must have rank >= 2.
     - Reshape the RHS to have at least 2 dimensions.
   * - :ref:`NCC_EVRF048 <error-code-evrf048>`
     - ScaledMatmul custom call batch dimension mismatch.
     - Ensure the product of LHS and RHS batch dimension sizes match.
   * - :ref:`NCC_EVRF049 <error-code-evrf049>`
     - ScaledMatmul custom call could not parse backend_config.
     - Provide valid JSON with integer values in each dimension array.
   * - :ref:`NCC_EVRF050 <error-code-evrf050>`
     - ScaledMatmul custom call contracting dimension sizes mismatch.
     - Ensure LHS and RHS contracting dimensions have equal size.
   * - :ref:`NCC_EVRF051 <error-code-evrf051>`
     - Data type F8E4M3FN is not supported on TRN1/TRN2.
     - For QuantizeMX, target Trn3 or later without the F8E4M3 conversion flag.
   * - :ref:`NCC_EVRF052 <error-code-evrf052>`
     - Data type F8E4M3 is not supported on hardware newer than Trn3.
     - Use F8E4M3FN instead of F8E4M3.
   * - :ref:`NCC_EVRF053 <error-code-evrf053>`
     - ScaledMatmul custom call contracting dimension overlaps with batch dimension.
     - Ensure batch dimensions and contracting dimensions are disjoint.
   * - :ref:`NCC_EVRF054 <error-code-evrf054>`
     - ScaledMatmul custom call batch dimension index out of bounds.
     - Use dimension indices within valid range (0 <= dim < rank).
   * - :ref:`NCC_EVRF055 <error-code-evrf055>`
     - ScaledMatmul custom call contracting dimension index out of bounds.
     - Use dimension indices within valid range (0 <= dim < rank).
   * - :ref:`NCC_EVRF056 <error-code-evrf056>`
     - Operation gather encountered out of bound indices.
     - Ensure that the iota dimension size is less than or equal to the size of the corresponding operand dimension. Check that your model's max_position_embeddings is >= sequence_length.
   * - :ref:`NCC_EVRF057 <error-code-evrf057>`
     - QuantizeMX custom call must return a tuple with exactly 2 outputs.
     - Declare a 2-element tuple result type (quantized_data, scale).
   * - :ref:`NCC_EVRF058 <error-code-evrf058>`
     - QuantizeMX custom call input dimension must be divisible by 4.
     - Pad or reshape the input so the quantization dimension size is a multiple of 4.
   * - :ref:`NCC_EVRF059 <error-code-evrf059>`
     - Kernel file referenced by AwsNeuronCustomNativeKernel instruction does not exist on the host.
     - Ensure the NKI kernel artifact file exists at the specified path before compilation. Clear the NKI file cache and retrace the model, or copy artifacts into the compiler launch directory.
   * - :ref:`NCC_EXSP001 <error-code-exsp001>`
     - The combined memory needed for the model's activation tensors exceeds the high-bandwidth memory limit.
     - You may need to reduce batch/tensor size or utilize pipeline/tensor parallelism via neuronx-distributed.
   * - :ref:`NCC_EXTP004 <error-code-extp004>`
     - The number of instructions generated exceeds the limit.
     - Consider applying model parallelism as partitioning the model will help break large computational graphs into smaller subgraphs.

.. toctree::
    :hidden:
    :maxdepth: 1

    EARG001
    EBIR023
    EBVF030
    EHCA005
    EMOD025
    EOOM001
    EOOM002
    ESFH002
    ESPP004
    ESPP047
    EUOC002
    EVRF001
    EVRF004
    EVRF005
    EVRF006
    EVRF007
    EVRF009
    EVRF010
    EVRF011
    EVRF013
    EVRF015
    EVRF016
    EVRF017
    EVRF018
    EVRF019
    EVRF022
    EVRF031
    EVRF036
    EVRF037
    EVRF038
    EVRF039
    EVRF040
    EVRF041
    EVRF042
    EVRF043
    EVRF044
    EVRF045
    EVRF046
    EVRF047
    EVRF048
    EVRF049
    EVRF050
    EVRF051
    EVRF052
    EVRF053
    EVRF054
    EVRF055
    EVRF056
    EVRF057
    EVRF058
    EVRF059
    EXSP001
    EXTP004
