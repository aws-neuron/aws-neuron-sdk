.. _nxdi-troubleshooting:

Troubleshooting Guide for NxD Inference
=======================================

This guide provides solutions for common issues encountered when using NxD Inference.

.. contents:: Table of contents
   :local:
   :depth: 2

Accuracy Issues
----------------

The primary methods for validating model accuracy on Neuron involve both token-by-token output matching and logit-level error analysis (relative or max absolute error) against a pre-calibrated GPU FP32 or CPU FP32 reference. When output deviations are observed, these can be systematically attributed to factors such as tokenizer/input discrepancies, amplification from large weight norms (high Lipschitz constants), quantization or precision loss, differences in operator implementation or kernel fusion, compiler optimization, or unintended hardware-level datatype casts.

When validating model accuracy on Neuron, it is important to recognize that predicting the exact output deviations from a high-precision reference (like CPU or GPU FP32) is theoretically NP-hard, due to the complex and nonlinear nature of large neural networks. Rather than attempting to anticipate every possible numerical difference, the recommended strategy is to systematically identify, localize, and diagnose deviations as they occur.

Accuracy Degradation with Auto-Cast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: You may observe accuracy degradation in model outputs when using the default auto-cast behavior of the Neuron compiler.

**Explanation**: By default, the Neuron compiler automatically casts certain operations to lower precision data types (BF16) to improve performance. While this works well for most cases, it can sometimes lead to accuracy issues, especially in operations involving integer-to-float conversions.

**Solution**: Use the ``--auto-cast=none`` compiler flag to disable automatic casting. This preserves the original precision of operations at the cost of some performance.

Example using inference_demo:

.. code:: bash

   inference_demo --model-type llama --task-type causal-lm run \
       --model-path <path> \
       --compiled-model-path <path> \
       --torch-dtype bfloat16 \
       --tp-degree <value> \
       --batch-size <value> \
       --max-context-length <value> \
       --seq-len <value> \
       --on-device-sampling \
       --prompt "Your prompt here" \
       --compiler-args "--auto-cast=none"

Example using NeuronConfig:

.. code:: python

   from neuronx_distributed_inference.models.config import NeuronConfig
   
   neuron_config = NeuronConfig(
       tp_degree=32,
       batch_size=1,
       max_context_length=1024,
       seq_len=2048,
       compiler_args="--auto-cast=none"
   )

Integer-to-Float Conversion Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Operations involving integer-to-float conversions (such as in rotary embeddings) may experience significant accuracy degradation when auto-cast is enabled.

**Explanation**: When integer values are converted to floating point and then automatically cast to lower precision (like BF16), the precision loss can be substantial. This is particularly problematic in operations like rotary embeddings where position IDs are converted to floating point for computing sin/cos values.

**Solution**: Use the ``--auto-cast=none`` compiler flag to prevent downcasting these operations. This is especially important for models that use rotary embeddings or similar position encoding mechanisms.

**Technical Details**: The issue occurs in operations like:

.. code:: python

   # Integer position IDs are converted to float for sin/cos computation
   # Downcasting to BF16 here can cause significant precision loss
   position_ids = position_ids.to(torch.bfloat16)
   sin, cos = self.compute_sin_cos(position_ids)

Memory Usage Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Note**: Using ``--auto-cast=none`` will increase memory usage as operations will use higher precision data types. Ensure your instance has sufficient memory when using this flag.

Performance Impact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Note**: Disabling auto-cast will typically result in slower inference. The exact performance impact depends on your model architecture and hardware configuration. Consider this trade-off when optimizing for accuracy.


Array indexing and in-place operations in Neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: When building attention masks, operations that combine array slicing with in-place modifications (e.g., ``mask_i[: arx[0] * arx[1], :ntok] = 0``) can cause accuracy issues in Neuron. This is particularly problematic when the array indices are dynamically computed.

**Explanation**: The accuracy issue stems from two main factors:

1. Array Slicing with Dynamic Ranges:

.. code:: python

   # Problematic: Array slicing with dynamic range (arx[0] * arx[1])
   mask_i[: arx[0] * arx[1], :ntok] = 0

- Uses computed indices to access specific portions of the tensor
- Dynamic ranges can lead to unpredictable memory access patterns

2. In-place Modifications:

.. code:: python

   # Problematic: Modifying tensor in-place
   mask_i[...] = 0  # Direct modification of the original tensor

- Changes the original tensor's values directly
- Can cause issues with Neuron's memory management and optimization

**Solution**: Replace array slicing and in-place operations with element-wise operations:

.. code:: python

   # Instead of array slicing and in-place modification:
   mask_i[: arx[0] * arx[1], :ntok] = 0  # Problematic

   # Use element-wise operations:
   arx_mask = (torch.arange(num_chunks, device=x.device) >= (arx[0] * arx[1])).to(dtype=x.dtype)
   mask_i[:, :ntok] *= arx_mask.view(num_chunks, 1, 1)  # Neuron-friendly

**Example**: 
File: `test/unit/models/mllama/test_vision_encoder_attention_mask.py <https://github.com/aws-neuron/neuronx-distributed-inference/blob/9b90cd02ffc3cc76bb3e81113a177f10d7a350a8/test/unit/models/mllama/test_vision_encoder_attention_mask.py>`__

.. code:: python

   # CPU version (problematic in Neuron):
   def build_encoder_attention_mask_meta(x, ar, ntok, num_chunks, n_heads):
       masks = []
       for arx in ar:
           mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
           mask_i[: arx[0] * arx[1], :ntok] = 0  # Problematic: array slicing + in-place
           # ...

   # Neuron-friendly version:
   def build_encoder_attention_mask(x, ar, ntok, num_chunks, n_heads):
       masks = []
       for arx in ar:
           mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype, device=x.device)
           arx_mask = (torch.arange(num_chunks, device=x.device) >= (arx[0] * arx[1])).to(dtype=x.dtype)
           mask_i[:, :ntok] *= arx_mask.view(num_chunks, 1, 1)  # Element-wise operation
           # ...

**Note**: This pattern applies to similar operations where array slicing and in-place modifications are used together. 
Consider using element-wise operations and avoiding in-place modifications for better Neuron compatibility.


Performance Issues
--------------------


Skip model warmup during inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: You may observe slower performance for the first few inference requests, particularly on Trn2.

**Explanation**: By default, model warmup is disabled (``skip_warmup=True``) on Trn2 since warmup feature is not yet implemented for Trn2. This means the model needs to "warm up" naturally through actual inference requests, leading to slower performance during the initial requests.


**Solution**: There are approaches to ensure initial request performance:

1. Enable built-in warmup if your configuration supports it (on Inf2, Trn1):

.. code:: python

   neuron_config = NeuronConfig(
       tp_degree=32,
       batch_size=1,
       # skip_warmup=True is the default for Trn2 in release 2.23
       # skip_warmup=False is the default for Trn1, Inf2 in release 2.23
   )

2. Implement manual warmup by sending dummy requests (on all instance types):

.. code:: python

   # Send a few dummy requests before serving real traffic
   dummy_prompt = "This is a warmup request."
   for _ in range(3):  # Number of warmup iterations
       model.generate(
           prompt=dummy_prompt,
           max_new_tokens=32
       )


**Note**:
 
- When using vLLM for serving, the same initial performance impact applies if warmup is disabled.
- Use `--override-neuron-config "{\"skip_warmup\":false}"` to change the warmup setting

**Best Practice**: 

- For production environments where initial latency is critical, test if your configuration supports built-in warmup.
- If built-in warmup isn't supported, implement manual warmup before serving real traffic.
- For development or non-latency-critical scenarios, the default configuration (warmup disabled) is sufficient.

Other Common Issues
--------------------

Tensor Materialization During Tracing caused unexpected model behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: Developers may inadvertently write code that forces tensor materialization during model tracing, leading to fixed computation paths and unexpected behaviors.

**Explanation**: When model logic depends on tensor values during the forward pass, the compiler may try to evaluate these values during tracing time. This "fixes" the computation path based on the initial values, resulting in a model that doesn't properly handle different runtime values.

Example of problematic code:

.. code:: python

   def forward(self, tensor):
       if tensor[0] == 1:  # Forces tensor sync during tracing
           return tensor
       else:
           return tensor * 2

**Solution**: There are two debugging approaches to detect tensor materialization issues:

1. Enable warning messages:

.. code:: python

   import os
   
   # Set before model tracing
   os.environ['PT_XLA_DEBUG_LEVEL'] = '2'  # Will print warnings when tensor sync occurs

2. Force errors on tensor materialization:

.. code:: python

   import torch_xla
   
   # Set before model tracing
   torch_xla._XLAC._set_allow_execution(False)  # Will raise an error if tensor sync is attempted

**Best Practice**: 

- Avoid control flow that depends on tensor values during tracing. Instead, consider setting flags through configurations that should not change during runtime. See below example:

.. code:: python

   class TestModel(torch.nn.Module):
      def __init__(self, flag=1):
         super().__init__()
         # the flag should be pre-determined based on the model configuration
         # it should not be an input of the model during runtime
         self.flag = flag

      def forward(self, tensor):
         if self.flag:
               return tensor
         else:
               return tensor * 2

- If dynamic model path is required, consider using JIT inference (See: :ref:`trace-vs-xla-lazytensor`)


Input Data Type Handling for int64/fp64 due to compiler dtype compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


**Issue**: While you may be using 64-bit data types (int64/fp64) from tokenizers or other input sources, be aware that these are automatically converted to 32-bit types inside `ModelWrapper <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/model_wrapper.py>`__.

**Explanation**: The Neuron compiler is optimized for 32-bit data types. To ensure optimal accuracy and compatibility, the model wrapper automatically converts 64-bit inputs (like those from Hugging Face tokenizers) to their 32-bit equivalents (int64 → int32, fp64 → fp32).

**Note**: No action is required from users as this conversion is handled automatically.

**Best Practice**:
 
- Continue using your tokenizers and input pipelines as normal
- Be aware that 64-bit inputs are automatically converted to 32-bit when using `ModelWrapper <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/model_wrapper.py>`__
- If you're implementing custom pre-processing, using 32-bit types directly can be more efficient

This automatic conversion ensures consistent accuracy and compatibility with the Neuron compiler while maintaining ease of use with standard tokenizers and input pipelines.