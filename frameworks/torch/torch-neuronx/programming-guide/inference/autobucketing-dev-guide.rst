.. _torch-neuronx-autobucketing-devguide:

Autobucketing for Inference (|torch-neuronx|)
=============================================

.. contents:: Table of Contents
    :depth: 3

Introduction
------------

Autobucketing is a feature that enables you to use multiple bucket models. Each bucket model accepts a static input shape and a bucket kernel function. The models are then packaged into a single traced PyTorch model that can accept multiple different input shapes. 

This gives you increased flexibility for inputs into Neuron models without the need to manage multiple Neuron models. The applications of this are extensive, from optimal model selection based on image resolution, to efficient sampling for token generation in language models.

While Autobucketing offers increased flexibility, Autobucketing is also useful for latency sensitive applications since small and large inputs can be applied on small and large models respectively, based on the bucket kernel function.

This Developer Guide will discuss best practices for implementing Autobucketing for your use case. For this Developer Guide, a BERT model will be used, where we bucket on the sequence length dimension.

Before continuing, it is recommended to familiarize yourself with the Autobucketing APIs, which can be found :ref:`here <torch-neuronx-autobucketing>`.

Bucket Kernels
--------------

Bucket kernels are user-defined functions that take in the model input as input to the function and return a tuple containing a *potentially* manipulated model input and a tensor representing the bucket index.
An important aspect of this function is that it must be able to be adapted to the TorchScript representation using :func:`torch.jit.script`. This is because to support saving a traced bucket model with :func:`torch.jit.save` and :func:`torch.jit.load`, you need all elements of the model to be in TorchScript.
The below example shows a bucket kernel that is adaptable to TorchScript in this way.

.. code-block:: python

    import torch
    from typing import List

    def sequence_length_bucket_kernel(tensor_list: List[torch.Tensor]):
      x = tensor_list[0]
      bucket_dim = 1
      x_shape = x.shape
      tensor_sequence_length = x_shape[bucket_dim]
      batch_size = x_shape[bucket_dim - 1]
      buckets = [128, 512]
      idx = 0
      num_inputs = 3
      bucket = buckets[0]
      reshaped_tensors: List[torch.Tensor] = []
      bucket_idx = 0
      for idx, bucket in enumerate(buckets):
          if tensor_sequence_length <= bucket:
              bucket_idx = idx
              for tensor in tensor_list:
                  if num_inputs == 0:
                      break
                  delta = bucket - tensor_sequence_length
                  padding_shape: List[int] = [batch_size, delta]
                  zeros = torch.zeros(padding_shape, dtype=x.dtype)
                  reshaped_tensors.append(torch.cat([tensor, zeros], dim=bucket_dim))
                  num_inputs -= 1
              break
      return reshaped_tensors, torch.tensor([bucket_idx])

  def get_bucket_kernel(*_):
      bk = torch.jit.script(sequence_length_bucket_kernel)
      return bk


In the above example we define a bucket kernel that takes in an input to a transformers model, which is ``[input_ids,attention_mask,token_type_ids]``. We first obtain the first tensor in that list, since that tensor contains ``sequence_length`` as a dimension, and retrieve the ``sequence_length`` and ``batch_size``. We also define the sequence length buckets. The next major part of the code is the for loop, which first finds the matching sequence length bucket and then iterates through the tensors in the list to right pad the tensors to the desired sequence length. After this is done, we return the padded inputs as a list of tensors and a tensor containing the bucket index. Finally, we create a function ``get_bucket_kernel`` which returns a version of the bucket kernel that has been adapted to TorchScript using using :func:`torch.jit.script`. We can use this bucket kernel to pass in a tokenized input of sequence length 1-512, which is padded to the nearest bucket size rounded up.

Note that we call :func:`torch.jit.script` instead of :func:`torch.jit.trace`. This
is because we rely on control flow logic evaluating correctly for all inputs. This
results in certain challenges when writing compatible and accurate bucket kernels. We
discuss these challenges and resolutions in the next section.

Torchscript Best Practices for Bucket Kernels
---------------------------------------------

Below are some recommendations when creating these Bucket Kernels:

    - **Type annotate non-tensor-like data types**: Functions that have been adapted to the TorchScript representation using using :func:`torch.jit.script` treat 
      variables that are defined by using another variable as tensor-like when they might not be. This can be seen when defining
      ``padding_shape`` in the above bucket kernel.
    - **Index selection support is limited**: Functions that have been adapted to the TorchScript representation using using :func:`torch.jit.script` don't support the use of variables
      for indexing very well. It could work in some scenarios, but there isn't a discernable pattern to it,
      so for more reliable TorchScript-adapted functions relying on indexes, use an enumerated for loop or literals if possible.
    - **Initializing variables with literals**: The Torchscript compiler often incorrectly removes
      a variable if it finds another variable initialized with the same literal, such as ``0``. The compiler might also reuse variables initialized with a
      literal for other operations, such as indexing or function parameters. This can cause inaccurate results for certain inputs. Therefore, always validate the
      function by testing with the expected inputs. If the lowering does not behave as expected, you can see the lowered representation by calling ``bucket_kernel.graph``, where ``bucket_kernel`` is the return value of ``get_bucket_kernel``, and analyze the graph for inaccurate lowerings.
    - **Use of aten functions might be necessary to guarantee correct lowering**: The TorchScript interpreter supports certain operations, such as slicing, but can
      lower them in unexpected ways when using normal syntax. For example, with slicing, the most common way to slice is with indexing syntax such as ``tensor[:,:2,:]``. However,
      this can cause lowering issues due to the aforementioned reasons. To mitigate this, it might be necessary to call the respective aten function directly.
      See the below example with ``shared_state_buffer_preprocessor``.

Shared State Buffers
--------------------

Autobucketing supports the concept of a shared buffer between bucket models. You can use this to define how the shared buffer can be manipulated to be fed as input to a bucket model via the ``shared_state_buffer_preprocessor``.

The above recommendations also apply when defining a ``shared_state_buffer_preprocessor``.

An example where a shared buffer is useful between bucket models is maintaining a KV Cache between bucket models for LLMs.

Below is an example of a KV Cache preprocessor for Autobucketing.

.. code-block:: python

  def state_preprocessor(shapes_collection: List[List[List[int]]], states: List[torch.Tensor], bucket_idx_tensor: torch.Tensor)->List[torch.Tensor]:
    bucket_idx = torch.ops.aten.Int(bucket_idx_tensor)
    shapes = shapes_collection[bucket_idx]
    sliced_state_tensors = []
    
    for i in range(len(shapes)):
        expected_shape = shapes[i]
        state_tensor = states[i]
        state_tensor_shape = state_tensor.shape
        for j,npos in enumerate(expected_shape):
            state_tensor_dim_length = state_tensor_shape[j]
            state_tensor = torch.ops.aten.slice(state_tensor,dim=j,start=state_tensor_dim_length-npos,end=state_tensor_dim_length)
        sliced_state_tensors.append(state_tensor)
    
    return sliced_state_tensors
  
  def get_state_preprocessor():
    sp = torch.jit.script(state_preprocessor)
    return sp

In this example, we take in ``shapes_collection``, ``states``, and ``bucket_idx_tensor``. The input ``shapes_collection`` is essentially a list of expected shapes for each state tensor defined for each bucket kernel. For example, we can have ``shapes_collection = [[[1,128],[1,128]],[[1,512],[1,512]]]`` where ``shapes_collection[0][1]`` retrieves the expected shape for the second state tensor in the first bucket. The input ``states`` is the actual list of tensors in the shared buffer, which contains tensors of the largest shape. Finally, ``bucket_idx_tensor`` is the same tensor returned by the bucket kernel.

Two things to note is that we use two aten functions directly: ``aten::Int`` to convert the ``bucket_idx_tensor`` to an integer, and ``aten::slice`` to perform slicing given non-const or non-literal parameters.

.. note::

    The above shared state function is not used in the BERT example

Bucket Model Config
-------------------

Given the above two examples, we can initialize a :class:`torch_neuronx.BucketModelConfig` object as follows:

.. code-block:: python

  import torch
  import torch_neuronx

  from typing import List

  # above code

  bucket_config = torch_neuronx.BucketModelConfig(get_bucket_kernel,shared_state_buffer_preprocessor=get_state_preprocessor)


Putting it all Together
-----------------------

Here is a simple example using the BERT model:

.. code-block:: python

  import torch
  import torch_neuronx

  from transformers import AutoTokenizer, AutoModelForSequenceClassification

  from typing import List

  def encode(tokenizer, *inputs, max_length=128, batch_size=1):
      tokens = tokenizer.encode_plus(
          *inputs,
          max_length=max_length,
          padding='max_length',
          truncation=True,
          return_tensors="pt"
      )
      return (
          torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
          torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
      )

  def get_bert_model(*args):
      name = "bert-base-cased-finetuned-mrpc"
      model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)

      return model,{}

  def sequence_length_bucket_kernel(tensor_list: List[torch.Tensor]):
      x = tensor_list[0]
      bucket_dim = 1
      x_shape = x.shape
      tensor_sequence_length = x_shape[bucket_dim]
      batch_size = x_shape[bucket_dim - 1]
      buckets = [128, 512]
      idx = 0
      num_inputs = 3
      bucket = buckets[0]
      reshaped_tensors: List[torch.Tensor] = []
      bucket_idx = 0
      for idx, bucket in enumerate(buckets):
          if tensor_sequence_length <= bucket:
              bucket_idx = idx
              for tensor in tensor_list:
                  if num_inputs == 0:
                      break
                  delta = bucket - tensor_sequence_length
                  padding_shape: List[int] = [batch_size, delta]
                  zeros = torch.zeros(padding_shape, dtype=x.dtype)
                  reshaped_tensors.append(torch.cat([tensor, zeros], dim=bucket_dim))
                  num_inputs -= 1
              break
      return reshaped_tensors, torch.tensor([bucket_idx])

  def get_bucket_kernel(*_):
      bk = torch.jit.script(sequence_length_bucket_kernel)
      return bk
  
  if __name__ == '__main__':

      name = "bert-base-cased-finetuned-mrpc"

      # Build tokenizer and model
      tokenizer = AutoTokenizer.from_pretrained(name)
      model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)

      # Setup some example inputs
      sequence_0 = "The company HuggingFace is based in New York City"
      sequence_1 = "HuggingFace is named after the huggingface emoji"
      sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

      paraphrase_s128 = encode(tokenizer, sequence_0, sequence_2)
      paraphrase_s122 = encode(tokenizer, sequence_0, sequence_2, max_length=122)
      
      paraphrase_s512 = encode(tokenizer, sequence_0, sequence_1, max_length=512)
      paraphrase_s444 = encode(tokenizer, sequence_0, sequence_1, max_length=444)

      # Note: Run on CPU before trace. Avoids running with XLA allocated params
      paraphrase_expected_s128 = torch.argmax(model(*paraphrase_s128)[0])
      paraphrase_expected_s512 = torch.argmax(model(*paraphrase_s512)[0])
      

      # Trace model
      bucket_config = torch_neuronx.BucketModelConfig(get_bucket_kernel)
      bucket_trace_neuron = torch_neuronx.bucket_model_trace(get_bert_model, [paraphrase_s128,paraphrase_s512], bucket_config)

      # Run traced model with shorter inputs to test bucket rounding
      paraphrase_actual_s128 = torch.argmax(bucket_trace_neuron(*paraphrase_s122)[0])
      paraphrase_actual_s512 = torch.argmax(bucket_trace_neuron(*paraphrase_s444)[0])
      

      # Compare outputs
      assert paraphrase_expected_s128 == paraphrase_actual_s128
      assert paraphrase_expected_s512 == paraphrase_actual_s512


Autobucketing for Neuronx-Distributed
-------------------------------------

To see this same example applied on Neuronx-Distributed, go to this section on the :ref:`Neuronx-Distributed Inference Developer Guide <nxd-inference-devguide-autobucketing>`