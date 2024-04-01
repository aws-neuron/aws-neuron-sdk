.. _neuronx_distributed_inference_developer_guide:

Developer guide for Neuronx-Distributed  Inference (``neuronx-distributed`` )
=================================================================

Overview
^^^^^^^^

Neuronx-Distributed started with mostly targeting distributed device training workloads.
Now, Neuronx-Distributed is now quickly expanding to support distributed device inference workloads. 
Currently, Tensor Parallelism (TP) is the only supported form of parallelism for Neuronx-Distributed,
with other forms such as Pipeline Parallelism coming in future releases.
Beyond this, Neuronx-Distributed inference also supports weight separation amongst TP shards, as well as
autobucketing support for TP models.
These will be covered in this Developer Guide using BERT, and in the end, there will 
be two samples (T5 3B and Llama-v2 7B) that showcase Neuronx-Distributed inference for larger models.

For training workflows, check out the other written Developer Guides for Neuronx-Distributed.

Pre-Requisites
^^^^^^^^^^^^^^

Before we start, let's install transformers.

.. code:: ipython3

    pip install transformers==4.26.0

For this guide we'll use BERT. Before we run the inference,
let’s get a checkpoint that we can use.

Let’s run the below block of code:

.. code:: ipython3

    import torch
    import torch_neuronx
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    name = "bert-base-cased-finetuned-mrpc"

    model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
    torch.save({"model":model.state_dict()}, "bert.pt")

Creating a Tensor Parallel (TP) Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TP models are created by introducing layers that are built to utilize TP,
such as ``RowParallelLinear`` and ``ColumnParallelLinear``. To see how these
layers work, please see the :ref:`Tensor Parallel Developer Guide <tp_developer_guide>`.

Below is an example using BERT:

.. code:: ipython3

    import os
    import torch
    import torch_neuronx
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput

    import neuronx_distributed
    from neuronx_distributed.parallel_layers import layers, parallel_state


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
            torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
        )


    # Create the tokenizer and model
    name = "bert-base-cased-finetuned-mrpc"
    tokenizer = AutoTokenizer.from_pretrained(name)


    # Set up some example inputs
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    paraphrase = encode(tokenizer, sequence_1, sequence_2)
    not_paraphrase = encode(tokenizer, sequence_1, sequence_1)

    def get_model():
        model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
        # Here we build a model with tensor-parallel layers.
        # Note: If you already have a Model class that does this, we can use that directly
        # and load the checkpoint in it.
        class ParallelSelfAttention(BertSelfAttention):
            def __init__(self, config, position_embedding_type=None):
                super().__init__(config, position_embedding_type)
                self.query = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
                self.key = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
                self.value = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
                self.num_attention_heads = self.num_attention_heads // parallel_state.get_tensor_model_parallel_size()
                self.all_head_size = self.all_head_size // parallel_state.get_tensor_model_parallel_size()

        class ParallelSelfOutput(BertSelfOutput):
            def __init__(self, config):
                super().__init__(config)
                self.dense = layers.RowParallelLinear(config.hidden_size,
                                        config.hidden_size,
                                        input_is_parallel=True)

        for layer in model.bert.encoder.layer:
            layer.attention.self = ParallelSelfAttention(model.config)
            layer.attention.output = ParallelSelfOutput(model.config)

        # Here we created a checkpoint as mentioned above. We pass sharded=False, since the checkpoint
        # we obtained is unsharded. In case you are using the checkpoint from the tensor-parallel training,
        # you can set the sharded=True, as that checkpoint will contain shards from each tp rank.
        neuronx_distributed.parallel_layers.load("bert.pt", model, sharded=False)

        # These io aliases would enable us to mark certain input tensors as state tensors. These
        # state tensors are going to be device tensors.
        io_aliases = {}
        return model, io_aliases

Notice that the ``get_model()`` function returns not only the model, but also ``io_aliases``. This is a
dictionary containing model tensors that are marked as containing state. This is necessary for models
that have dynamic tensors during each inference pass. One such use case is for models with KV-Caching,
which can be seen in the T5 and Llama-v2 samples linked at the bottom of the guide.
In this example, we don't have such tensors, so we return an empty dictionary.

Tracing the Tensor Parallel (TP) Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After introducing these layers to the model, we need to trace the model
for inference. This is done by the ``parallel_model_trace`` API. This
will produce model shards per tp degree, and is saved and loaded by
custom Neuronx-Distributed APIs: ``parallel_model_load`` and ``parallel_model_save``.

``parallel_model_trace`` has a few distinctions from ``torch_neuronx.trace``. First,
instead of passing in a model directly, we pass in a function that returns the model
and a dictionary of states. This is done for serialization purposes when tracing using
XLA multiprocessing as is done in ``parallel_model_trace``. Another difference is the
keyword arguments unique to ``parallel_model_trace``. The most important one is the
``tp_degree``, which determines the number of model shards to produce in a TP scheme.

Below code shows the earlier written ``get_model()`` function used in ``parallel_model_trace``, as well as
saving and loading the traced tp model:

.. code:: ipython3

    if __name__ == "__main__":

        # Note how we are passing a function that returns a model object, which needs to be traced.
        # This is mainly done, since the model initialization needs to happen within the processes
        # that get launched internally within the parallel_model_trace.
        model = neuronx_distributed.trace.parallel_model_trace(get_model, paraphrase, tp_degree=2)

        # Once traced, we now save the trace model for future inference. This API takes care
        # of saving the checkpoint from each tensor parallel worker
        neuronx_distributed.trace.parallel_model_save(model, "tp_models")

        # We now load the saved model and will run inference against it
        model = neuronx_distributed.trace.parallel_model_load("tp_models")
        cpu_model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
        assert torch.argmax(model(*paraphrase)[0]) == torch.argmax(cpu_model(*paraphrase)[0])


Weight separation
^^^^^^^^^^^^^^^^^^^^

One more difference to note is the ``inline_weights_to_neff`` keyword argument. While
this also exists in ``torch_neuronx.trace`` it's important to note that since
``parallel_model_trace`` produces many NEFFs, this means that this keyword argument
enables weight separation, which is done by separating out common weights between
the shards from the NEFFs. Benefits that can come from weight separation is lower
memory usage, as well as faster neff loading times.

.. note::
    It might be confusing to enable weight separation by disabling a flag. This is because
    the original way that Neuron models handle weights is by having the weights embedded/inlined
    into the NEFF, making it impossible to replace. To preserve default behavior, the flag is
    set to ``True`` by default. When the flag is set to ``False``, weights are no longer inlined into
    the neff and are now separate, which enables new workflows.

To enable weight separation, set ``inline_weights_to_neff=False`` in ``parallel_model_trace``:

.. code:: ipython3

    model = neuronx_distributed.trace.parallel_model_trace(get_model, paraphrase, tp_degree=2, inline_weights_to_neff=False)

The full API reference for all trace related functions can be found :ref:`here <nxd_tracing>`.

.. _nxd-inference-devguide-autobucketing:

Autobucketing
^^^^^^^^^^^^^

Autobucketing is a feature that enables you to use multiple bucket models. Each bucket model accepts a static input shape and a bucket kernel function. The models are then packaged into a single traced PyTorch model that can accept multiple different input shapes. 

This gives you increased flexibility for inputs into Neuron models without the need to manage multiple Neuron models. The applications of this are extensive, from optimal model selection based on image resolution, to efficient sampling for token generation in language models.
For more information, see the torch_neuronx section on :ref:`Autobucketing <torch-neuronx-autobucketing>`, and this :ref:`developer guide<torch-neuronx-autobucketing-devguide>`.

``neuronx_distributed`` supports autobucketing via the ``bucket_config`` parameter. The following example shows how to use this with BERT to bucket it on sequence length:

.. code:: python

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
    
    # same encode function
    paraphrase = encode(tokenizer, sequence_1, sequence_2)
    paraphrase_long = encode(tokenizer, sequence_1, sequence_2,max_length=512)
    
    if __name__ == '__main__':
        #same as original main function

        bucket_config = torch_neuronx.BucketModelConfig(get_bucket_kernel)

        # note: inline_weights_to_neff must be set to False, otherwise a ValueError is raised
        model = neuronx_distributed.trace.parallel_model_trace(get_model, [paraphrase, paraphrase_long], inline_weights_to_neff=False, bucket_config=bucket_config, tp_degree=2)

        #rest is the same

With the above example, we can supply inputs of sequence length from 1-512 without pre-padding, as the bucket kernel takes care of that. Autobucketing is useful for latency sensitive applications where using smaller and large inputs on small and large models respectively.

.. note::
    We do not yet have autobucketing integrated with our NxD Llama2 example, and
    will be done so in an upcoming release.


Conclusion
^^^^^^^^^^

Neuronx-Distributed inference is quickly expanding to support more features, and this guide will be updated to reflect these features. However,
Neuronx-Distributed inference already supports some large models such as T5 3B and Llama-v2 7B. The samples for each can be found:

1. T5 3B inference tutorial :ref:`[html] </src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`
2. Llama-v2 7B tutorial :ref:`[html] <src/examples/pytorch/neuronx_distributed/llama/llama2_inference.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/llama/llama2_inference.ipynb>`