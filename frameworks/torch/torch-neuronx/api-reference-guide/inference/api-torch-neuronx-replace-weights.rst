.. _torch_neuronx_replace_weights_api:

PyTorch Neuron (``torch-neuronx``) Weight Replacement API for Inference
========================================================================

.. py:function:: torch_neuronx.replace_weights(neuron_model, weights)

    Replaces the weights in a Neuron Model with split weights.
    This function will emit a warning of the supplied Neuron model does not
    contain any separated weights.

    .. warning::

        The below API is only applicable for models traced with the
        parameter ``inline_weights_to_neff=False``, which is ``True`` by
        default. See :func:`torch_neuronx.trace` for details.

    :arg ~torch.jit.RecursiveScriptModule neuron_model: A Neuron model compiled with split weights

    :arg ~torch.nn.Module,Dict[str, ~torch.Tensor] weights: Either the original model with the new weights,
        or the state_dict of a model.
    
    :returns: ``None``, this function performs the weight replacement inline.
    :rtype: ``None``

    .. rubric:: Examples

    *Using a model*

    .. code-block:: python

        import torch
        import torch_neuronx


        class Network(torch.nn.Module):
            def __init__(self, hidden_size=4, layers=3) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    *(torch.nn.Linear(hidden_size, hidden_size) for _ in range(layers)))

            def forward(self, tensor):
                return self.layers(tensor)
    

        # initialize two networks
        network = Network()
        network2 = Network()
        network.eval()
        network2.eval()

        inp = torch.rand(2,4)

        # trace weight separated model with first network
        weight_separated_trace = torch_neuronx.trace(network,inp,inline_weights_to_neff=False)

        # replace with weights from second network
        torch_neuronx.replace_weights(weight_separated_trace,network2.state_dict())

        # get outputs from neuron and cpu networks
        out_network2 = network2(inp)
        out_neuron = weight_separated_trace(inp)
        
        # check that they are equal
        print(out_network2,out_neuron)



    *Using safetensors*

    The `safetensors`_ library is useful for storing/loading model tensors safely and quickly.

    .. code-block:: python

        import torch
        import torch_neuronx

        from safetensors import safe_open
        from safetensors.torch import save_model


        class Network(torch.nn.Module):
            def __init__(self, hidden_size=4, layers=3) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    *(torch.nn.Linear(hidden_size, hidden_size) for _ in range(layers)))

            def forward(self, tensor):
                return self.layers(tensor)
    

        # initialize two networks
        network = Network()
        network2 = Network()
        network.eval()
        network2.eval()

        inp = torch.rand(2,4)

        # trace weight separated model with first network
        weight_separated_trace = torch_neuronx.trace(network,inp,inline_weights_to_neff=False)

        # save network2 weights to safetensors
        safetensor_path = f"{directory}/network2.safetensors"
        save_model(network2,safetensor_path)

        #load safetensors from network2 into traced_weight separated model
        tensors = {}
        with safe_open(safetensor_path,framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        # replace with weights from second network
        torch_neuronx.replace_weights(weight_separated_trace,tensors)

        # get outputs from neuron and cpu networks
        out_network2 = network2(inp)
        out_neuron = weight_separated_trace(inp)
        
        # check that they are equal
        print(out_network2,out_neuron)


.. note::

    For non-safetensors models, use ``torch.load`` to load the model, and pass the model's ``state_dict`` inside like the first example.

.. _safetensors: https://huggingface.co/docs/safetensors/index
.. _torch-xla: https://github.com/pytorch/xla
.. _torchscript: https://pytorch.org/docs/stable/jit.html
