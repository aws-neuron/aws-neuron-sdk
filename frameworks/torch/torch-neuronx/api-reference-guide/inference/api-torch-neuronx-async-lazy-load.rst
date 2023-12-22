.. _torch_neuronx_lazy_async_load_api:

PyTorch NeuronX Lazy and Asynchronous Loading API
===================================================

The :func:`torch_neuronx.lazy_load` and :func:`torch_neuronx.async_load` Python APIs allow
for more fine-grained control of loading a model onto the Neuron cores. They are designed to
enable different load behaviours (i.e. lazy or asynchronous loading) that, in certain cases, 
can speed up the load time. Both APIs take as input a :class:`~torch.jit.ScriptModule` model
created by :ref:`torch_neuronx_trace_api`. **They should be called immediately after** :func:`torch_neuronx.trace`
**returns, before saving the model via** :func:`torch.jit.save`

.. py:function:: torch_neuronx.lazy_load(trace, enable_lazy_load=True)

    Enables(or disables) lazy load behaviour on the traced Neuron ScriptModule ``trace``.
    By default, lazy load behaviour is disabled, so this API must be called immediately after
    :func:`torch_neuronx.trace` returns if lazy load behaviour is desired.

    In this context, lazy loading means that **calling** ``torch.jit.load`` **will not immediately load
    the model onto the Neuron core.** Instead, the model will be loaded onto the Neuron core at a later
    time, either via a call to :ref:`torch_neuronx_dataparallel_api`, or automatically when the model's
    ``forward`` method executes.

    There are several scenarios where lazy loading is useful. For instance, if one wants to use
    the DataParallel API to load the model onto multiple Neuron cores, typically
    one would first call ``torch.jit.load`` to load the saved model from disk, and then call ``DataParallel``
    on the object returned by ``torch.jit.load``. Doing this will cause redundant loading, because calling ``torch.jit.load``
    first will by default load the model onto one Neuron core, while calling ``DataParallel`` next will
    first unload the model from the Neuron core, and then load again according to user-specified ``device_ids``.
    This redundant load is avoided if one enables lazy loading by calling ``torch_neuronx.lazy_load`` prior to saving
    the model. This way, ``torch.jit.load`` will not load the model onto the Neuron core, so ``DataParallel`` can 
    directly load the model onto the desired cores.

    *Required Arguments*

    :arg ~torch.jit.ScriptModule trace: Model created by the
        :ref:`torch_neuronx_trace_api`, for which lazy loading is to be enabled.

    *Optional Arguments*

    :arg bool enable_lazy_load: Whether to enable lazy loading, defaults to True.

    Simple example usage:

        >>> neuron_model = torch_neuronx.trace(model, inputs)
        >>> torch_neuronx.lazy_load(neuron_model)
        >>> torch.jit.save(neuron_model, "my_model")
        
        Then some time later:

        >>> neuron_model = torch.jit.load("my_model") # neuron_model will not be loaded onto the Neuron core until it is run or it is passed to DataParallel

.. py:function:: torch_neuronx.async_load(trace, enable_async_load=True)
    
    Enables(or disables) asynchronous load behaviour on the traced Neuron ScriptModule ``trace``.
    
    By default, loading onto the Neuron core is a synchronous, blocking operation. This API
    can be called immediately after :func:`torch_neuronx.trace` returns in order to make
    loading this model onto the Neuron core a non-blocking operation. This means that when
    a load onto the Neuron core is triggered, either through a call to ``torch.jit.load`` or
    ``DataParallel``, a new thread is launched to perform the load, while the calling function
    will immediately return. The load will proceed asynchronously in the background, and only
    when it finishes will the model successfully execute. If the model's ``forward`` method is invoked
    before the asynchronus load finishes, ``forward`` will wait until the load completes before
    executing the model.

    This API is useful when one wants to load multiple models onto the Neuron core in parallel.
    It allows multiple calls to load different models to execute concurrently on different threads,
    which can significantly reduce the total load time when there are multiple CPU cores on the host.
    It is especially useful in cases where a single model pipeline has several compiled Neuron models.
    In this case, one can enable asynchronous load on each Neuron model and load all of them in parallel.

    Note that this API differs from :func:`torch_neuronx.lazy_load`. Lazy loading will
    only delay the load onto the Neuron core from when ``torch.jit.load`` is called to some later time, 
    but when the load does occur, it is still a synchronous, blocking operation. Asynchronous loading
    will make the load an asynchronous, non-blocking operation, but it does not delay when the load starts,
    meaning that calling ``torch.jit.load`` will still start the load, but the load will proceed asynchronously
    in the background.

    *Required Arguments*

    :arg ~torch.jit.ScriptModule trace: Model created by the
        :ref:`torch_neuronx_trace_api`, for which asynchronous loading is to be enabled.

    *Optional Arguments*

    :arg bool enable_async_load: Whether to enable asynchronous loading, defaults to True.

    Simple example usage:

        >>> neuron_model1 = torch_neuronx.trace(model1, inputs1)
        >>> torch_neuronx.async_load(neuron_model1)
        >>> torch.jit.save(neuron_model1, "my_model1")

        >>> neuron_model2 = torch_neuronx.trace(model2, inputs2)
        >>> torch_neuronx.async_load(neuron_model2)
        >>> torch.jit.save(neuron_model2, "my_model2")
        
        Then some time later:

        >>> neuron_model1 = torch.jit.load("my_model1") # neuron_model1 will start loading onto the Neuron core immediately, but the load will occur in a separate thread in the background.
        >>> neuron_model2 = torch.jit.load("my_model2") # neuron_model2 will start loading onto the Neuron core immediately, but the load will occur in a separate thread in the background.

        Both neuron_model1 and neuron_model2 will load concurrently.
        
        >>> output1 = neuron_model1(input1) # This call will block until the asynchronous load launched above finishes.
        >>> output2 = neuron_model2(input2) # This call will block until the asynchronous load launched above finishes.


Using :func:`torch_neuronx.lazy_load` and :func:`torch_neuronx.async_load` Together
--------

You can also enable lazy load and asynchronous load together for the same model.
To do so, simply call each API independently before saving the model with ``torch.jit.save``:

    >>> neuron_model = torch_neuronx.trace(model, inputs)
    >>> torch_neuronx.lazy_load(neuron_model)
    >>> torch_neuronx.async_load(neuron_model)
    >>> torch.jit.save(neuron_model, "my_model")

This will both delay loading the model onto the Neuron core, and make the load asynchronous.

For another example usage, please refer to the `Github sample <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd2_512_inference.ipynb>`_ we provide for running inference on HuggingFace Stable Diffusion 2.1,
where we use both ``lazy_load`` and ``async_load`` to speed up the total load time of the four Neuron models that make 
up that pipeline.
