In the following example, we use
:func:`torch_neuronx.DataParallel.disable_dynamic_batching` to disable dynamic
batching. We provide an example of a batch size that will not work when dynamic
batching is disabled as well as an example of a batch size that does work when
dynamic batching is disabled.

.. code-block:: python

    import torch
    import torch_neuronx
    from torchvision import models

    # Load the model and set it to evaluation mode
    model = models.resnet50(pretrained=True)
    model.eval()

    # Compile with an example input
    image = torch.rand([1, 3, 224, 224])
    model_neuron = torch_neuronx.trace(model, image)

    # Create the DataParallel module and use 2 NeuronCores
    model_parallel = torch_neuronx.DataParallel(model_neuron, device_ids=[0, 1], dim=0)

    # Disable dynamic batching
    model_parallel.disable_dynamic_batching()

    # Create a batched input (this won't work)
    batch_size = 4
    image_batched = torch.rand([batch_size, 3, 224, 224])

    # This will fail because dynamic batching is disabled and
    # image_batched.shape[dim] / len(device_ids) != image.shape[dim]
    # output = model_parallel(image_batched)

    # Create a batched input (this will work)
    batch_size = 2
    image_batched = torch.rand([batch_size, 3, 224, 224])

    # This will work because
    # image_batched.shape[dim] / len(device_ids) == image.shape[dim]
    output = model_parallel(image_batched)
