The following example uses the ``device_ids`` argument to use the first three
NeuronCores for DataParallel inference.

.. code-block:: python

    import torch
    import torch_neuron
    from torchvision import models

    # Load the model and set it to evaluation mode
    model = models.resnet50(pretrained=True)
    model.eval()

    # Compile with an example input
    image = torch.rand([1, 3, 224, 224])
    model_neuron = torch.neuron.trace(model, image)

    # Create the DataParallel module, run on the first three NeuronCores
    # Equivalent to model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=[0, 1, 2])
    model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=['nc:0', 'nc:1', 'nc:2'])

    # Create a batched input
    batch_size = 5
    image_batched = torch.rand([batch_size, 3, 224, 224])

    # Run inference with a batched input
    output = model_parallel(image_batched)