The default DataParallel use mode will replicate the model
on all available NeuronCores in the current process. The inputs will be split
on ``dim=0``.

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

    # Create the DataParallel module
    model_parallel = torch_neuronx.DataParallel(model_neuron)

    # Create a batched input
    batch_size = 5
    image_batched = torch.rand([batch_size, 3, 224, 224])

    # Run inference with a batched input
    output = model_parallel(image_batched)