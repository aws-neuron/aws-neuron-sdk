In this example we run DataParallel inference using four NeuronCores and
``dim = 2``. Because ``dim != 0``, dynamic batching is not enabled.
Consequently, the DataParallel inference-time batch size must be four times the
compile-time batch size. DataParallel will generate a warning that dynamic
batching is disabled because ``dim != 0``.

.. code-block:: python

    import torch
    import torch_neuron

    # Create an example model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.conv(x) + 1

    model = Model()
    model.eval()

    # Compile with an example input
    image = torch.rand([1, 3, 8, 8])
    model_neuron = torch.neuron.trace(model, image)

    # Create the DataParallel module using 4 NeuronCores and dim = 2
    model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=[0, 1, 2, 3], dim=2)

    # Create a batched input
    # Note that image_batched.shape[dim] / len(device_ids) == image.shape[dim]
    batch_size = 4 * 8
    image_batched = torch.rand([1, 3, batch_size, 8])

    # Run inference with a batched input
    output = model_parallel(image_batched)