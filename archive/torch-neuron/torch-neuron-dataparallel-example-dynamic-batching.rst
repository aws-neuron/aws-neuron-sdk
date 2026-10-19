.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11


.. warning::

   This document is archived. torch-neuron (Inf1) is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/archive/frameworks/index`.

In the following example, we use the :func:`torch.neuron.DataParallel` module
to run inference using several different batch sizes without recompiling the
Neuron model.

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

    # Create the DataParallel module
    model_parallel = torch.neuron.DataParallel(model_neuron)

    # Create batched inputs and run inference on the same model
    batch_sizes = [2, 3, 4, 5, 6]
    for batch_size in batch_sizes:
        image_batched = torch.rand([batch_size, 3, 224, 224])

        # Run inference with a batched input
        output = model_parallel(image_batched)
