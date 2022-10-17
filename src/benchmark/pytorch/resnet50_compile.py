import torch
import torch.neuron
import torchvision

import neuronperf as npf
import neuronperf.torch

# Add to these lists or change as needed
model_name = "resnet50"
batch_sizes = [1, 6]
pipeline_sizes = [1]


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)


if __name__ == "__main__":
    model = torchvision.models.resnet50(pretrained=True)
    inputs = [get_batch(batch_size) for batch_size in batch_sizes]
    filename = f"{model_name}.json"

    # Compile
    print("Compiling {}".format(filename))
    npf.torch.compile(
        model,
        inputs,
        batch_sizes=batch_sizes,
        pipeline_sizes=pipeline_sizes,
        filename=filename,
        model_name=model_name,
    )
