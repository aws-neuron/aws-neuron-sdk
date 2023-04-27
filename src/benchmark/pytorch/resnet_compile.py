import torch
import torchvision
import neuronperf as npf
import neuronperf.torch

# Add to these lists or change as needed
model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
batch_sizes = [1, 8, 64]
pipeline_sizes = [1]


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)


if __name__ == "__main__":
    for model_name in model_names:
        model = getattr(torchvision.models, model_name)(pretrained=True)
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