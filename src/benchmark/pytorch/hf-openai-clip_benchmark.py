import torch
import neuronperf
import neuronperf.torch
import torch_neuronx
import os

from torchvision.datasets import CIFAR100
from transformers import CLIPProcessor, CLIPModel

def benchmark(model_name, batch_size):
    # Build the model, preprocessor, and dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, return_dict=False)

    # Prepare a sample input
    image = cifar100[0][0]
    text = []
    for c in cifar100.classes:
        text.append(f'a photo of a {c}')

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    image = inputs['pixel_values']
    # (b, c, h, w)
    image = image.repeat(batch_size, 1, 1, 1)
    inputs = (inputs['input_ids'], image)

    # Trace the model
    model.eval()
    traced = torch_neuronx.trace(model, inputs, compiler_args='--enable-saturate-infinity')
    filename = 'model.pt'
    torch.jit.save(traced, filename)
    reports = neuronperf.torch.benchmark(filename, [inputs], batch_sizes=[batch_size])
    # View and save results
    print("======== {} ========".format(filename))
    neuronperf.print_reports(reports)
    neuronperf.write_csv(reports)
    neuronperf.write_json(reports)

if __name__ == '__main__':
    # Recommended batch sizes for throughput
    # openai/clip-vit-base-patch32: 64
    # openai/clip-vit-large-patch14: 4
    model_name = 'openai/clip-vit-base-patch32'
    batch_size = 64
    benchmark(model_name, batch_size)