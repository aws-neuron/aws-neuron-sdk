import torch
import torch_neuron

import neuronperf
import neuronperf.torch

from torchvision import models


# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Select a few batch sizes to test
filename = 'resnet50.json'
batch_sizes = [5, 6, 7]

# Construct example inputs
inputs = [torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32) for batch_size in batch_sizes]

# Compile
neuronperf.torch.compile(
	model, 
	inputs, 
	batch_sizes=batch_sizes, 
	filename=filename,
)

# Benchmark
reports = neuronperf.torch.benchmark(filename, inputs)

# View and save results
neuronperf.print_reports(reports)
neuronperf.write_csv(reports, 'resnet50_results.csv')
neuronperf.write_json(reports, 'resnet50_results.json')
