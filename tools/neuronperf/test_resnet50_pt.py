import torch
import torch_neuron

import neuronperf as npf
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
npf.torch.compile(
	model, 
	inputs, 
	batch_sizes=batch_sizes, 
	filename=filename,
)

# Benchmark
reports = npf.torch.benchmark(filename, inputs)

# View and save results
npf.print_reports(reports)
npf.write_csv(reports, 'resnet50_results.csv')
npf.write_json(reports, 'resnet50_results.json')
