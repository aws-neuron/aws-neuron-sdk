import torch
import torch.neuron

import neuronperf
import neuronperf.torch


# Define a simple model
class Model(torch.nn.Module):
    def forward(self, x):
        x = x * 3
        return x + 1


# Instantiate
model = Model()
model.eval()

# Define some inputs
batch_sizes = [1]
inputs = [torch.ones((batch_size, 3, 224, 224)) for batch_size in batch_sizes]

# Compile for Neuron
model_neuron = torch.neuron.trace(model, inputs)
model_neuron.save("model_neuron_b1.pt")

# Benchmark
reports = neuronperf.torch.benchmark("model_neuron_b1.pt", inputs, batch_sizes)

# View and save results
neuronperf.print_reports(reports)
neuronperf.write_csv(reports, "model_neuron_b1.csv")
