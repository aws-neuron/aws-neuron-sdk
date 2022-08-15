import torch
import torch.neuron

import neuronperf as npf
import neuronperf.torch

# Add to these lists or change as needed
model_name = "resnet50"
batch_sizes = [1, 6]


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)


if __name__ == "__main__":
    inputs = [get_batch(batch_size) for batch_size in batch_sizes]
    filename = f"{model_name}.json"

    # Benchmark
    print("Benchmarking {}".format(filename))
    reports = npf.torch.benchmark(filename, inputs)

    # View and save results
    print("======== {} ========".format(filename))
    npf.print_reports(reports)
    npf.write_csv(reports)
    npf.write_json(reports)
