import torch
import neuronperf as npf
import neuronperf.torch

# Add to these lists or change as needed
model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
batch_sizes = [1, 8, 64]
n_models = [1, 2]
workers_per_model = [1, 2] # optimized for latency or throughput


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)


if __name__ == "__main__":
    for model_name in model_names:
        inputs = [get_batch(batch_size) for batch_size in batch_sizes]
        filename = f"{model_name}.json"

        # Benchmark
        print("Benchmarking {}".format(filename))
        reports = npf.torch.benchmark(filename, inputs, n_models=n_models, workers_per_model=workers_per_model) 

        # View and save results
        print("======== {} ========".format(filename))
        npf.print_reports(reports)
        npf.write_csv(reports)
        npf.write_json(reports)
