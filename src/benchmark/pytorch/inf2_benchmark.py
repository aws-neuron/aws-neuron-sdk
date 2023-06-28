# primary Script used for inf2 Benchmarking

import torch
import neuronperf
import neuronperf.torch
import torch_neuronx
from transformers import (
    AutoModel, AutoModelForSequenceClassification  # Any other model class respective to the model we want to infer on
)

def benchmark(model_name, batch_size, sequence_length):
    model = AutoModel.from_pretrained(model_name, torchscript=True)
    model.eval()

    example = (
        torch.zeros(batch_size, sequence_length, dtype=torch.int),  # input_ids
        torch.zeros(batch_size, sequence_length, dtype=torch.int),  # attention_mask
    )

    traced = torch_neuronx.trace(model, example)
    filename = 'model.pt'
    torch.jit.save(traced, filename)
    reports = neuronperf.torch.benchmark(filename, [example])
    # View and save results
    print("======== {} ========".format(filename))
    neuronperf.print_reports(reports)
    neuronperf.write_csv(reports)
    neuronperf.write_json(reports)

if __name__ == '__main__':
    # benchmark(model_name, batch_size, sequence_length)
    # Below are a few examples -
    # benchmark('bert-base-cased', 16, 128)
    # benchmark('bert-base-uncased', 4, 128)
    # benchmark('gpt2', 16, 256)
