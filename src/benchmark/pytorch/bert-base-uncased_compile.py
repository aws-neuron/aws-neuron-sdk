import torch
import torch.neuron

import neuronperf
import neuronperf.torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Add to these lists or change as needed
model_names = ["bert-base-uncased"]
sequence_lengths = [128]
batch_sizes = [6]
pipeline_sizes = [1]


def get_batch(tokenizer, sequence_length, batch_size):
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
    paraphrase = tokenizer.encode_plus(
        sequence_0,
        sequence_1,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        torch.cat([paraphrase["input_ids"]] * batch_size, 0),
        torch.cat([paraphrase["attention_mask"]] * batch_size, 0),
    )
    return inputs


if __name__ == "__main__":
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)
        for sequence_length in sequence_lengths:
            inputs = [
                get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes
            ]
            filename = f"{model_name}_sl{sequence_length}.json"

            # Compile
            print("Compiling {}".format(filename))
            neuronperf.torch.compile(
                model,
                inputs,
                batch_sizes=batch_sizes,
                pipeline_sizes=pipeline_sizes,
                filename=filename,
                model_name=model_name,
            )
