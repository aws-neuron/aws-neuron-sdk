# Add to these lists or change as needed
model_names = ["distilbert-base-uncased-finetuned-sst-2-english"]
sequence_lengths = [128]
batch_sizes = [128]
pipeline_sizes = [1]

# Silence an irrelevant warning from transformers library
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import neuronperf as npf
import neuronperf.tensorflow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


def get_batch(tokenizer, sequence_length, batch_size):
    sequence = "I am sorry. I really want to like it, but I just can not stand sushi."
    paraphrase = tokenizer.encode_plus(
        sequence,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    inputs = {
        "input_ids": np.concatenate([paraphrase["input_ids"]] * batch_size, axis=0),
        "attention_mask": np.concatenate([paraphrase["attention_mask"]] * batch_size, axis=0),
    }
    return inputs


if __name__ == "__main__":
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)
        for sequence_length in sequence_lengths:
            inputs = [
                get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes
            ]
            filename = f"{model_name}_sl{sequence_length}.json"

            # Compile
            print("Compiling {}".format(filename))
            npf.tensorflow.compile(
                model,
                inputs,
                batch_sizes=batch_sizes,
                pipeline_sizes=pipeline_sizes,
                filename=filename,
                model_name=model_name,
            )
