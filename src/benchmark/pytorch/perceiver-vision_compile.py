import torch
import transformers  # ==4.32.0
import neuronperf as npf
import neuronperf.torch

# Add to these lists or change as needed
models_list = [
    ("PerceiverForImageClassificationLearned", "deepmind/vision-perceiver-learned"),
    ("PerceiverForImageClassificationFourier", "deepmind/vision-perceiver-fourier"),
    ("PerceiverForImageClassificationConvProcessing", "deepmind/vision-perceiver-conv"),
]
batch_sizes = [1]
pipeline_sizes = [1]


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)


if __name__ == "__main__":
    for class_name, pretrained_name in models_list:
        model_name = pretrained_name.split("/")[1]

        model = getattr(transformers, class_name).from_pretrained(pretrained_name)
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