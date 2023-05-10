import os
import neuronperf as npf
import torch
from transformers import AutoTokenizer

"""
Run the sample at this link to get the split model state_dict (opt-13b-split):
https://github.com/aws-neuron/aws-neuron-samples-staging/blob/master/torch-neuronx/transformers-neuronx/inference/facebook-opt-13b-sampling.ipynb

Make sure transformers is installed

Change the variables below for opt30b or opt66b models
"""


BATCH_SIZE = 2
TP_DEGREE = 2
SEQ_LEN = 2048
TOKENIZER = AutoTokenizer.from_pretrained("facebook/opt-13b")
MODEL_DIR = "./opt-13b-split"


class Wrapper(torch.nn.Module):
    def __init__(self, filename):
        super().__init__()
        from transformers_neuronx.opt.model import OPTForSampling
        self.neuron_model = OPTForSampling.from_pretrained(
            filename, batch_size=BATCH_SIZE, tp_degree=TP_DEGREE, amp="f16"
        )
        self.neuron_model.to_neuron()

    def forward(self, *inputs):
        return self.neuron_model.sample(torch.concat(inputs), sequence_length=SEQ_LEN)

# Custom load to let our Wrapper class handle things
def load_fn(filename, **kwargs):
    return Wrapper(filename)

# NeuronPerf can't see tp_degree at the moment, so just expose all cores
def env_setup_fn(*_):
    del os.environ["NEURON_RT_VISIBLE_CORES"]

def preprocess_fn(inputs):
    return [TOKENIZER.encode(text, return_tensors="pt") for text in inputs]

def postprocess_fn(outputs):
    return [TOKENIZER.decode(seq) for seq in outputs]

def benchmark():
    inputs = ["Hello, I'm a language model,"] * BATCH_SIZE
    reports = npf.benchmark(
        load_fn,
        MODEL_DIR,
        [inputs],  # treat batch as 1 input and let Wrapper handle batching
        batch_sizes=1,  # ^
        n_models=1,  # only load 1 copy of model
        max_infers=5,
        max_duration=0,  # sampling can take a while, so let's not timeout
        workers_per_model=1,  # no bottleneck on model inputs, so 1 is fine
        env_setup_fn=env_setup_fn,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
    )
    
    # grab the only report (we only benchmarked 1 config)
    report = reports[0]
    
    # let's update throughput to be tokens / second and add a new record
    new_tokens = sum(SEQ_LEN - len(TOKENIZER.encode(i)) for i in inputs)
    tokens_per_s = round(new_tokens / (report["latency_ms_avg"] / 1000), 2)
    report["throughput_avg"] = report["tokens_per_s"] = tokens_per_s
    
    # display and save results
    npf.print_report(report)
    print(f"Results saved to: {npf.write_json(report)}")


if __name__ == "__main__":
    benchmark()
