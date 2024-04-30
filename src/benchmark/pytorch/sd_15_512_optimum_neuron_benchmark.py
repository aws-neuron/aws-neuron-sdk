import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
import time
import math

from optimum.neuron import NeuronStableDiffusionPipeline

# Specialized benchmarking class for stable diffusion.
def benchmark(n_runs, test_name, model, model_inputs):
    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs,)
    
    warmup_run = model(*model_inputs)

    latency_collector = LatencyCollector()
    # can't use register_forward_pre_hook or register_forward_hook because StableDiffusionPipeline is not a torch.nn.Module
    
    for _ in range(n_runs):
        latency_collector.pre_hook()
        res = model(*model_inputs)
        latency_collector.hook()
    
    p0_latency_ms = latency_collector.percentile(0) * 1000
    p50_latency_ms = latency_collector.percentile(50) * 1000
    p90_latency_ms = latency_collector.percentile(90) * 1000
    p95_latency_ms = latency_collector.percentile(95) * 1000
    p99_latency_ms = latency_collector.percentile(99) * 1000
    p100_latency_ms = latency_collector.percentile(100) * 1000

    report_dict = dict()
    report_dict["Latency P0"] = f'{p0_latency_ms:.1f}'
    report_dict["Latency P50"]=f'{p50_latency_ms:.1f}'
    report_dict["Latency P90"]=f'{p90_latency_ms:.1f}'
    report_dict["Latency P95"]=f'{p95_latency_ms:.1f}'
    report_dict["Latency P99"]=f'{p99_latency_ms:.1f}'
    report_dict["Latency P100"]=f'{p100_latency_ms:.1f}'

    report = f'RESULT FOR {test_name}:'
    for key, value in report_dict.items():
        report += f' {key}={value}'
    print(report)

class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)

    def percentile(self, percent):
        latency_list = self.latency_list
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        latency_list = sorted(latency_list)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]

# # For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'sd_1_5_fp32_512_compile_workdir'
pipe = NeuronStableDiffusionPipeline.from_pretrained(COMPILER_WORKDIR_ROOT)

prompt = "a photo of an astronaut riding a horse on mars"
n_runs = 20
benchmark(n_runs, "stable_diffusion_15_512", pipe, prompt)
