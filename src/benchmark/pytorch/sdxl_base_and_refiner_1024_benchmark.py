import os

import torch
import torch.nn as nn
import torch_neuronx

from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

import time
import math

# Define datatype
DTYPE = torch.float32

# Specialized benchmarking class for stable diffusion.
# We cannot use any of the pre-existing benchmarking utilities to benchmark E2E stable diffusion performance,
# because the top-level StableDiffusionPipeline cannot be serialized into a single Torchscript object.
# All of the pre-existing benchmarking utilities (in neuronperf or torch_neuronx) require the model to be a
# traced Torchscript.
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

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
 
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple
    
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
 
    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.to(dtype=DTYPE).expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)
    
# Helper function to run both refiner and base pipes and return the final image
def run_refiner_and_base(base, refiner, prompt, n_steps=40, high_noise_frac=0.8, generator=None):
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator,
    ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    return image
    
    
# --- Load all compiled models and run pipeline ---
COMPILER_WORKDIR_ROOT = 'sdxl_base_and_refiner_compile_dir_1024'
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

unet_base_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet_base/model.pt')
unet_refiner_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet_refiner/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

# ------- Load base -------
pipe_base = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=DTYPE, low_cpu_mem_usage=True)

# Load the compiled UNet onto two neuron cores.
pipe_base.unet = NeuronUNet(UNetWrap(pipe_base.unet))
device_ids = [0,1]
pipe_base.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_base_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe_base.vae.decoder = torch.jit.load(decoder_filename)
pipe_base.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)


# ------- Load refiner -------
# refiner shares text_encoder_2 and vae with the base
pipe_refiner = DiffusionPipeline.from_pretrained(
    refiner_model_id,
    text_encoder_2=pipe_base.text_encoder_2,
    vae=pipe_base.vae,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Refiner - load the compiled UNet onto two neuron cores.
pipe_refiner.unet = NeuronUNet(UNetWrap(pipe_refiner.unet))
device_ids = [0,1]
pipe_refiner.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_refiner_filename), device_ids, set_dynamic_batching=False)



# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8


prompt = "a photo of an astronaut riding a horse on mars"
inputs = (pipe_base, pipe_refiner, prompt, n_steps, high_noise_frac, torch.manual_seed(0),)

n_runs = 50
benchmark(n_runs, "stable_diffusion_1024", run_refiner_and_base, inputs)
