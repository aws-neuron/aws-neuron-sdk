import torch
import torch.nn as nn
import torch_neuronx
import os
import time
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from diffusers.models.attention_processor import Attention

import threading
import argparse
import sys
import copy
import PIL
import math

torch.manual_seed(0)

def parse_argsuments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='Face of a yellow cat, high resolution, sitting on a park bench', help="user input for text to image use case")
    parser.add_argument('--target_dir', type=str, default='./sd21_inpainting_512_neuron', help="directory to save neuron compield model")
    args=parser.parse_args()
    return args

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

DTYPE = torch.bfloat16

# Have to do this double wrapper trick to compile the unet, because
# of the special UNet2DConditionOutput output type.
class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None, added_cond_kwargs=None, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample.to(dtype=DTYPE), timestep.to(dtype=DTYPE).expand((sample.shape[0],)), encoder_hidden_states.to(dtype=DTYPE))[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]

# Optimized attention
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # Check for square matmuls
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled


def main():
    
    inputs=parse_argsuments()
    print(inputs.target_dir)

    # For saving compiler artifacts
    COMPILER_WORKDIR_ROOT = inputs.target_dir

    model_id = "stabilityai/stable-diffusion-2-inpainting"
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    vae_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_encoder/model.pt')
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0,1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.encoder = torch.jit.load(vae_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    
    height = 624
    width = 936
    base_image = PIL.Image.open('sd2_inpainting_photo.png')
    mask = PIL.Image.open('sd2_inpainting_mask.png')
    image = pipe(prompt=inputs.prompt, image=base_image, mask_image=mask, height=height, width=width).images[0]
    image.save("sd2_inpainting_output.png")
    
    n_runs = 10
    benchmark(n_runs, "stable_diffusion_inpainting", pipe, (inputs.prompt, base_image, mask, None, height, width))

if __name__ == "__main__":
    main()