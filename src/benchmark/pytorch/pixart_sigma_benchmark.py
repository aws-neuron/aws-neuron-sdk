import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

import copy
import diffusers
import math
import numpy as npy
import time
import torch
import torch_neuronx
import torch.nn as nn
import torch.nn.functional as F

from diffusers import PixArtSigmaPipeline
from IPython.display import clear_output
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from torch import nn

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers import Transformer2DModel

# Define datatype
DTYPE = torch.bfloat16

# Specialized benchmarking class for PixArt models.
# We cannot use any of the pre-existing benchmarking utilities to benchmark E2E PixArt performance,
# because the top-level PixArt pipeline cannot be serialized into a single Torchscript object.
# All of the pre-existing benchmarking utilities (in neuronperf or torch_neuronx) require the model to be a
# traced Torchscript.
def benchmark(n_runs, test_name, model, model_inputs):
  if not isinstance(model_inputs, tuple):
    model_inputs = (model_inputs,)
  
  warmup_run = model(*model_inputs)

  latency_collector = LatencyCollector()
  # can't use register_forward_pre_hook or register_forward_hook because PixArt pipeline is not a torch.nn.Module
  
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

class InferenceTextEncoderWrapper(nn.Module):
  def __init__(self, dtype, t: T5EncoderModel, seqlen: int):
    super().__init__()
    self.dtype = dtype
    self.device = t.device
    self.t = t
  def forward(self, text_input_ids, attention_mask=None):
    return [self.t(text_input_ids, attention_mask)['last_hidden_state'].to(self.dtype)]

class InferenceTransformerWrapper(nn.Module):
  def __init__(self, transformer: Transformer2DModel):
    super().__init__()
    self.transformer = transformer
    self.config = transformer.config
    self.dtype = transformer.dtype
    self.device = transformer.device
  def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, 
              encoder_attention_mask=None, added_cond_kwargs=None,
              return_dict=False):
    output = self.transformer(
      hidden_states, 
      encoder_hidden_states, 
      timestep, 
      encoder_attention_mask)
    return output

class SimpleWrapper(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
  def forward(self, x):
    output = self.model(x)
    return output

# --- Load all compiled models and benchmark pipeline ---
def get_pipe(resolution, dtype):
  if resolution == 256:
    transformer = Transformer2DModel.from_pretrained(
      "PixArt-alpha/PixArt-Sigma-XL-2-256x256", 
      subfolder='transformer', 
      torch_dtype=dtype,
    )
    return PixArtSigmaPipeline.from_pretrained(
      "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
      transformer=transformer,
      torch_dtype=dtype,
    )
  elif resolution == 512:
    transformer = Transformer2DModel.from_pretrained(
      "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
      subfolder='transformer', 
      torch_dtype=dtype,
    )
    return PixArtSigmaPipeline.from_pretrained(
      "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
      transformer=transformer,
      torch_dtype=dtype,
    )
  else:
    raise Exception(f"Unsupport resolution {resolution} for PixArt Sigma")

COMPILER_WORKDIR_ROOT = 'pixart_sigma_compile_dir'
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

# Select the desired resolution ()
resolution = 256
# resolution = 512

pipe = get_pipe(resolution, DTYPE)
seqlen = 300

_neuronTextEncoder = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, seqlen)
_neuronTextEncoder.t = torch.jit.load(text_encoder_filename)
pipe.text_encoder = _neuronTextEncoder
assert pipe._execution_device is not None

device_ids = [0, 1]
_neuronTransformer = InferenceTransformerWrapper(pipe.transformer)
_neuronTransformer.transformer = torch_neuronx.DataParallel(torch.jit.load(transformer_filename), device_ids, set_dynamic_batching=False)
pipe.transformer = _neuronTransformer
pipe.vae.decoder = SimpleWrapper(torch.jit.load(decoder_filename))
pipe.vae.post_quant_conv = SimpleWrapper(torch.jit.load(post_quant_conv_filename))

prompt = "a photo of an astronaut riding a horse on mars"
n_runs = 20
benchmark(n_runs, "pixart_alpha", pipe, prompt)
