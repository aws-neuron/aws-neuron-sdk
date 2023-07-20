import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import copy
import time
import torch
import torch.nn as nn
import torch_neuronx

from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from diffusers.models.cross_attention import CrossAttention

def get_attention_scores(self, query, key, attn_mask):    
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if(query.size() == key.size()):
        attention_scores = cust_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = cust_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


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

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = torch.float32
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]


class NeuronSafetyModelWrap(nn.Module):
    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    def forward(self, clip_inputs):
        return list(self.safety_model(clip_inputs).values())



# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'sd_1_5_fp32_512_compile_workdir'

# Model ID for SD version pipeline
model_id = "runwayml/stable-diffusion-v1-5"


# --- Compile CLIP text encoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
text_encoder = copy.deepcopy(pipe.text_encoder)
del pipe

# Apply the wrapper to deal with custom return type
text_encoder = NeuronTextEncoder(text_encoder)

# Compile text encoder
# This is used for indexing a lookup table in torch.nn.Embedding,
# so using random numbers may give errors (out of range).
emb = torch.tensor([[49406, 18376,   525,  7496, 49407,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0]])

with torch.no_grad():
    start_time = time.time()
    text_encoder_neuron = torch_neuronx.trace(
            text_encoder.neuron_text_encoder, 
            emb, 
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
            )
    text_encoder_neuron_compile_time = time.time() - start_time
    print('text_encoder_neuron_compile_time:', text_encoder_neuron_compile_time)

# Save the compiled text encoder
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
torch_neuronx.async_load(text_encoder_neuron)
torch.jit.save(text_encoder_neuron, text_encoder_filename)

# delete unused objects
del text_encoder
del text_encoder_neuron
del emb

# --- Compile VAE decoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
decoder = copy.deepcopy(pipe.vae.decoder)
del pipe

# Compile vae decoder
decoder_in = torch.randn([1, 4, 64, 64])
with torch.no_grad():
    start_time = time.time()
    decoder_neuron = torch_neuronx.trace(
        decoder, 
        decoder_in, 
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
        compiler_args=["--enable-fast-loading-neuron-binaries"]
    )
    vae_decoder_compile_time = time.time() - start_time
    print('vae_decoder_compile_time:', vae_decoder_compile_time)

# Save the compiled vae decoder
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch_neuronx.async_load(decoder_neuron)
torch.jit.save(decoder_neuron, decoder_filename)

# delete unused objects
del decoder
del decoder_in
del decoder_neuron

# --- Compile UNet and save ---

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Replace original cross-attention module with custom cross-attention module for better performance
CrossAttention.get_attention_scores = get_attention_scores

# Apply double wrapper to deal with custom return type
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

# Only keep the model being compiled in RAM to minimze memory pressure
unet = copy.deepcopy(pipe.unet.unetwrap)
del pipe

# Compile unet - FP32
sample_1b = torch.randn([1, 4, 64, 64])
timestep_1b = torch.tensor(999).float().expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 768])
example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

with torch.no_grad():
    start_time = time.time()
    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
        compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
    )
    unet_compile_time = time.time() - start_time
    print('unet_compile_time:', unet_compile_time)

# Enable asynchronous and lazy loading to speed up model load
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)

# save compiled unet
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
torch.jit.save(unet_neuron, unet_filename)

# delete unused objects
del unet
del unet_neuron
del sample_1b
del timestep_1b
del encoder_hidden_states_1b


# --- Compile VAE post_quant_conv and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
del pipe

# Compile vae post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 64, 64])
with torch.no_grad():
    start_time = time.time()
    post_quant_conv_neuron = torch_neuronx.trace(
        post_quant_conv, 
        post_quant_conv_in,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
        compiler_args=["--enable-fast-loading-neuron-binaries"]
    )
    vae_post_quant_conv_compile_time = time.time() - start_time
    print('vae_post_quant_conv_compile_time:', vae_post_quant_conv_compile_time)

# Save the compiled vae post_quant_conv
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
torch_neuronx.async_load(post_quant_conv_neuron)
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# delete unused objects
del post_quant_conv



# --- Compile safety checker and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
safety_model = copy.deepcopy(pipe.safety_checker.vision_model)
del pipe

clip_input = torch.randn([1, 3, 224, 224])
with torch.no_grad():
    start_time = time.time()
    safety_model = torch_neuronx.trace(
        safety_model, 
        clip_input,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model'),
        compiler_args=["--enable-fast-loading-neuron-binaries"]
    )
    safety_model_compile_time = time.time() - start_time
    print('safety_model_compile_time:', safety_model_compile_time)

# Save the compiled safety checker
safety_model_neuron_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model/model.pt')
torch_neuronx.async_load(safety_model)
torch.jit.save(safety_model, safety_model_neuron_filename)

# delete unused objects
del safety_model

print('Total compile time:', text_encoder_neuron_compile_time + vae_decoder_compile_time + unet_compile_time + vae_post_quant_conv_compile_time + safety_model_compile_time)
