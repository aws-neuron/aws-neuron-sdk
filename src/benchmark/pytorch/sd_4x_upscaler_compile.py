import os

import requests
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

from PIL import Image
from io import BytesIO

import diffusers
from diffusers import StableDiffusionUpscalePipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from packaging import version

def apply_neuron_attn_override(
    diffusers_pkg, get_attn_scores_func, neuron_scaled_dot_product_attention
):
    diffusers_version = version.parse(diffusers_pkg.__version__)
    use_new_diffusers = diffusers_version >= version.parse("0.18.0")
    if use_new_diffusers:
        diffusers_pkg.models.attention_processor.Attention.get_attention_scores = (
            get_attn_scores_func
        )
    else:
        diffusers_pkg.models.cross_attention.CrossAttention.get_attention_scores = (
            get_attn_scores_func
        )

    # If Pytorch 2 is available, a F.scaled_dot_product_attention will be used, so we need to
    # monkey patch that too to be Neuron optimized attention
    if hasattr(F, "scaled_dot_product_attention"):
        F.scaled_dot_product_attention = neuron_scaled_dot_product_attention


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = cust_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)

    else:
        attention_scores = cust_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)

    return attention_probs


def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


def neuron_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=None, is_causal=None
):
    orig_shape = None
    if len(query.shape) == 4:
        orig_shape = query.shape

        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])

        query, key, value = map(to3d, [query, key, value])

    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)

    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=-1)

    attn_out = torch.bmm(attention_probs, value)

    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )

    return attn_out


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels,
        cross_attention_kwargs=None,
    ):
        out_tuple = self.unet(
            sample, timestep, encoder_hidden_states, class_labels, return_dict=False
        )
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels,
        cross_attention_kwargs=None,
        return_dict=False,
    ):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            class_labels,
        )[0]
        return UNet2DConditionOutput(sample=sample)


class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask=None):
        return [self.neuron_text_encoder(emb)["last_hidden_state"]]

# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'stable_diffusion_upscaler_fp32'

# Model ID for SD version pipeline
model_id = "stabilityai/stable-diffusion-x4-upscaler"

# --- Compile CLIP text encoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = StableDiffusionUpscalePipeline.from_pretrained(
model_id, torch_dtype=torch.float32
)
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

text_encoder_neuron = torch_neuronx.trace(
        text_encoder.neuron_text_encoder,
        emb,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
        )

# Save the compiled text encoder
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
torch.jit.save(text_encoder_neuron, text_encoder_filename)

# delete unused objects
del text_encoder

# --- Compile VAE decoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = StableDiffusionUpscalePipeline.from_pretrained(
model_id, torch_dtype=torch.float32
)
decoder = copy.deepcopy(pipe.vae.decoder)
del pipe

# # Compile vae decoder
decoder_in = torch.randn([1, 4, 128, 128])
decoder_neuron = torch_neuronx.trace(
    decoder,
    decoder_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
)

# Save the compiled vae decoder
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch.jit.save(decoder_neuron, decoder_filename)

# delete unused objects
del decoder

# --- Compile UNet and save ---

pipe = StableDiffusionUpscalePipeline.from_pretrained(
model_id, torch_dtype=torch.float32
)

# Replace original cross-attention module with custom cross-attention module for better performance
apply_neuron_attn_override(
diffusers, get_attention_scores_neuron, neuron_scaled_dot_product_attention
)

# Apply double wrapper to deal with custom return type
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

# Only keep the model being compiled in RAM to minimze memory pressure
unet = copy.deepcopy(pipe.unet.unetwrap)
del pipe

# Compile unet - FP32
sample_1b = torch.randn([1, 7, 128, 128])
timestep_1b = torch.tensor(999).float().expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 1024])
class_labels = torch.tensor([20])
example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b, class_labels

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

# save compiled unet
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
torch.jit.save(unet_neuron, unet_filename)

# delete unused objects
del unet

# --- Compile VAE post_quant_conv and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = StableDiffusionUpscalePipeline.from_pretrained(
model_id, torch_dtype=torch.float32
)
post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
del pipe

# # # Compile vae post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 128, 128])
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv,
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)


# # Save the compiled vae post_quant_conv
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# delete unused objects
del post_quant_conv
