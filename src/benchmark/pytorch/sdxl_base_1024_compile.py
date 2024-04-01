import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

import math
import copy
import diffusers
from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

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

# Define datatype
DTYPE = torch.float32

# Optimized attention
def get_attention_scores_neuron(self, query, key, attn_mask):    
    if query.size() == key.size():
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=-1)
  
    return attention_probs
 
def custom_badbmm(a, b, scale):
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

# Replace original cross-attention module with custom cross-attention module for better performance
apply_neuron_attn_override(
    diffusers, get_attention_scores_neuron, neuron_scaled_dot_product_attention
)

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None
    ):
        out_tuple = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_cond_kwargs=None,
        return_dict=False,
        cross_attention_kwargs=None,
    ):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"],
        )[0]
        return UNet2DConditionOutput(sample=sample)

class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(text_embeds=out_tuple[0], last_hidden_state=out_tuple[1], hidden_states=out_tuple[2])
    
class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        out_tuple = self.text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)
        return out_tuple

# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'sdxl_base_compile_dir_1024'

# Model ID for SD XL version pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"


# --- Compile Text Encoders and save ---

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)


# Apply wrappers to make text encoders traceable
traceable_text_encoder = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
traceable_text_encoder_2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))

del pipe

text_input_ids_1 = torch.tensor([[49406,   736,  1615, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])


text_input_ids_2 = torch.tensor([[49406,   736,  1615, 49407,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]])


# Text Encoder 1
neuron_text_encoder = torch_neuronx.trace(
    traceable_text_encoder,
    text_input_ids_1,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)

text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
torch.jit.save(neuron_text_encoder, text_encoder_filename)


# Text Encoder 2
neuron_text_encoder_2 = torch_neuronx.trace(
    traceable_text_encoder_2,
    text_input_ids_2,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2'),
)

text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
torch.jit.save(neuron_text_encoder_2, text_encoder_2_filename)



# --- Compile Text Encoders and save ---

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Apply wrappers to make text encoders traceable
traceable_text_encoder = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
traceable_text_encoder_2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))

del pipe

text_input_ids_1 = torch.tensor([[49406,   736,  1615, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407]])


text_input_ids_2 = torch.tensor([[49406,   736,  1615, 49407,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]])


# Text Encoder 1
neuron_text_encoder = torch_neuronx.trace(
    traceable_text_encoder,
    text_input_ids_1,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)

text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
torch.jit.save(neuron_text_encoder, text_encoder_filename)

# Text Encoder 2
neuron_text_encoder_2 = torch_neuronx.trace(
    traceable_text_encoder_2,
    text_input_ids_2,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2'),
)

text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
torch.jit.save(neuron_text_encoder_2, text_encoder_2_filename)



# --- Compile UNet and save ---

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)


# Replace original cross-attention module with custom cross-attention module for better performance
Attention.get_attention_scores = get_attention_scores_neuron

# Apply double wrapper to deal with custom return type
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

# Only keep the model being compiled in RAM to minimze memory pressure
unet = copy.deepcopy(pipe.unet.unetwrap)
del pipe

# Compile unet - FP32
sample_1b = torch.randn([1, 4, 128, 128], dtype=DTYPE)
timestep_1b = torch.tensor(999, dtype=DTYPE).expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 2048], dtype=DTYPE)
added_cond_kwargs_1b = {"text_embeds": torch.randn([1, 1280], dtype=DTYPE),
                        "time_ids": torch.randn([1, 6], dtype=DTYPE)}
example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, added_cond_kwargs_1b["text_embeds"], added_cond_kwargs_1b["time_ids"],)

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

# Enable asynchronous and lazy loading to speed up model load
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)

# save compiled unet
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
torch.jit.save(unet_neuron, unet_filename)

# delete unused objects
del unet
del unet_neuron



# --- Compile VAE decoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
decoder = copy.deepcopy(pipe.vae.decoder)
del pipe

# Compile vae decoder
decoder_in = torch.randn([1, 4, 128, 128], dtype=DTYPE)
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder')
)

# Enable asynchronous loading to speed up model load
torch_neuronx.async_load(decoder_neuron)

# Save the compiled vae decoder
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch.jit.save(decoder_neuron, decoder_filename)

# delete unused objects
del decoder
del decoder_neuron



# --- Compile VAE post_quant_conv and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
del pipe

# Compile vae post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 128, 128], dtype=DTYPE)
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

# Enable asynchronous loading to speed up model load
torch_neuronx.async_load(post_quant_conv_neuron)

# Save the compiled vae post_quant_conv
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# delete unused objects
del post_quant_conv
del post_quant_conv_neuron
