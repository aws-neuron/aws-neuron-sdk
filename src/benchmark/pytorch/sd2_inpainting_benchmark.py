import torch
import torch.nn as nn
import torch_neuronx
import os
from diffusers import StableDiffusionInpaintPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from diffusers.models.attention_processor import Attention

import argparse
import copy

torch.manual_seed(0)

def parse_argsuments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='Face of a yellow cat, high resolution, sitting on a park bench', help="user input for text to image use case")
    parser.add_argument('--target_dir', type=str, default='./sd21_inpainting_512_neuron', help="directory to save neuron compield model")
    args=parser.parse_args()
    return args

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

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.bfloat16().expand((sample.shape[0],)), encoder_hidden_states)[0]
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

inputs=parse_argsuments()
print(inputs.target_dir)
# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = inputs.target_dir

def trace_vae_encoder(model_id, height, width):
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    vae_encoder = copy.deepcopy(pipe.vae.encoder)
    del pipe

    sample_input = torch.randn([1, 3, height, width])
    vae_encoder_neuron = torch_neuronx.trace(
            vae_encoder, 
            sample_input, 
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_encoder'),
            )

    # Save the compiled text encoder
    vae_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_encoder/model.pt')
    torch.jit.save(vae_encoder_neuron, vae_encoder_filename)

    # delete unused objects
    del vae_encoder
    del vae_encoder_neuron

def trace_unet(model_id, height, width):
    # --- Compile UNet and save ---
    DTYPE = torch.bfloat16
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=DTYPE)

    # Replace original cross-attention module with custom cross-attention module for better performance
    Attention.get_attention_scores = get_attention_scores

    # Apply double wrapper to deal with custom return type
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

    # Only keep the model being compiled in RAM to minimze memory pressure
    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe

    sample_1b = torch.randn([1, 9, height, width], dtype=DTYPE)
    timestep_1b = torch.tensor(999, dtype=DTYPE).expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 1024], dtype=DTYPE)
    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
        compiler_args=["--model-type=unet-inference", "--verbose=info"],
    )

    # save compiled unet
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    torch.jit.save(unet_neuron, unet_filename)

    # delete unused objects
    del unet
    del unet_neuron
    

def main():
    
    model_id = "stabilityai/stable-diffusion-2-inpainting"
    height = 624
    width = 936

    trace_unet(model_id, height // 8, width // 8)
    trace_vae_encoder(model_id, height, width)

    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
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
    del text_encoder_neuron

    # --- Compile VAE decoder and save ---

    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    decoder = copy.deepcopy(pipe.vae.decoder)
    del pipe

    # Compile vae decoder
    decoder_in = torch.randn([1, 4, height // 8, width // 8])
    decoder_neuron = torch_neuronx.trace(
        decoder, 
        decoder_in, 
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
        compiler_args=["--verbose", "info"]
    )

    # Save the compiled vae decoder
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    torch.jit.save(decoder_neuron, decoder_filename)

    # delete unused objects
    del decoder
    del decoder_neuron
    
    # --- Compile VAE post_quant_conv and save ---

    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
    del pipe

    # Compile vae post_quant_conv
    post_quant_conv_in = torch.randn([1, 4, height // 8 , width // 8])
    post_quant_conv_neuron = torch_neuronx.trace(
        post_quant_conv, 
        post_quant_conv_in,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
        compiler_args=["--verbose", "info"]
    )

    # Save the compiled vae post_quant_conv
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
    torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

    # delete unused objects
    del post_quant_conv
    del post_quant_conv_neuron
    

if __name__ == "__main__":
    main()