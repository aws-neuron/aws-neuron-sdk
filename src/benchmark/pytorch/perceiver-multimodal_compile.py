import base64
import os
import ssl
import re
from urllib import request
import time
import random
from tqdm import tqdm
import numpy as np

from typing import Optional, Tuple, Union
from transformers import PerceiverForMultimodalAutoencoding
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder, PerceiverClassifierOutput
from transformers.models.perceiver.modeling_perceiver import restructure
import torch
import torch.nn as nn
import torch_neuronx

class MultimodalPerceiverWrapper(nn.Module):
    def __init__(self, perceiver_model, nchunks, image_chunk_size, audio_chunk_size):
        super().__init__()
        self.perceiver_model = perceiver_model
        self.nchunks = nchunks
        self.image_chunk_size = image_chunk_size
        self.audio_chunk_size = audio_chunk_size
    
    def forward(self, inputs: torch.FloatTensor,
        neuron_decoder,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):


        output_attentions = output_attentions if output_attentions is not None else self.perceiver_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.perceiver_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.perceiver_model.config.use_return_dict
        
        if self.perceiver_model.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.perceiver_model.input_preprocessor(inputs)
        else:
            modality_sizes = None
            inputs_without_pos = None
            if inputs.size()[-1] != self.perceiver_model.config.d_model:
                raise ValueError(
                    f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to config.d_model:"
                    f" {self.perceiver_model.config.d_model}. Make sure to set config.d_model appropriately."
                )

        batch_size, seq_length, _ = inputs.size()
        device = inputs.device

        # If no attention mask is provided, make them all ones
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        # Make the attention mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = self.perceiver_model.invert_attention_mask(attention_mask)

        head_mask = self.perceiver_model.get_head_mask(head_mask, self.perceiver_model.config.num_blocks * self.perceiver_model.config.num_self_attends_per_block)
        embedding_output = self.perceiver_model.embeddings(batch_size=batch_size)

        encoder_outputs = self.perceiver_model.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        reconstruction = {}
        for chunk_idx in tqdm(range(self.nchunks)):
            subsampled_output_points = {
            'image': torch.arange(
                self.image_chunk_size * chunk_idx, self.image_chunk_size * (chunk_idx + 1)).to(device),
            'audio': torch.arange(
                self.audio_chunk_size * chunk_idx, self.audio_chunk_size * (chunk_idx + 1)).to(device),
            'label': None,
            }
            
            logits = neuron_decoder(sequence_output, extended_attention_mask, 
                                             inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_output_points)

            reconstruction['label'] = logits['label']
            if 'image' not in reconstruction:
                reconstruction['image'] = logits['image']
                reconstruction['audio'] = logits['audio']
            else:
                reconstruction['image'] = torch.cat(
                    [reconstruction['image'], logits['image']], dim=1)
                reconstruction['audio'] = torch.cat(
                    [reconstruction['audio'], logits['audio']], dim=1)
            
            del logits

        return reconstruction

def custom_model_forward(
        self,
        nchunks,
        image_chunk_size,
        audio_chunk_size,
        neuron_decoder,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PerceiverClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        perceiver_wrapper = MultimodalPerceiverWrapper(self.perceiver, nchunks, image_chunk_size, audio_chunk_size)
        outputs = perceiver_wrapper(
            inputs,
            neuron_decoder,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


def custom_decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
    if self.position_encoding_type == "none":  # Queries come from elsewhere
        raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
    if subsampled_points is not None:
        # subsampled_points are the indices if the inputs would be flattened
        # however, the inputs aren't flattened, that's why we use unravel_index
        # to get the indices for the unflattened array
        # unravel_index returns a tuple (x_idx, y_idx, ...)
        # stack to get the [n, d] tensor of coordinates

        def unravel_indices(indices, shape):
            coord = []

            for dim in reversed(shape):
                coord.append(indices % dim)
                indices = indices // dim

            coord = torch.stack(coord[::-1], dim=-1)

            return coord

        pos = unravel_indices(subsampled_points, self.output_index_dims)

        batch_size = inputs.shape[0]
        # Map these coordinates to [-1, 1]
        pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
        pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_emb = self.output_position_encodings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_emb = self.output_position_encodings(
                self.output_index_dims, batch_size=batch_size, device=inputs.device, dtype=inputs.dtype, pos=pos
            )

        # Optionally project them to a target dimension.
        pos_emb = self.positions_projection(pos_emb)
        pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
    else:
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[2:]

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_emb = self.output_position_encodings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_emb = self.output_position_encodings(
                index_dims, batch_size, device=inputs.device, dtype=inputs.dtype
            )

        # Optionally project them to a target dimension.
        pos_emb = self.positions_projection(pos_emb)

    if self.concat_preprocessed_input:
        if inputs_without_pos is None:
            raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
        pos_emb = torch.cat([inputs_without_pos, pos_emb], dim=-1)

    return pos_emb


# Define wrapper for tracing encoder
class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, embedding_output, inputs, extended_attention_mask):
        output = self.encoder(embedding_output, inputs=inputs, inputs_mask=extended_attention_mask)
        return output

class NeuronEncoder(nn.Module):
    def __init__(self, encoder_wrapper):
       super().__init__()
       self.encoder_wrapper = encoder_wrapper
    
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True):

        last_hidden_states = self.encoder_wrapper(hidden_states, inputs, inputs_mask)['last_hidden_state']
        return BaseModelOutputWithCrossAttentions(last_hidden_state=last_hidden_states)


# Define wrapper for tracing decoder
class DecoderWrapper(nn.Module):
    def __init__(self, decoder, decoder_query_audio, decoder_query_image, decoder_query_label, output_postprocessor):
        super().__init__()
        self.decoder = decoder
        self.decoder_query_audio = decoder_query_audio
        self.decoder_query_image = decoder_query_image
        self.decoder_query_label = decoder_query_label
        self.output_postprocessor = output_postprocessor
        self.num_query_channels = decoder.num_query_channels
    
    def forward(self, z, query_mask,
                audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
                image_input, image_input_without_pos, image_subsampled_point, image_padding,
                label_input, label_input_without_pos, label_padding):
        audio_query = self.decoder_query_audio(inputs=audio_input, inputs_without_pos=audio_input_without_pos, subsampled_points=audio_subsampled_point)
        image_query = self.decoder_query_image(inputs=image_input, inputs_without_pos=image_input_without_pos, subsampled_points=image_subsampled_point)
        label_query = self.decoder_query_label(inputs=label_input, inputs_without_pos=label_input_without_pos)

        def embed(x, pos):
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = torch.broadcast_to(pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]])
            return torch.cat([x, pos], dim=2)

        audio_padded = embed(audio_query, audio_padding)
        image_padded = embed(image_query, image_padding)
        label_padded = embed(label_query, label_padding)

        decoder_query = torch.cat([audio_padded, image_padded, label_padded], dim=1)
        logits = self.decoder(decoder_query, z, query_mask).logits
        
        output_modality_sizes = {"audio": audio_subsampled_point.shape[0],
                                 "image": image_subsampled_point.shape[0],
                                 "label": 1}
        logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)
        return logits

class NeuronDecoder(nn.Module):
    def __init__(self, decoder_wrapper):
        super().__init__()
        self.decoder_wrapper = decoder_wrapper
        self.modalities = decoder_wrapper.decoder.modalities
        self.padding = decoder_wrapper.decoder.padding

    def forward(self, z, query_mask, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None, output_attentions=False):
        # Partition the flat inputs among the different modalities
        inputs = restructure(modality_sizes, inputs)

        assert(subsampled_points is not None)
        assert(inputs_without_pos is not None)

        for modality, decoder in self.modalities.items():
            if modality == "audio":
                audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding = inputs[modality], inputs_without_pos[modality], subsampled_points[modality].to(torch.float32), self.padding[modality]
            elif modality == "image":
                image_input, image_input_without_pos, image_subsampled_point, image_padding = inputs[modality], inputs_without_pos[modality], subsampled_points[modality].to(torch.float32), self.padding[modality]
            else:
                # label doesn't have subsampled point
                label_input, label_input_without_pos, label_padding = inputs[modality], inputs_without_pos[modality], self.padding[modality]

        assert(audio_input_without_pos is not None)
        assert(audio_subsampled_point is not None)
        assert(image_input_without_pos is not None)
        assert(image_subsampled_point is not None)
        assert(label_input_without_pos is not None)

        output = self.decoder_wrapper(z, query_mask, 
                                        audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
                                        image_input, image_input_without_pos, image_subsampled_point, image_padding,
                                        label_input, label_input_without_pos, label_padding)
        return output


model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver", 
                                                                   low_cpu_mem_usage=True)
COMPILER_WORKDIR_ROOT="perceiver_multimodal_compile_dir"

PerceiverForMultimodalAutoencoding.forward = custom_model_forward
PerceiverBasicDecoder.decoder_query = custom_decoder_query


# --- Compile Encoder ---
# Define sample inputs for tracing encoder
embedding_output = torch.randn(1, 784, 512)
sample_inputs = torch.randn(1, 52097, 704)
extended_attention_mask = torch.zeros(1, 1, 1, 52097)

# Wrap and trace the encoder, save the traced encoder
COMPILER_WORKDIR_ENCODER = os.path.join(COMPILER_WORKDIR_ROOT, "encoder")
neuron_encoder = NeuronEncoder(EncoderWrapper(model.perceiver.encoder))

# You might see a warning from trace about unused input - these are safe to ignore.
print("Compiling Encoder...")
neuron_encoder.encoder_wrapper = torch_neuronx.trace(
  neuron_encoder.encoder_wrapper,
  (embedding_output, sample_inputs, extended_attention_mask),
  compiler_workdir=COMPILER_WORKDIR_ENCODER,
  compiler_args=[f"--temp-dir={COMPILER_WORKDIR_ENCODER}", "--auto-cast=none"] # --auto-cast=none is needed to avoid numerical error.
)

# Save compiled encoder
encoder_fname = os.path.join(COMPILER_WORKDIR_ENCODER, 'model.pt')
torch.jit.save(neuron_encoder.encoder_wrapper, encoder_fname)


# --- Compile Decoder ---
# Define sample inputs for tracing decoder
z = torch.randn(1, 784, 512)
query_mask = torch.zeros(1, 1, 1, 52097)

audio_input = torch.randn(1, 1920, 704)
audio_input_without_pos = torch.randn(1, 1920, 16)
audio_subsampled_point = torch.arange(0, 15, dtype=torch.float32) # 15 = 1920/128
audio_padding = torch.randn(1, 641)

image_input = torch.randn(1, 50176, 704)
image_input_without_pos = torch.randn(1, 50176, 48)
image_subsampled_point = torch.arange(0, 6272, dtype=torch.float32) # 6272 = 224*224*16/128
image_padding = torch.randn(1, 831)

label_input = torch.randn(1, 1, 704)
label_input_without_pos = torch.randn(1, 1, 700)
label_padding = torch.randn(1, 2)

# Wrap and trace the decoder, save the traced decoder
COMPILER_WORKDIR_DECODER = os.path.join(COMPILER_WORKDIR_ROOT, "decoder")
neuron_decoder = NeuronDecoder(DecoderWrapper(model.perceiver.decoder, model.perceiver.decoder.modalities['audio'].decoder_query, \
                                              model.perceiver.decoder.modalities['image'].decoder_query, model.perceiver.decoder.modalities['label'].decoder_query, \
                                              model.perceiver.output_postprocessor))

# You might see a warning from trace about unused input - these are safe to ignore.
print("Compiling decoder...")
neuron_decoder.decoder_wrapper = torch_neuronx.trace(
   neuron_decoder.decoder_wrapper,
   (z, query_mask, audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
        image_input, image_input_without_pos, image_subsampled_point, image_padding,
        label_input, label_input_without_pos, label_padding),
   compiler_workdir=COMPILER_WORKDIR_DECODER,
   compiler_args=[f"--temp-dir={COMPILER_WORKDIR_DECODER}", "--auto-cast=none"] # --auto-cast=none is needed to avoid numerical error.
)

# Save compiled decoder
decoder_fname = os.path.join(COMPILER_WORKDIR_DECODER, 'model.pt')
torch.jit.save(neuron_decoder.decoder_wrapper, decoder_fname)

print("Done")