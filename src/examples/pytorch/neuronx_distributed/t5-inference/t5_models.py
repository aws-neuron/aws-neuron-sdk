import torch
import neuronx_distributed

from transformers import T5Tokenizer, T5ForConditionalGeneration

from wrapper import EncoderWrapper, DecoderWrapper
from t5_model_layers import load_pretrained_with_parallel_attn

def get_wrapped_encoder(max_length, num_beams, tp_degree, model):
    encoder = EncoderWrapper(model.encoder, model.decoder, model.config, num_beams, max_length, "xla", num_beams, tp_degree=tp_degree)
    encoder.eval()
    
    # We are alaising the cache, so that way we keep the cache always on device.
    aliases = {}
    for i in range(len(encoder.past_key_values_sa)):
        aliases[encoder.past_key_values_sa[i]] = i
    
    for i in range(len(encoder.past_key_values_ca)):
        aliases[encoder.past_key_values_ca[i]] = len(encoder.past_key_values_sa) + i

    return encoder, aliases


def get_wrapped_decoder(max_length, num_beams, tp_degree, model):
    decoder = DecoderWrapper(decoder=model.decoder,
                             lm_head=model.lm_head,
                             model_config=model.config,
                             num_beams=num_beams,
                             max_length=max_length,
                             device="xla",
                             tp_degree=tp_degree)
    
    decoder.eval()
    
    num_outputs_from_trace = 3 if num_beams > 1 else 1
    aliases = {}
    for i in range(len(decoder.past_key_values_sa)):
        aliases[decoder.past_key_values_sa[i]] = i + num_outputs_from_trace
    for i in range(len(decoder.past_key_values_ca)):
        aliases[decoder.past_key_values_ca[i]] = len(decoder.past_key_values_sa) + i + num_outputs_from_trace

    return decoder, aliases

### Callable functions used by neuronx-distributed trace

def get_t5_3b_4_128_tp8_encoder():

    max_length=128
    num_beams=4
    tp_degree=8
    model = load_pretrained_with_parallel_attn("t5-3b")
    
    return get_wrapped_encoder(max_length, num_beams, tp_degree, model)

def get_t5_3b_4_128_tp8_decoder():
    
    max_length=128
    num_beams=4
    tp_degree=8

    model = load_pretrained_with_parallel_attn("t5-3b")
    
    return get_wrapped_decoder(max_length, num_beams, tp_degree, model)


def parallel_trace_encoder(model_name: str,
                           max_length: int,
                           num_beams: int,
                           tp_degree: int):
    
    print("starting encoder parallel trace")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    get_encoder_callable = get_t5_3b_4_128_tp8_encoder

    # Trace encoder
    batch_encoding = tokenizer("translate English to German: Lets go home now",
                               max_length=max_length, truncation=True, padding='max_length', return_tensors="pt")
    input_ids = batch_encoding['input_ids']
    attention_mask = batch_encoding['attention_mask']

    # Here we are tracing the encoder and cache together. Cache is marked as state and we are aliasing.
    traced_encoder = neuronx_distributed.trace.parallel_model_trace(get_encoder_callable, (
            input_ids,
            attention_mask,
        ), 
        tp_degree=tp_degree, 
        compiler_workdir="/tmp/encoder/",
        )
    setattr(traced_encoder, 'main_input_name', 'input_ids')  # Attribute required by beam search

    print("completed encoder parallel trace")

    return traced_encoder


def parallel_trace_decoder(model: T5ForConditionalGeneration,
                           model_name: str,
                           num_beams: int,
                           max_length: int,
                           tp_degree: int):

    print("starting decoder trace")

    # Determine which func to pass NxD based on request parameters
    get_decoder_callable = get_t5_3b_4_128_tp8_decoder
    
    # We create mock inputs so we can trace the decoder
    decoder_input_ids = torch.ones((num_beams, 1), dtype=torch.int64)
    decoder_attention_mask = torch.ones((num_beams, max_length), dtype=torch.int32)
    encoder_attention_mask = torch.ones((num_beams, max_length), dtype=torch.int64)
    encoder_hidden_states = torch.ones((num_beams, max_length, model.config.d_model), dtype=torch.float32)

    beam_idx = torch.arange(0, num_beams, dtype=torch.int64)
    beam_scores = torch.zeros((num_beams,), dtype=torch.float)

    traced_decoder = neuronx_distributed.trace.parallel_model_trace(get_decoder_callable, (
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            beam_idx,
            beam_scores
        ), 
        tp_degree=tp_degree,
        compiler_workdir="/tmp/decoder/",
        )

    print("complete decoder trace")

    return traced_decoder
