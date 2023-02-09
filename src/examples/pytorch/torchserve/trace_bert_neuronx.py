import torch
import torch_neuronx

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "HuggingFace's headquarters are situated in Manhattan"

max_length = 128
batch_size = 6

paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

example_inputs_paraphrase = (
    torch.cat([paraphrase['input_ids']] * batch_size, 0),
    torch.cat([paraphrase['attention_mask']] * batch_size, 0),
    torch.cat([paraphrase['token_type_ids']] * batch_size, 0)
)

# Run torch.neuron.trace to generate a TorchScript that is optimized by AWS Neuron
model_neuron_batch = torch_neuronx.trace(model, example_inputs_paraphrase)

# Save the batched model
model_neuron_batch.save('bert_neuron_b{}.pt'.format(batch_size))
