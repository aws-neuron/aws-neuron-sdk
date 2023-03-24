import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import transformers
import os
import warnings

from detect_instance import get_instance_type, get_num_neuroncores

instance_type = get_instance_type() 

print(f"Detected instance type: {instance_type}")

if 'inf1' in instance_type:
    print(" - using torch_neuron.trace")
    from torch_neuron import trace
else:
    print(" - using torch_neuronx.xla_impl.trace")
    from torch_neuronx.xla_impl.trace import trace
print()

os.environ['TOKENIZERS_PARALLELISM']='false'
batch_size = 6

# Setting up NeuronCore groups for inf1.6xlarge with 16 cores
num_cores = get_num_neuroncores(instance_type)
print(f"Number of cores = {num_cores}")
os.environ['NEURON_RT_NUM_CORES'] = str(num_cores)

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

max_length=128
paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
not_paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

# Convert example inputs to a format that is compatible with TorchScript tracing
example_inputs_paraphrase = (
    torch.cat([paraphrase['input_ids']] * batch_size,0),
    torch.cat([paraphrase['attention_mask']] * batch_size,0),
    torch.cat([paraphrase['token_type_ids']] * batch_size,0)
)

# Run torch.neuron.trace to generate a TorchScript that is optimized by AWS Neuron
try:
    model_neuron = trace(model, example_inputs_paraphrase)
except Exception as e:
    print(e)
    print("libtorch_demo: Model tracing failed - check tutorial steps and preconditions")
    print("libtorch_demo: If this does not resolve your issue - Report a bug at ")
    print("https://github.com/aws-neuron/aws-neuron-sdk/issues")
    exit(1)

# Verify the TorchScript works on both example inputs
try:
    paraphrase_classification_logits_neuron = model_neuron(*example_inputs_paraphrase)
except:
    print("libtorch_demo: Neuron runtime failed - check tutorial steps and preconditions")
    print("libtorch_demo: If this does not resolve your issue - Report a bug at ")
    print("https://github.com/aws-neuron/aws-neuron-sdk/issues")
    exit(1)

# Save the TorchScript for later use
model_neuron.save(f'bert_neuron_b{batch_size}.pt')
