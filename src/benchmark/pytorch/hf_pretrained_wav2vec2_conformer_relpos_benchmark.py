import torch
import torch_neuronx
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC
import neuronperf as npf
import neuronperf.torch

BATCH_SIZE = 1
def benchmark():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
    model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
    model.eval()

    # take the first entry in the dataset as our input
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    inputs = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
    inputs = inputs.repeat([BATCH_SIZE, 1])
    example = (inputs,)

    traced = torch_neuronx.trace(model, example, compiler_args='--model-type=transformer')
    filename = 'model.pt'
    torch.jit.save(traced, filename)
    
    model_neuron = torch.jit.load(filename)
    output = model_neuron(inputs)
    print(f"output is {output}")

    reports = neuronperf.torch.benchmark(filename, [example], multiprocess=False, batch_sizes=[BATCH_SIZE])
    # View and save results
    print("======== {} ========".format(filename))
    neuronperf.print_reports(reports)
    neuronperf.write_csv(reports)
    neuronperf.write_json(reports)

if __name__ == '__main__':
    benchmark()
