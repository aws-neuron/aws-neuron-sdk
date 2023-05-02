import torch
import neuronperf
import neuronperf.torch
import torch_neuronx

from PIL import Image
import requests
from transformers import ViTImageProcessor, ViTForImageClassification

def benchmark(batch_size):
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', torchscript=True)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs['pixel_values'].repeat([batch_size, 1, 1, 1])
    example = (inputs,)

    traced = torch_neuronx.trace(model, example)
    filename = 'model.pt'
    torch.jit.save(traced, filename)
    reports = neuronperf.torch.benchmark(filename, [example], batch_sizes=[batch_size])
    # View and save results
    print("======== {} ========".format(filename))
    neuronperf.print_reports(reports)
    neuronperf.write_csv(reports)
    neuronperf.write_json(reports)

if __name__ == '__main__':
   benchmark(batch_size=32)