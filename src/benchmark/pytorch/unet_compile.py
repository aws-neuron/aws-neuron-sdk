import torch

import neuronperf as npf
import neuronperf.torch

# Add to these lists or change as needed
model_name = "UNet"
batch_sizes = [1, 4]
pipeline_sizes = [1]

def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

if __name__ == "__main__":
    # UNet Implementation from https://github.com/milesial/Pytorch-UNet
    # load the model
    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False)
    # load the weights
    state_dict = torch.hub.load_state_dict_from_url('https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth', map_location="cpu")
    model.load_state_dict(state_dict)

    inputs = [get_batch(batch_size) for batch_size in batch_sizes]
    filename = f"{model_name}.json"

    # Compile
    print("Compiling {}".format(filename))
    npf.torch.compile(
        model,
        inputs,
        batch_sizes=batch_sizes,
        pipeline_sizes=pipeline_sizes,
        filename=filename,
        model_name=model_name,
    )