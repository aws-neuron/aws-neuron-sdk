import os
import time
from optimum.neuron import NeuronStableDiffusionPipeline

# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'sd_1_5_fp32_512_compile_workdir'

# Model ID for SD version pipeline
model_id = "runwayml/stable-diffusion-v1-5"

# Compilation config
compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16","inline_weights_to_neff": "True"}
input_shapes = {"batch_size": 1, "height": 512, "width": 512}

# --- Compile the model
start_time = time.time()
stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(model_id, export=True, **compiler_args, **input_shapes)

# Save the compiled model
stable_diffusion.save_pretrained(COMPILER_WORKDIR_ROOT)

compile_time = time.time() - start_time
print('Total compile time:', compile_time)
