.. meta::
   :description: Learn how to use Pipeline Execution Mode to optimize performance for large models with multiple submodels using NxD Inference
   :date_updated: 2025-09-19

.. _how-to-use-fpem:

=======================================================================
How to Use On-device Forward Pipeline Execution Mode for Optimization
=======================================================================

Task Overview
-------------

This topic shows you how to use Pipeline Execution Mode to optimize performance for large models with multiple submodels using the NxD Inference. This technique keeps intermediate tensors from sub models on the device to reduce data transfer overhead and minimize model latency.

In this guide, you'll learn to:

* Configure pipeline execution flags for optimal performance
* Set up multi-stage model wrappers that communicate efficiently
* Manage intermediate tensor placement between pipeline stages
* Implement a simple vision-text pipeline as a practical example

Sample Architecture
-------------------

This guide uses a vision-text multimodal model to demonstrate pipeline execution. The architecture consists of:

**Vision Model**: Processes image inputs through convolutional layers and outputs vision embeddings

**Text Model**: Takes vision embeddings and text inputs, then produces final classification results

This two-stage pipeline shows how intermediate vision embeddings can remain on the device, avoiding costly CPU transfers between model stages. The same principles apply to other multi-stage architectures like transformer decoder chains, diffusion model denoisers, or encoder-decoder pairs.

Prerequisites
-------------

- **NeuronX Distributed Inference (NxDI)**: You must have NxDI installed and configured. See NxD Inference Setup Guide.
- **Multi-stage model**: Your model should have intermediate tensors in a pipeline structure, such as Llama4-style models, Pixtral, or diffusion-based models.

The following diagram shows how intermediate tensors flow through a multi-stage pipeline::

    Input Data
        |
        v
    ┌─────────────┐
    │   Stage 1   │  <- Vision Model (Conv2D + Pooling)
    │ (SubModel)  │
    └─────────────┘
        |
        v
    Intermediate    <- Kept on device with pipeline_execution=True
    Tensors            and return_ranked_to_cpu=False
        |
        v
    ┌─────────────┐
    │   Stage 2   │  <- Text Model (Embedding + Fusion)
    │ (SubModel)  │
    └─────────────┘
        |
        v
    Final Output    <- Returned to CPU with return_ranked_to_cpu=True

Without pipeline execution, intermediate tensors transfer between CPU and device at each stage, creating overhead. With pipeline execution enabled, intermediate tensors remain on the device, reducing latency.

.. note::

   **Padding Requirements**: When passing outputs between ModelWrapper instances, you must manually pad the list of lists to ensure consistent input dimensions. Padding is crucial to maintain tensor compatibility across pipeline stages.

Instructions
------------

**1:** Import required modules and define model classes

Start by importing the necessary modules and defining your model architectures:

.. code-block:: python

    import torch
    from torch import nn
    from neuronx_distributed_inference.models.encoder_base import NeuronEncoderBase
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
    from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
    from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

    # Vision Model Definition
    class VisionModel(NeuronEncoderBase):
        def __init__(self, config: InferenceConfig):
            super().__init__(config)
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, config.vision_embedding_size)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    # Text Model Definition
    class TextModel(NeuronEncoderBase):
        def __init__(self, config: InferenceConfig):
            super().__init__(config)
            self.embedding = nn.Linear(config.text_input_size, config.text_embedding_size)
            self.fusion = nn.Linear(
                config.vision_embedding_size + config.text_embedding_size,
                config.output_size
            )

        def forward(self, vision_features, text_input):
            text_features = self.embedding(text_input)
            combined = torch.cat([vision_features, text_features], dim=1)
            return self.fusion(combined)

**2:** Configure ModelWrappers with pipeline execution flags

Set up your ModelWrapper classes with appropriate pipeline execution parameters:

.. code-block:: python

    # Vision Model Wrapper - keeps output on device
    class VisionModelWrapper(ModelWrapper):
        def __init__(self, config: InferenceConfig):
            super().__init__(
                config=config,
                model_cls=VisionModel,
                pipeline_execution=True,
                return_ranked_to_cpu=False,  # Keep output ranked for efficient pipeline
                tag="vision_model"
            )

        def input_generator(self):
            # Generate sample input for compilation
            x = torch.randn(
                self.neuron_config.batch_size,
                3,
                224,
                224
            )
            return [(x,)]

    # Text Model Wrapper - returns final output to CPU
    class TextModelWrapper(ModelWrapper):
        def __init__(self, config: InferenceConfig):
            super().__init__(
                config=config,
                model_cls=TextModel,
                pipeline_execution=True,
                return_ranked_to_cpu=True,  # Return final output to CPU
                tag="text_model"
            )

        def input_generator(self):
            # Generate sample inputs for compilation
            vision_features = torch.randn(
                self.neuron_config.batch_size,
                self.config.vision_embedding_size
            )
            text_input = torch.randn(
                self.neuron_config.batch_size,
                self.config.text_input_size
            )
            return [(vision_features, text_input)]

**3:** Create application classes

Build application classes that use your configured ModelWrappers:

.. code-block:: python

    # Application Classes
    class VisionModelApp(NeuronApplicationBase):
        def __init__(self, model_path: str, config: InferenceConfig):
            super().__init__(model_path=model_path, config=config)
            self.model = VisionModelWrapper(config)
            self.models.append(self.model)

        def forward(self, x):
            return self.models[0].forward(x)

    class TextModelApp(NeuronApplicationBase):
        def __init__(self, model_path: str, config: InferenceConfig):
            super().__init__(model_path=model_path, config=config)
            self.model = TextModelWrapper(config)
            self.models.append(self.model)

        def forward(self, vision_features, text_input):
            return self.models[0].forward(vision_features, text_input)

**4:** Run the complete pipeline example

Execute your pipeline with the configured models:

.. code-block:: python

    def main():
        # Configure models
        config = InferenceConfig(
            NeuronConfig(batch_size=32, torch_dtype=torch.float32, tp_degree=2),
            vision_embedding_size=512,
            text_input_size=256,
            text_embedding_size=512,
            output_size=1024
        )

        # Create applications
        vision_app = VisionModelApp("path/to/vision/model", config)
        text_app = TextModelApp("path/to/text/model", config)

        # Compile models
        vision_app.compile("path/to/compiled/vision")
        text_app.compile("path/to/compiled/text")

        # Load models
        vision_app.load("path/to/compiled/vision")
        text_app.load("path/to/compiled/text")

        # Example inference
        image_input = torch.randn(32, 3, 224, 224)
        text_input = torch.randn(32, 256)

        # Forward pass through vision model
        # Returns ranked output (list of lists) since return_ranked_to_cpu=False
        vision_features = vision_app.forward(image_input)

        # Forward pass through text model
        # Returns CPU tensor since return_ranked_to_cpu=True
        final_output = text_app.forward(vision_features, text_input)

        print(f"Final output shape: {final_output.shape}")  # [32, 1024]

Confirm your work
-----------------

To confirm you have successfully configured pipeline execution mode, check that your model outputs have the expected tensor placement:

.. code-block:: python

    # Check intermediate output placement
    print(f"Vision features type: {type(vision_features)}")  # Should be list of lists
    print(f"Final output shape: {final_output.shape}")       # Should be [32, 1024]
    print(f"Final output device: {final_output.device}")     # Should be CPU

Common issues
-------------

.. rubric:: Tensor dimension mismatch between pipeline stages

- **Possible solution**: Ensure you manually pad the list of lists when passing outputs between ModelWrapper instances to maintain consistent input dimensions.

.. rubric:: Performance not improving with pipeline execution

- **Possible solution**: Verify that your model has intermediate tensors in a pipeline structure. Pipeline execution works best with models like Llama4-style, Pixtral, or diffusion-based models.

.. rubric:: Memory issues with large models

- **Possible solution**: Adjust your batch size and tensor parallelism degree (tp_degree) in the NeuronConfig to better fit your available memory.