# Scalable Model Registry Design

<!-- meta: description: Model registry and factory pattern -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

vLLM Neuron's model registry provides dynamic selection of model implementations based on hardware platform, user configuration, and model-owner defined rules. This enables vLLM Neuron to support multiple variants of the same model architecture (e.g., BF16 vs MxFP4 quantization) while maintaining a clean integration with vLLM's model registry.

The implementation uses a **per-model factory pattern** where each architecture has its own factory class containing explicit selection logic that model owners can customize.

## Why Model Factories?

## Multiple Implementation Variants

Inference on Neuron hardware often requires different model implementations:

- **Quantization Variants**: BF16, MxFP4, INT8 optimized implementations
- **Sharding Variants**: Different sharding strategies optimized for latency vs throughput
- **Platform Variants**: TRN2-optimized vs TRN3-optimized kernels customized to hardware platform
- **Speculation Variants**: Both the base model and speculator may have variants

## Selection Requirements

The right implementation must be selected based on:

1. **Hardware Platform**: A reference example - TRN2 may default to BF16, TRN3 may default to MX data types. Note that this can be chosen by the model author.
2. **User Configuration**: Either based on HuggingFace config or explicit `neuron_config` override
3. **Model-architecture specific rules**: Custom conditions (e.g., different implementation for Llama3 and Llama3.1 due to RoPE differences)

## Integration Constraint

vLLM's `ModelRegistry` expects a single class per architecture name. The factory pattern provides this single entry point while internally selecting the appropriate implementation.

## Factory Architecture

## Class Structure

Each model architecture has a factory class that:

1. Shares the same name as the HuggingFace architecture (e.g., `GptOssForCausalLM`)
2. Implements `from_configs(hf_config, neuron_config)` as the entry point
3. Contains explicit selection logic customizable by model authors
4. Includes a validation hook for optional config validation. This allows a way for model authors to throw clear error messages for unsupported configurations.

We provide a sample factory implementation for GPT-OSS below.

``` python
class GptOssForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate GptOss implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
    """

    def __init__(self, hf_config, neuron_config):
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(cls, hf_config, neuron_config):
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        platform = get_platform_target()
        quantization = getattr(neuron_config, 'quantization', None) if neuron_config else None

        # User override takes priority
        if quantization == "mxfp4":
            from .model_mxfp4 import GptOssForCausalLM as Model
            return Model.from_configs(hf_config, neuron_config)

        if quantization == "bf16":
            from .model_bf16 import GptOssForCausalLM as Model
            return Model.from_configs(hf_config, neuron_config)

        # Platform default
        if platform == "trn3":
            from .model_mxfp4 import GptOssForCausalLM as Model
            return Model.from_configs(hf_config, neuron_config)

        # Fallback to BF16
        from .model_bf16 import GptOssForCausalLM as Model
        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(cls, hf_config, neuron_config):
        """Validate that the configuration is supported. Add rules as needed."""
        pass
```

**Key Design Decisions:**

- **Explicit Logic**: Selection is plain human readable code without abstractions
- **Lazy Imports**: Implementation modules are imported only when selected
- **Validation Hook**: `_validate_config()` is called before selection for future validation
- **Model-Owner Control**: Each factory is independent, allowing custom rules per model

## File Organization

``` text
vllm_neuron/model/
├── gpt_oss/
│   ├── __init__.py          # Exports factory as GptOssForCausalLM
│   ├── factory.py           # Factory with selection logic
│   ├── model_bf16.py        # BF16 implementation
│   ├── model_mxfp4.py       # MxFP4 implementation
│   └── weight_loaders.py    # Shared weight loading utilities
│
├── llama3/
│   ├── __init__.py          # Exports factories
│   ├── factory.py           # LlamaForCausalLM + Eagle3LlamaForCausalLM factories
│   ├── model.py             # BF16 implementation
│   └── eagle3_model.py      # Eagle3 speculation implementation
│
├── new_model/               # Example additional model architecture
│   ├── __init__.py          # Exports factories
│   ├── factory.py           # NewModelForCausalLM factory
│   ├── model_impl1.py       # Implementation type 1
│   ├── model_impl2.py       # Implementation type 2
│   └── model_impl3.py       # Implementation type 3
│
└── registry.py              # Imports factories from __init__.py (unchanged)
```

## Integration Flow

## Registration Flow

The factory integrates seamlessly with the existing vLLM registration:

``` text
┌─────────────────────────────────────────────────────────────────────────┐
│                         REGISTRATION FLOW                               │
└─────────────────────────────────────────────────────────────────────────┘

1. NeuronWorker.__init__() calls:
   for arch, model_cls in registry.get_models():
       ModelRegistry.register_model(arch, model_cls)

2. registry.get_models() imports from model packages:
   from .gpt_oss import GptOssForCausalLM      # ← Factory class
   from .llama3 import LlamaForCausalLM        # ← Factory class
   from .llama3 import Eagle3LlamaForCausalLM  # ← Factory class

3. Each __init__.py exports the factory (not the raw implementation)
```

**No changes required to:**

- `registry.py`
- `NeuronWorker`
- `NeuronModelRunner`

## Instantiation Flow

When vLLM loads a model, the factory selects the appropriate implementation:

``` text
┌─────────────────────────────────────────────────────────────────────────┐
│                         INSTANTIATION FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

1. NeuronModelRunner.load_model() calls:
   model_cls.from_configs(hf_config, neuron_config)

2. Factory.from_configs() executes:
   a) _validate_config() - check config is supported
   b) Selection logic    - explicit if/else based on platform + config, completely controlled by model author
   c) Import + instantiate the selected implementation

3. Returns actual model instance

4. All subsequent calls go directly to the implementation
```

## Config Validation

The `_validate_config()` method provides a hook for validating configurations before selection. Currently a pass-through, model owners can add validation rules. We show an example below:

``` python
@classmethod
def _validate_config(cls, hf_config, neuron_config):
    """Validate that the configuration is supported."""
    platform = get_platform_target()
    quantization = getattr(neuron_config, 'quantization', None) if neuron_config else None

    # Example: Validate quantization is known
    valid_quantizations = {"bf16", "mxfp4", None}
    if quantization not in valid_quantizations:
        raise UnsupportedConfigError(
            f"Unsupported quantization '{quantization}' for GptOss. "
            f"Valid options: {valid_quantizations - {None}}"
        )

    # Example: Validate MxFP4 is only used on supported platforms
    if quantization == "mxfp4" and platform not in ("trn2", "trn3"):
        raise UnsupportedConfigError(
            f"MxFP4 quantization requires TRN2 or TRN3, but running on '{platform}'"
        )

    # Add more validations as needed...
```

## Speculative Decoding Support

Speculative decoding models (e.g., Eagle3) have their own factory classes:

``` python
class Eagle3LlamaForCausalLM:
    """Factory for Eagle3 Llama speculative decoding implementation."""

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        cls._validate_config(hf_config, neuron_config)

        # Currently only BF16, ready for future MxFP4 variant
        from .eagle3_model import Eagle3LlamaForCausalLM as Model
        return Model.from_configs(hf_config, neuron_config)
```

Each speculator model:

- Gets its own factory class
- Has its own architecture name in vLLM registry
- Can have independent selection rules from the base model

## Example: Adding a New Variant

To add a new MxFP4 variant to an existing model:

1. **Create the implementation file**

    ``` bash
    # Create llama3/model_mxfp4.py with LlamaForCausalLM class
    ```

2. **Update the factory**

    ``` python
    # In llama3/factory.py
    class LlamaForCausalLM:
        @classmethod
        def from_configs(cls, hf_config, neuron_config):
            cls._validate_config(hf_config, neuron_config)

            platform = get_platform_target()
            quantization = getattr(neuron_config, 'quantization', None) if neuron_config else None

            if quantization == "mxfp4":
                from .model_mxfp4 import LlamaForCausalLM as Model
                return Model.from_configs(hf_config, neuron_config)

            if platform == "trn3":
                from .model_mxfp4 import LlamaForCausalLM as Model
                return Model.from_configs(hf_config, neuron_config)

            from .model import LlamaForCausalLM as Model
            return Model.from_configs(hf_config, neuron_config)
    ```

3. **No changes needed to**

    - `__init__.py` (already exports factory)
    - `registry.py` (unchanged)
    - vLLM integration code (unchanged)

## Out-of-Tree Models

This factory pattern is fully compatible with out-of-tree (external) model implementations. External teams or private models not part of vLLM Neuron can use the same design without modifying vLLM Neuron core code.

**How to Create an Out-of-Tree Model:**

1. **Create the factory following the same pattern:**

    ``` python
    # my_external_model/factory.py
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    class MyCustomModelForCausalLM:
        """Factory for external model with multiple variants."""

        @classmethod
        def from_configs(cls, hf_config, neuron_config):
            cls._validate_config(hf_config, neuron_config)

            platform = get_platform_target()
            # ... custom selection logic ...

            from .model_impl import MyCustomModelForCausalLM as Model
            return Model.from_configs(hf_config, neuron_config)

        @classmethod
        def _validate_config(cls, hf_config, neuron_config):
            pass
    ```

2. **Register directly with vLLM's ModelRegistry:**

    ``` python
    # my_external_model/__init__.py or registration script
    from vllm import ModelRegistry
    from .factory import MyCustomModelForCausalLM

    # Register the factory with vLLM
    ModelRegistry.register_model("MyCustomModelForCausalLM", MyCustomModelForCausalLM)
    ```

3. **File organization:**

    ``` text
    my_external_model/
    ├── __init__.py          # Registration with vLLM
    ├── factory.py           # Factory with selection logic
    ├── model_impl1.py       # Implementation choice 1
    └── model_impl2.py       # Implementation choice 2
    ```

**Features for Out-of-Tree Models:**

- No changes to vLLM Neuron core code required
- Full control over selection logic
- Can use vLLM Neuron utilities like module APIs and weight loading APIs
- Independent release cycle from vLLM Neuron
- Same validation and selection patterns
