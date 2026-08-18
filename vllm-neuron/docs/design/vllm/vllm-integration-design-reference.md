# vLLM Integration Design Reference

<!-- meta: description: vLLM integration design reference -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

This document provides a comprehensive design reference for the vLLM Neuron integration in vLLM Neuron (Neuron Distributed Inference). It covers the complete flow from plugin registration to request execution, detailing the APIs, data structures, and implementation patterns required for integrating custom hardware backends with vLLM.

### High-Level Architecture

``` text
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM Engine                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Scheduler     │  │   Executor      │  │   Frontend      │  │
│  │                 │  │                 │  │   (API Server)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           Platform Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────┐  │
│  │   GPU Platform  │  │   TPU Platform  │  │  vLLM Neuron Platform  │  │
│  └─────────────────┘  └─────────────────┘  └────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Worker Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   GPU Worker    │  │   TPU Worker    │  │  Neuron Worker  │  │
│  │                 │  │                 │  │                 │  │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │
│  │ │Model Runner │ │  │ │Model Runner │ │  │ │Model Runner │ │  │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Hardware Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │      CUDA       │  │   TPU Cores     │  │ Neuron Cores    │ │
│  │     Devices     │  │                 │  │ (Inferentia/    │ │
│  │                 │  │                 │  │  Trainium)      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **NeuronPlatform**: Integrates with vLLM's platform detection system
2. **NeuronWorker**: Implements the WorkerBase interface for Neuron hardware
3. **NeuronModelRunner**: Handles model execution and token sampling
4. **Registration System**: Auto-discovers and registers the plugin with vLLM

## vLLM Plugin System & Registration Flow

### Plugin Discovery Mechanism

vLLM uses Python's entry point system to discover and load hardware plugins. The registration process follows this sequence:

``` text
Package Import
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Entry Point Discovery                                        │
│    - vLLM scans for 'vllm.platform_plugins' entry points        │
│    - Calls registration function from each plugin               │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Platform Registration                                        │
│    - Plugin returns platform class path if hardware detected    │
│    - vLLM registers platform in global registry                 │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Platform Selection                                           │
│    - vLLM iterates through registered platforms                 │
│    - First platform that claims compatibility is selected       │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Configuration & Worker Setup                                 │
│    - Platform configures vLLM settings                          │
│    - Worker class is specified and instantiated                 │
└─────────────────────────────────────────────────────────────────┘
```

### Entry Point Configuration

The plugin is registered via `pyproject.toml`:

``` toml
[project.entry-points."vllm.platform_plugins"]
neuron = "vllm_neuron:register"
```

### Registration Function

The registration function detects hardware and returns the platform class:

``` python
def register() -> str | None:
    """Register the vLLM Neuron platform if Neuron devices are present."""
    if not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping vLLM Neuron plugin registration.",
            category=UserWarning,
        )
        return None
    return "vllm_neuron.vllm.platform.NeuronPlatform"

def _is_neuron_dev() -> bool:
    """Detect Neuron device by checking for /dev/neuron* devices."""
    neuron_devices = glob.glob('/dev/neuron*')
    return len(neuron_devices) > 0
```

### Platform Class Implementation

The platform class configures vLLM for Neuron hardware:

``` python
class NeuronPlatform(Platform):
    _enum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "neuron"  # Logical device name
    device_type: str = "neuron"  # Device type for vLLM
    ray_device_key: str = "neuron_cores"  # Ray resource key
    supported_quantization: list[str] = ["neuron_quant"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Configure vLLM for vLLM Neuron platform."""
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_neuron.vllm.worker.neuron_worker.NeuronWorker"

        cache_config = vllm_config.cache_config
        cache_config.block_size = 16  # Neuron-optimized block size
```

## Server Startup Flow

The vLLM server startup follows a well-defined sequence of API calls. Understanding this flow is crucial for implementing a compatible worker.

### Initialization Sequence

``` text
vLLM Engine Start
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Platform Detection & Configuration                           │
│    Platform.check_and_update_config(vllm_config)               │
│    - Updates worker class, cache settings, device config        │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Worker Instantiation                                         │
│    worker = WorkerClass(vllm_config, local_rank, rank, ...)     │
│    - Creates worker instances for each device/process           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Device Initialization                                        │
│    worker.init_device()                                         │
│    - Initialize hardware, set device context                    │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Model Loading                                                │
│    worker.load_model()                                          │
│    - Load model weights, compile for target hardware            │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Memory Profiling                                             │
│    available_memory = worker.determine_available_memory()       │
│    - Profile memory usage, determine KV cache capacity          │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Cache Initialization                                         │
│    kv_cache_config = create_kv_cache_config(...)               │
│    worker.initialize_from_config(kv_cache_config)               │
│    - Allocate KV cache memory, set up cache management          │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Model Compilation/Warmup                                     │
│    worker.compile_or_warm_up_model()                            │
│    - Compile model for hardware, run warmup iterations          │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Server Ready                                                 │
│    - vLLM server starts accepting requests                      │
│    - Scheduler begins processing request queue                   │
└─────────────────────────────────────────────────────────────────┘
```

### API Call Details

#### init_device() (Registration)

Called once per worker to initialize the hardware device.

``` python
def init_device(self) -> None:
    """Initialize Neuron device and set device context."""
    # Example implementation:
    # - Set up Neuron runtime
    # - Initialize device context
    # - Configure memory allocators
    self.device = torch.device("cpu")  # Neuron uses CPU device type
    logger.info("Neuron device initialized")
```

#### load_model() (Registration)

Loads the model onto the target device.

``` python
def load_model(self) -> None:
    """Load model for Neuron execution."""
    # Example implementation:
    # - Load model weights from checkpoint
    # - Convert to Neuron-compatible format
    # - Apply any hardware-specific optimizations
    self.model_runner.load_model()
    logger.info("Model loaded on Neuron device")
```

#### determine_available_memory() (Registration)

Profiles memory usage to determine KV cache capacity.

``` python
@torch.inference_mode()
def determine_available_memory(self) -> int:
    """Determine available memory for KV cache."""
    # Example implementation:
    # - Query Neuron device memory
    # - Account for model memory usage
    # - Reserve memory for intermediate tensors
    available_memory_bytes = 1 * 1024 * 1024 * 1024  # 1GB example
    return available_memory_bytes
```

#### initialize_from_config()

Sets up KV cache with the calculated configuration.

``` python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    """Initialize KV cache from configuration."""
    # Example implementation:
    # - Allocate KV cache tensors
    # - Set up cache management structures
    # - Configure memory layout for Neuron
    self.model_runner.initialize_kv_cache(kv_cache_config)
```

## Request Execution Flow

Once the server is running, each request follows this execution path:

### Request Processing Pipeline

``` text
Client Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Request Ingestion                                            │
│    - Parse request (prompt, sampling params, etc.)              │
│    - Create Request object with unique ID                       │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Scheduler Processing                                         │
│    - Add request to queue                                       │
│    - Batch compatible requests                                  │
│    - Allocate KV cache blocks                                   │
│    - Create SchedulerOutput                                     │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Worker Execution                                             │
│    output = worker.execute_model(scheduler_output)              │
│    - Process batch of requests                                  │
│    - Return ModelRunnerOutput or None                           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Token Sampling (if execute_model returned None)              │
│    output = worker.sample_tokens(grammar_output)                │
│    - Sample next tokens from logits                             │
│    - Apply sampling constraints                                 │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Response Generation                                          │
│    - Update request states                                      │
│    - Generate response for completed requests                   │
│    - Continue processing for ongoing requests                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Through Components

``` text
SchedulerOutput
     │
     │ Contains:
     │ - scheduled_new_reqs: List[NewRequestData]
     │ - scheduled_cached_reqs: CachedRequestsData  
     │ - num_scheduled_tokens: int
     │ - total_num_scheduled_tokens: int
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Worker.execute_model()                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              ModelRunner.execute_model()                    ││
│  │                                                             ││
│  │  1. Extract request IDs and batch info                      ││
│  │  2. Prepare input tensors (input_ids, positions)            ││
│  │  3. Forward through model                                   ││
│  │  4. Return logits tensor                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              ModelRunner.sample_tokens()                    ││
│  │                                                             ││
│  │  1. Apply sampling strategy (greedy, top-k, top-p)          ││
│  │  2. Generate token IDs and logprobs                         ││
│  │  3. Create ModelRunnerOutput                                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
ModelRunnerOutput
     │
     │ Contains:
     │ - req_ids: List[str]
     │ - req_id_to_index: Dict[str, int]
     │ - sampled_token_ids: List[List[int]]
     │ - logprobs: Optional[LogprobsLists]
     │ - prompt_logprobs_dict: Dict[str, List[PromptLogprobs]]
     │
     ▼
Response to Client
```

## Worker API Reference

The WorkerBase interface defines the contract that all vLLM workers must implement. Here's a comprehensive reference with examples from GPU and TPU implementations.

### Core Lifecycle Methods

#### init_device()

Initializes the hardware device and sets up the execution context.

*GPU Implementation Example:*

``` python
def init_device(self) -> None:
    if self.device_config.device.type == "cuda":
        # Set CUDA device
        torch.cuda.set_device(self.device_config.device)
        _check_if_gpu_supports_dtype(self.model_config.dtype)
        torch.cuda.empty_cache()
        self.init_gpu_memory = torch.cuda.mem_get_info()[0]
    else:
        raise RuntimeError(f"Not support device type: {self.device_config.device}")

    # Initialize custom ops
    from vllm import _custom_ops as ops
    ops.init_custom_ar()
```

*TPU Implementation Example:*

``` python
def init_device(self) -> None:
    # Initialize JAX for TPU
    import jax
    self.device = jax.devices()[self.local_rank]

    # Set up TPU-specific configurations
    os.environ["TPU_LIBRARY_PATH"] = "/usr/local/lib/python3.8/dist-packages/libtpu/libtpu.so"
```

*Neuron Implementation:*

``` python
def init_device(self) -> None:
    """Initialize Neuron device and runtime."""
    # Set up Neuron runtime environment
    os.environ["NEURON_RT_NUM_CORES"] = str(self.parallel_config.tensor_parallel_size)

    # Initialize Neuron device context
    self.device = torch.device("cpu")  # Neuron uses CPU device type

    # Set up Neuron-specific memory management
    self._setup_neuron_memory()
```

#### load_model()

Loads the model onto the target device and prepares it for execution.

*GPU Implementation Example:*

``` python
def load_model(self) -> None:
    with CudaMemoryProfiler() as m:
        self.model_runner.load_model()

    self.init_gpu_memory = torch.cuda.mem_get_info()[0]
    if not self.model_config.enforce_eager:
        self.model_runner.capture_model(self.gpu_cache)
```

*TPU Implementation Example:*

``` python
def load_model(self) -> None:
    # Load model with JAX/Flax
    self.model_runner = TPUModelRunner(
        model_config=self.model_config,
        parallel_config=self.parallel_config,
        scheduler_config=self.scheduler_config,
        device_config=self.device_config,
    )
    self.model_runner.load_model()
```

*Neuron Implementation:*

``` python
def load_model(self) -> None:
    """Load and compile model for Neuron execution."""
    self.model_runner.load_model()

    # Compile model for Neuron if not already compiled
    if not self._is_model_compiled():
        self._compile_model_for_neuron()
```

### Memory Management Methods

#### determine_available_memory()

Profiles memory usage to determine how much memory is available for KV cache.

*GPU Implementation Example:*

``` python
@torch.inference_mode()
def determine_available_memory(self) -> int:
    """Profile memory usage and determine available memory for KV cache."""
    torch.cuda.empty_cache()

    # Get current memory usage
    free_memory, total_memory = torch.cuda.mem_get_info()

    # Account for model memory
    model_memory = total_memory - self.init_gpu_memory

    # Reserve memory for activations and other overhead
    reserved_memory = self._calculate_reserved_memory()

    available_memory = free_memory - reserved_memory
    return max(0, available_memory)
```

*Neuron Implementation:*

``` python
@torch.inference_mode()
def determine_available_memory(self) -> int:
    """Determine available memory for KV cache on Neuron."""
    # Query Neuron device memory
    neuron_memory_info = self._get_neuron_memory_info()

    # Account for model memory usage
    model_memory = self._estimate_model_memory()

    # Calculate available memory for KV cache
    available_memory = neuron_memory_info.free - model_memory

    # Apply safety margin
    safety_margin = int(available_memory * 0.1)  # 10% safety margin
    return max(0, available_memory - safety_margin)
```

#### get_cache_block_size_bytes()

Returns the size of each KV cache block in bytes.

*GPU Implementation Example:*

``` python
def get_cache_block_size_bytes(self) -> int:
    """Calculate cache block size based on model configuration."""
    head_size = self.model_config.get_head_size()
    num_heads = self.model_config.get_num_kv_heads(self.parallel_config)
    num_layers = self.model_config.get_num_layers(self.parallel_config)

    key_cache_block = self.cache_config.block_size * num_heads * head_size
    value_cache_block = self.cache_config.block_size * num_heads * head_size
    total = num_layers * (key_cache_block + value_cache_block)

    dtype_size = _get_dtype_size(self.cache_config.cache_dtype)
    return dtype_size * total
```

### Execution Methods

#### execute_model() (Worker)

The core execution method that processes a batch of requests.

*GPU Implementation Example:*

``` python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    kv_caches: List[torch.Tensor],
    intermediate_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Optional[ModelRunnerOutput]:
    if scheduler_output.is_empty():
        return None

    # Execute model through model runner
    output = self.model_runner.execute_model(
        scheduler_output, kv_caches, intermediate_tensors
    )

    # For async execution, may return None and require sample_tokens call
    return output
```

*Neuron Implementation:*

``` python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    kv_caches: Optional[List[torch.Tensor]] = None,
    intermediate_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> ModelRunnerOutput:
    """Execute model on Neuron hardware."""
    if self._is_empty_batch(scheduler_output):
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Execute through Neuron model runner
    logits = self.model_runner.execute_model(scheduler_output, kv_caches)

    # Sample tokens immediately (Neuron doesn't support async sampling)
    sample_metadata = self._create_sampling_metadata(scheduler_output)
    output = self.model_runner.sample_tokens(logits, sample_metadata)

    return output
```

## Model Runner API Reference

The ModelRunner is responsible for the actual model execution, input preparation, and token sampling. It works closely with the Worker but focuses on the computational aspects.

### Core Responsibilities

1. **Input Processing**: Convert SchedulerOutput to model inputs
2. **Model Execution**: Forward pass through the neural network
3. **Output Processing**: Convert model outputs to tokens
4. **Sampling**: Apply sampling strategies to generate next tokens
5. **KV Cache Management**: Manage key-value cache for attention

### Key Methods

#### execute_model() (Model Runner)

Processes scheduler output and executes the model forward pass.

*GPU Implementation Pattern:*

``` python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    kv_caches: List[torch.Tensor],
    intermediate_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Optional[ModelRunnerOutput]:

    # 1. Prepare model inputs
    model_input = self._prepare_model_input(scheduler_output)

    # 2. Execute model forward pass
    hidden_states = self.model(
        input_ids=model_input.input_tokens,
        positions=model_input.input_positions,
        kv_caches=kv_caches,
        attn_metadata=model_input.attn_metadata,
    )

    # 3. Sample tokens if not using async sampling
    if not self.is_async_mode:
        return self._sample_tokens(hidden_states, scheduler_output)

    # 4. Store logits for async sampling
    self._store_logits_for_sampling(hidden_states)
    return None
```

*Neuron Implementation:*

``` python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    kv_caches: Optional[List[torch.Tensor]] = None
) -> torch.Tensor:
    """Execute model and return logits for sampling."""

    # 1. Extract request information
    self._extract_request_ids(scheduler_output)

    # 2. Prepare inputs for Neuron model
    input_ids, positions, attn_metadata = self._prepare_inputs(scheduler_output)

    # 3. Execute Neuron model
    logits = self.model(
        input_ids=input_ids,
        positions=positions,
        kv_caches=kv_caches,
        attn_metadata=attn_metadata
    )

    return logits
```

#### sample_tokens()

Applies sampling strategies to generate next tokens from model logits.

*Implementation Pattern:*

``` python
def sample_tokens(
    self,
    logits: torch.Tensor,
    sample_metadata: SamplingMetadata
) -> ModelRunnerOutput:
    """Sample next tokens from logits."""

    # 1. Apply logits processors (temperature, top-p, top-k, etc.)
    processed_logits = self._apply_logits_processors(logits, sample_metadata)

    # 2. Sample tokens based on strategy
    if sample_metadata.all_greedy:
        sampled_tokens = torch.argmax(processed_logits, dim=-1)
    else:
        sampled_tokens = self._multinomial_sample(processed_logits, sample_metadata)

    # 3. Calculate logprobs if requested
    logprobs = None
    if sample_metadata.max_num_logprobs > 0:
        logprobs = self._compute_logprobs(processed_logits, sampled_tokens, sample_metadata)

    # 4. Create output structure
    return ModelRunnerOutput(
        req_ids=self.batch_req_ids,
        req_id_to_index=self.req_id_to_index,
        sampled_token_ids=sampled_tokens.tolist(),
        logprobs=logprobs,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=None,
    )
```

### Input Preparation Patterns

#### Batch Processing

``` python
def _prepare_inputs(self, scheduler_output: SchedulerOutput) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
    """Prepare model inputs from scheduler output."""

    # Extract all requests in the batch
    all_requests = []
    all_requests.extend(scheduler_output.scheduled_new_reqs)
    if scheduler_output.scheduled_cached_reqs:
        all_requests.extend(scheduler_output.scheduled_cached_reqs.reqs)

    # Prepare input tensors
    input_ids_list = []
    position_ids_list = []

    for req in all_requests:
        # Get input tokens for this request
        if hasattr(req, 'input_tokens'):
            tokens = req.input_tokens
        else:
            tokens = req.token_ids  # Fallback for different vLLM versions

        input_ids_list.append(torch.tensor(tokens))

        # Generate position IDs
        seq_len = len(tokens)
        positions = torch.arange(seq_len)
        position_ids_list.append(positions)

    # Pad and batch tensors
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    position_ids = torch.nn.utils.rnn.pad_sequence(position_ids_list, batch_first=True, padding_value=0)

    # Create attention metadata (simplified for example)
    attn_metadata = None  # In real implementation, create proper AttentionMetadata

    return input_ids, position_ids, attn_metadata
```

## vLLM Neuron-Specific Implementation

### Production Requirements

A production vLLM Neuron integration requires:

1. **Model architecture**: Transformer model implementations for supported architectures
2. **Weight loading**: Support for the applicable checkpoint formats
3. **KV cache integration**: Attention that reads from and writes to the KV cache
4. **Sampling**: Token sampling with temperature, top-k, and top-p controls
5. **Distributed execution**: Tensor, data, and expert parallelism as applicable
6. **Neuron optimization**: Compilation and Neuron-specific kernel optimizations

## Appendix: Key Data Structures

### SchedulerOutput

The `SchedulerOutput` is the primary input to worker execution, containing all information about the batch of requests to process.

**Structure:**

``` python
@dataclass
class SchedulerOutput:
    """Output from the vLLM scheduler containing batch information."""

    # New requests being processed for the first time
    scheduled_new_reqs: List[NewRequestData]

    # Previously processed requests with cached KV data
    scheduled_cached_reqs: CachedRequestsData

    # Token counts for this batch
    num_scheduled_tokens: int
    total_num_scheduled_tokens: int

    # Additional metadata
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
```

**Key Fields:**

- **scheduled_new_reqs**: List of new requests entering the system

  ``` python
  @dataclass
  class NewRequestData:
      req_id: str                    # Unique request identifier
      input_tokens: List[int]        # Tokenized input prompt
      sampling_params: SamplingParams # Sampling configuration
      arrival_time: float            # When request arrived
      lora_request: Optional[LoRARequest] = None
  ```

- **scheduled_cached_reqs**: Requests with existing KV cache data

  ``` python
  @dataclass  
  class CachedRequestsData:
      req_ids: List[str]             # Request IDs in batch order
      req_id_to_index: Dict[str, int] # Mapping for quick lookup
      num_computed_tokens: int       # Tokens already processed
  ```

**Usage Pattern:**

``` python
def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
    # Check if batch is empty
    if not scheduler_output.scheduled_new_reqs and not scheduler_output.scheduled_cached_reqs:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Extract request IDs in processing order
    req_ids = []

    # Process new requests first
    for new_req in scheduler_output.scheduled_new_reqs:
        req_ids.append(new_req.req_id)

    # Then cached requests
    if scheduler_output.scheduled_cached_reqs:
        req_ids.extend(scheduler_output.scheduled_cached_reqs.req_ids)

    # Continue with model execution...
```

### ModelRunnerOutput

The `ModelRunnerOutput` contains the results of model execution, including sampled tokens and optional logprobs.

**Structure:**

``` python
@dataclass
class ModelRunnerOutput:
    """Output from model execution containing sampled tokens."""

    # Request identification
    req_ids: List[str]                              # Request IDs in batch order
    req_id_to_index: Dict[str, int]                 # Quick lookup mapping

    # Generated tokens
    sampled_token_ids: List[List[int]]              # Sampled tokens per request

    # Optional probability information
    logprobs: Optional[LogprobsLists]               # Token logprobs if requested
    prompt_logprobs_dict: Dict[str, List[PromptLogprobs]] # Prompt logprobs

    # Additional outputs
    pooler_output: Optional[List[torch.Tensor]]     # Pooled representations
    kv_connector_output: Optional[torch.Tensor]     # KV connector data
    num_nans_in_logits: Optional[int]              # NaN detection
```

**Key Fields:**

- **req_ids**: Must match the order from SchedulerOutput processing
- **sampled_token_ids**: List of token lists, one per request in batch
- **logprobs**: Detailed probability information when requested

**LogprobsLists Structure:**

``` python
@dataclass
class LogprobsLists:
    """Container for logprob information."""

    logprob_token_ids: List[List[int]]      # Token IDs for logprobs
    logprobs: List[List[float]]             # Actual logprob values
    sampled_token_ranks: List[List[int]]    # Rank of sampled tokens
    cu_num_generated_tokens: Optional[List[int]] = None
```

**Usage Pattern:**

``` python
def sample_tokens(self, logits: torch.Tensor, sample_metadata: SamplingMetadata) -> ModelRunnerOutput:
    # Sample tokens from logits
    sampled_tokens = torch.argmax(logits[:, -1, :], dim=-1)  # [batch_size]

    # Convert to required format
    sampled_token_ids = [[token.item()] for token in sampled_tokens]

    # Create logprobs if requested
    logprobs = None
    if sample_metadata.max_num_logprobs > 0:
        logprobs = self._compute_logprobs(logits, sampled_tokens, sample_metadata)

    return ModelRunnerOutput(
        req_ids=self.current_batch_req_ids,
        req_id_to_index=self.current_req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        prompt_logprobs_dict={},
        pooler_output=[None] * len(sampled_token_ids),
        kv_connector_output=None,
    )
```

### SamplingMetadata

The `SamplingMetadata` contains all information needed for token sampling, including sampling parameters and constraints.

**Structure:**

``` python
@dataclass
class SamplingMetadata:
    """Metadata for token sampling operations."""

    # Sampling strategy indicators
    all_greedy: bool                    # True if all requests use greedy sampling
    all_random: bool                    # True if all requests use random sampling

    # Sampling parameters (None means not used)
    temperature: Optional[torch.Tensor]  # Temperature per request
    top_p: Optional[torch.Tensor]       # Top-p values per request  
    top_k: Optional[torch.Tensor]       # Top-k values per request

    # Random number generation
    generators: Dict[int, torch.Generator] # RNG generators per request

    # Logprob configuration
    max_num_logprobs: Optional[int]     # Max logprobs to return

    # Penalty parameters
    no_penalties: bool                  # True if no penalties applied
    frequency_penalties: torch.Tensor  # Frequency penalty per request
    presence_penalties: torch.Tensor   # Presence penalty per request
    repetition_penalties: torch.Tensor # Repetition penalty per request

    # Token constraints
    allowed_token_ids_mask: Optional[torch.Tensor] # Allowed tokens mask
    bad_words_token_ids: Dict[int, List[List[int]]] # Forbidden token sequences

    # Processing pipeline
    logitsprocs: LogitsProcessors       # Logits processing pipeline
```

**Usage Pattern:**

``` python
def _apply_sampling(self, logits: torch.Tensor, metadata: SamplingMetadata) -> torch.Tensor:
    """Apply sampling strategy based on metadata."""

    # Apply temperature scaling
    if metadata.temperature is not None:
        logits = logits / metadata.temperature.unsqueeze(-1)

    # Apply penalties
    if not metadata.no_penalties:
        logits = self._apply_penalties(logits, metadata)

    # Apply top-k filtering
    if metadata.top_k is not None:
        logits = self._apply_top_k(logits, metadata.top_k)

    # Apply top-p filtering  
    if metadata.top_p is not None:
        logits = self._apply_top_p(logits, metadata.top_p)

    # Sample tokens
    if metadata.all_greedy:
        return torch.argmax(logits, dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

### KVCacheSpec & KVCacheConfig

These structures define the configuration and specifications for KV cache management.

**KVCacheSpec:**

``` python
@dataclass
class FullAttentionSpec(KVCacheSpec):
    """Specification for full attention KV cache."""

    block_size: int                     # Tokens per cache block
    num_kv_heads: int                   # Number of key-value heads
    head_size: int                      # Size of each attention head
    dtype: torch.dtype                  # Data type for cache tensors

    # Optional parameters
    sliding_window: Optional[int] = None        # Sliding window size
    attention_chunk_size: Optional[int] = None  # Attention chunking
```

**KVCacheConfig:**

``` python
@dataclass
class KVCacheConfig:
    """Configuration for KV cache initialization."""

    kv_cache_groups: List[KVCacheGroup]  # Cache groups configuration

@dataclass
class KVCacheGroup:
    """Configuration for a group of KV cache layers."""

    kv_cache_spec: KVCacheSpec          # Specification for this group
    layer_names: List[str]              # Layers using this spec
    num_blocks: int                     # Number of cache blocks to allocate
```

**Usage Pattern:**

``` python
def get_kv_cache_spec(self) -> Dict[str, KVCacheSpec]:
    """Return KV cache specifications for all layers."""

    # Create spec based on model configuration
    spec = FullAttentionSpec(
        block_size=16,  # Neuron-optimized block size
        num_kv_heads=self.model_config.get_num_kv_heads(),
        head_size=self.model_config.get_head_size(),
        dtype=torch.float16,
    )

    # Apply to all attention layers
    specs = {}
    for layer_idx in range(self.model_config.num_hidden_layers):
        layer_name = f"model.layers.{layer_idx}.self_attn"
        specs[layer_name] = spec

    return specs
```

### AttentionMetadata

The `AttentionMetadata` contains information needed for attention computation, including sequence lengths and KV cache locations.

**Structure:**

``` python
@dataclass
class AttentionMetadata:
    """Metadata for attention computation."""

    # Sequence information
    seq_lens: torch.Tensor              # Length of each sequence
    seq_lens_tensor: torch.Tensor       # Sequence lengths as tensor
    max_seq_len: int                    # Maximum sequence length in batch

    # KV cache information
    block_tables: torch.Tensor          # Block table for each sequence
    slot_mapping: torch.Tensor          # Slot mapping for KV cache

    # Attention masks and positions
    context_lens: torch.Tensor          # Context length per sequence
    query_lens: torch.Tensor            # Query length per sequence

    # Performance optimization
    use_cuda_graph: bool = False        # Whether to use CUDA graphs
```

**Usage in Model Forward:**

``` python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    """Model forward pass with attention metadata."""

    # Use attention metadata for efficient attention computation
    hidden_states = self.embed_tokens(input_ids)

    for layer_idx, layer in enumerate(self.layers):
        hidden_states = layer(
            hidden_states,
            positions=positions,
            kv_cache=kv_caches[layer_idx],
            attn_metadata=attn_metadata,
        )

    logits = self.lm_head(hidden_states)
    return logits
```

### Working with Data Structures

**Best Practices:**

1. **Request ID Consistency**: Always maintain the same request ID ordering between SchedulerOutput processing and ModelRunnerOutput creation.
2. **Tensor Device Management**: Ensure all tensors are on the correct device (CPU for Neuron, CUDA for GPU).
3. **Batch Size Handling**: Handle variable batch sizes gracefully, including empty batches.
4. **Memory Efficiency**: Reuse tensors and avoid unnecessary copies, especially for large KV caches.
5. **Error Handling**: Validate data structure contents and handle edge cases (empty sequences, invalid tokens, etc.).

**Example Integration:**

``` python
class NeuronModelRunner:
    def __init__(self):
        self.current_batch_req_ids = []
        self.current_req_id_to_index = {}

    def execute_model(self, scheduler_output: SchedulerOutput) -> torch.Tensor:
        # Extract and store request information
        self._extract_batch_info(scheduler_output)

        # Prepare inputs
        input_ids, positions, attn_metadata = self._prepare_inputs(scheduler_output)

        # Execute model
        return self.model(input_ids, positions, self.kv_caches, attn_metadata)

    def sample_tokens(self, logits: torch.Tensor, metadata: SamplingMetadata) -> ModelRunnerOutput:
        # Sample using stored batch information
        sampled_tokens = self._sample_with_metadata(logits, metadata)

        return ModelRunnerOutput(
            req_ids=self.current_batch_req_ids,
            req_id_to_index=self.current_req_id_to_index,
            sampled_token_ids=sampled_tokens,
            logprobs=None,  # Simplified
            prompt_logprobs_dict={},
            pooler_output=[None] * len(sampled_tokens),
            kv_connector_output=None,
        )
```

This comprehensive reference provides the foundation for implementing production-ready vLLM integrations with custom hardware backends like AWS Neuron.
