.. _pp_developer_guide:

Developer guide for Pipeline Parallelism (``neuronx-distributed`` )
=====================================================================

Training
^^^^^^^^

For training models with pipeline-parallelism, user needs to make few
changes to their model/training script. In the below steps, we walk through different 
changes user has to make to use pipeline parallelism.
For general changes please refer to `tensor parallel guidance <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tp_developer_guide.html>`__.

Creating Model
'''''''''''''''

To train with pipeline parallel, user needs to wrap their torch module with NeuronxDistributed's Pipeline Parallel model wrapper, i.e. ``NxDPPModel``
Let's take a look at our Llama example:

.. code:: ipython3

    # Create torch model
    config.return_dict = False
    model = transformers.LlamaForCausalLM(config)
    # Create pipeline cuts
    pipeline_cuts = create_partition(config, args)
    # Apply model wrapper
    model = NxDPPModel(
        model,
        transformer_layer_cls=LlamaDecoderLayer,
        num_microbatches=args.num_microbatches,
        output_loss_value_spec=(True, False),
        input_names=["input_ids", "attention_mask", "labels"],
        pipeline_cuts=pipeline_cuts,
        trace_file_path=args.trace_file_path,
        leaf_module_cls=[LlamaRMSNorm.__name__],
        autowrap_modules=[mappings],
        use_zero1_optimizer=args.use_zero1_optimizer,
    )
    model.move_model_to_device()

We first create the model from the Hugging Face model config. If tensor parallel needs to be applied to model
it must be done here before applying the pipeline parallel model wrapper. The next step is to create the partitions. Here
is an example to evenly partition the layers for all stages:

.. code:: ipython3

    def create_partition(config, args):
        """
        Evenly split the transformer layers between the PP ranks
        """
        assert config.num_hidden_layers % args.pipeline_parallel_size == 0
        num_layer_per_partition = config.num_hidden_layers  // args.pipeline_parallel_size
        pipeline_cuts = []
        current_cut = num_layer_per_partition - 1
        for i in range(args.pipeline_parallel_size-1):
            pipeline_cuts.append(f"model.layers.{current_cut}")
            current_cut += num_layer_per_partition
        if torch.distributed.get_rank() == 0:
            print(f"pipeline_cuts {pipeline_cuts}")
        return pipeline_cuts

Note that the pipeline cuts should be at the transformer layer module name, which 
in Llama model is indicated as ``model.layers.i`` where ``i`` is the layer index. Currently user is required to provide the pipeline cuts. 
In the future release, automated partitioning will be supported.
After pipeline cuts are decided, pipeline model wrapper is applied. Let's take a deeper look into each input of the model wrapper

- ``model``: The original Pytorch module, could be TPfied.
- ``transformer_layer_cls=LlamaDecoderLayer``: The transformer layer class, we will use it for partition
- ``num_microbatches=args.num_microbatches``: The number of microbatches we used for pipeline execution.
- ``output_loss_value_spec=(True, False)``: This tells ``NxDPPModel`` how to get the loss from the model output. In this case output is a tuple, where first value is loss and second value is something else. ``NxDPPModel`` will use loss to run backward and return loss as the output.
- ``input_names=["input_ids", "attention_mask", "labels"]``: The model input names that we will use to run training. As our partition uses FX symbolic trace to trace the model, we will use these input names to create ``concrete_args``. Usually this will be the same input as you will feed into model for the execution. For details please check https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace
- ``pipeline_cuts=pipeline_cuts``: The pipeline cuts to decide the stages
- ``leaf_module_cls=[LlamaRMSNorm.__name__]``: We can add some pytorch modules as leaf module so that FX symbolic trace won't trace it through. Here we mark the ``LlamaRMSNorm`` as one leaf module. If you hit any issue about tracing you can skip tracing that part by add the module as a leaf module here. The transformer layer module will be a leaf module by default.
- ``autowrap_modules``: This serves as the same functionality to simplify FX tracing. User can provide a **python** module here and all the methods from this python module will not be traced.
- ``use_zero1_optimizer``: When zero-1 optimizer is used, set this to True, so the PP model will understand that zero-1 optimizer will handle data parallel gradient averaging.

After applying model wrapper, ``NxDPPModel`` will partition the model based on the pipeline cuts. If the original model is not yet moved to device, we can call
``model.move_model_to_device()`` so that ``NxDPPModel`` will only move the local module to device.

Runtime execution:
'''''''''''''''''

To use pipeline runtime, user simply needs to replace their original model call with ``NxDPPModel.run_train``, rest will remain unchanged. 
Please note that the pipeline runtime will take care of both forward and backward call, so user will not need to explicitly make backward calls. 
The ``NxDPPModel.run_train`` call will return the loss that is achieved from ``output_loss_value_spec``.

Mixed precision training
------------------------
We support the torch autocast to do mixed precision, simply apply the context manager for the ``NxDPPModel.run_train`` call.
Here is an example:


.. code:: ipython3

    # replace loss, _ = model(input_ids, attention_mask, labels) with below
    with torch.autocast(enabled=args.use_amp > 0, dtype=torch.bfloat16, device_type="cuda"):
        loss = model.run_train(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


Things that require user attention:
'''''''''''''''''''''''''''''''''''

Model initialization
--------------------

When the model is large, it is easy to cause host OOM when full model is created on every Neuron core. We recommend 2 ways to deal with this situation:

Using torchdistx's deferred initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pytorch's torchdistx package (https://github.com/pytorch/torchdistx/tree/main) provides easy way to do deferred initialization. If you have torchdistx installed,
using deferred initialization is simple as below

.. code:: ipython3

    from torchdistx import deferred_init
    # Instead of model = LlamaForCausalLM(config)
    model = deferred_init.deferred_init(LlamaForCausalLM, config)

The model weights will be initialized in fake tensor mode which will not consume memory.
After applying the ``NxDPPModel`` model wrapper we will only materialize the weights that belong to the local module. 
Please be aware that the torchdistx package is not actively maintained by Meta, please use at your own risk.

Using meta device for initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeuronxDistributed also supports also offer a way to first create the model on meta device, then reinitialize it to host device with only the local modules.
To create the model on meta device, follow the below example:

.. code:: ipython3

    from neuronx_distributed.utils.model_utils import init_on_device
    with init_on_device(torch.device("meta")):
        model = LlamaForCausalLM(config)

With ``init_on_device(torch.device("meta"))`` context manager, all model weights will be create to meta device, which will not consume host memory.
Then during applying the PP model wrapper, user can pass the ``param_init_fn`` kwargs which can define how to reinit the parameter. Here is an example:

.. code:: ipython3
    
    def init_weights(module):
        from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
            if module.padding_idx:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (ParallelEmbedding, RowParallelLinear, ColumnParallelLinear)):
            module.init_weight_cpu()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
    
    model = NxDPPModel(...,param_init_fn=init_weights,...)

``param_init_fn`` should take a module as input and initialize how the weight of that module should be initialized.

Moving model to device
----------------------

When user create the model it is usually either created on CPU, or using meta device/torchdistx for delayed parameter initialization. It is important to understand 
when the delayed parameter will be materialized and how/when to move model to device.

Once the ``NxDPPModel`` wrapper is applied with the model together with the partition information, tracing and partition will happen immediately. After partition
we will materialize the local module if torchdistx is used or ``param_init_fn`` is passed. So the returned model of ``NxDPPModel`` wrapper will have local parameters on host device.

After model is wrapped with ``NxDPPModel`` user can do things that are recommended to run on CPU, e.g. loading shareded checkpoint. It is important to make sure to call ``model.move_model_to_device()``
before creating the optimizer, so that the optimizer can take the weights that are on the device. When using zero-1 optimizer, it is also required to use ``model.local_parameters()`` to create parameter groups so the optimizer can
infer the right device information from parameter groups.

Gradient checkpointing
----------------------

Gradient checkpointing (or activation checkpointing) is a common method used in deep learning to reduce memory footprint by doing 
recomputation of forward computation. The common way to apply the gradient checkpointing on XLA device is to use the torch_xla's 
`gradient checkpointing wrapper <https://github.com/pytorch/xla/blob/master/torch_xla/utils/checkpoint.py#L129>`__, which will apply an autograd function.
However FX's symbolic tracing does not understand autograd function, and as a result the checkpointing information will be ignored if the checkpoint wrapper
is traced during partition.
To handle this case, user can manually re-apply gradient checkpoint after partition. Here we provide an example to checkpoint every transformer layer
after partition.

.. code:: ipython3

    from typing import Any, Dict, Iterator, Tuple
    import torch.nn as nn

    import torch
    from torch_xla.utils.checkpoint import checkpoint as torch_checkpoint
    from neuronx_distributed.parallel_layers.parallel_state import rmsg
    from neuronx_distributed.utils.logger import get_logger
    from torch.distributed.utils import _replace_by_prefix

    logger = get_logger()

    _CHECKPOINT_WRAPPED_MODULE = "mod"
    _CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."

    class CheckPointWrapper(torch.nn.Module):
        def __init__(self, mod) -> None:
            super().__init__()
            self.mod = mod
            # state_dict post hook to remove prefix to allow loading into a
            # non-checkpoint wrapped module.
            self._register_state_dict_hook(self._post_state_dict_hook)
            # load_state_dict pre-hook to allow loading back into
            # checkpoint-wrapped module.
            self._register_load_state_dict_pre_hook(
                self._pre_load_state_dict_hook, with_module=True
            )


        def forward(self, *args, **kwargs):
            ordered_args = list(args)
            for value in kwargs.values():
                ordered_args += [value]

            # Note: checkpoint cannot accept kwargs
            return torch_checkpoint(self.mod, *ordered_args, use_reentrant=True)
        
        def named_parameters(
            self,
            *args,
            **kwargs,
        ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
            """
            Overrides :meth:`named_parameters()` to intercept parameter names and
            remove all occurrences of ``_CHECKPOINT_PREFIX``.
            """
            for param_name, param in super().named_parameters(*args, **kwargs):
                updated_name = param_name.replace(_CHECKPOINT_PREFIX, "")
                yield updated_name, param
        
        def named_modules(self,*args,**kwargs):
            for module_name, module in super().named_modules(*args, **kwargs):
                updated_name = module_name.replace(_CHECKPOINT_PREFIX, "")
                yield updated_name, module

        @staticmethod
        def _post_state_dict_hook(
            module: nn.Module,
            state_dict: Dict[str, Any],
            prefix: str,
            *args: Any,
        ) -> Dict[str, Any]:
            """
            _post_state_dict_hook() is called after the state_dict() of this
            FSDP module is executed. For ``checkpoint_wrapper``, it will strip
            checkpoint-wrapped module prefix so that this module can be loaded into
            non-checkpointed modules. It would still be able to be loaded into
            checkpoint-wrapped modules as this class adds the prefix back before
            loading the state_dict.
            """
            _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
            return state_dict
        
        @staticmethod
        def _pre_load_state_dict_hook(
            module: nn.Module,
            state_dict: Dict[str, Any],
            prefix: str,
            *args: Any,
        ) -> None:
            """
            ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()``
            is called. For ``checkpoint_wrapper``, it will add back the module
            prefix so that non-checkpointed modules can be loaded into
            checkpoint_wrapper modules properly.
            """
            _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}")

    def apply_checkpoint(dist_model, layers_to_checkpoint=None):
        checkpoint_wrapper_added = False
        if layers_to_checkpoint is not None and len(layers_to_checkpoint) == 0:
            raise RuntimeError(
                rmsg(f"invalid input layers_to_checkpoint {layers_to_checkpoint}, can't be empty")
            )
        for name, module in dist_model.local_module.named_children():
            # checkpoint layers that are provided in input
            # if layers not provide in input, then checkpoint if it is transformer layer
            if (layers_to_checkpoint and name in layers_to_checkpoint) or (
                not layers_to_checkpoint and type(module) == dist_model.transformer_layer_cls
            ):
                # add_module replaces old module with our own custom module.
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.add_module
                dist_model.local_module.add_module(name, CheckPointWrapper(module))
                checkpoint_wrapper_added = True
        if layers_to_checkpoint is not None and not checkpoint_wrapper_added:
            logger.warning(
                rmsg(f"layers_to_checkpoint {layers_to_checkpoint} do not exist in the graph")
            )
        elif layers_to_checkpoint is None and not checkpoint_wrapper_added:
            logger.warning(
                rmsg(
                    f"During applying activation checkpointing, transformer_layer_cls {dist_model.transformer_layer_cls.__name__} can not be found in stage {dist_model.pipeline_parallel_rank}, skipping..."
                )
            )

    model = NxDPPModel(...)
    # Will checkpoint every transformer layer
    apply_checkpoint(model)

``apply_checkpoint`` function will try to apply gradient checkpointing to every transformer layer. Please note we have plan to add this functionality into ``NxDPPModel`` in the future releases.


Model tracing
-------------
It is important to understand that the model cannot be partitioned without tracing.
The model tracing is currently done with FX's symbolic trace. There are `certain limitations for FX's symbolic trace <https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing>`__. So in order to avoid any tracing issue, 
we would like to trace as less operations as possible, which means that we only want to trace the structure of the model, and cut the pipeline stages on the transformer layers, we do not care how exactly the computations are in the model.
By default, we will mark all transformer layers as leaf nodes, so that the tracer will not trace inside these layers. If you have some module that might cause tracing problem, you can try to mark them as leaf nodes as well. Our previous example 
also marks the `LlamaRMSNorm` as leaf module for Llama model.

Special treatment for Hugging Face models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hugging Face offers FX support for many of its models. We will detect if user is using a Hugging Face model (by checking if the model class is `transformers.PreTrainedModel`), and if so we will use the Huggingface's FX tracer to do the symbolic trace.
The Hugging Face's tracer has implementation of many functionalities to help tracing, for details please refer to `here <https://github.com/huggingface/transformers/blob/main/src/transformers/utils/fx.py>`__.
However, please be aware that Hugging Face's tracer will check if the model class name belongs to one of the Hugging Face models. So if you create your model class based on some Huggingface model class, it is important to maintain the same class name. Below is an example:

.. code:: ipython3

    from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMHF

    # Keep the same class name as original one
    class LlamaForCausalLM(LlamaForCausalLMHF):
        ...