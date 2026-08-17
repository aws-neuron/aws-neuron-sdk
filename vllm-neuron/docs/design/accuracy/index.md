# Accuracy

Design documentation for the accuracy debugging tools — how they are built
internally. Read these only if you want to understand, extend, or contribute
to the tool implementations.

**To use the accuracy tools**, see:

- [Debugging accuracy issues](../../model-dev/accuracy-debugging-guide.md) — methodology
- [Accuracy debugger tools](../../model-dev/how-to-use-accuracy-debugger.md) — how to run them
- [Accuracy examples](https://github.com/vllm-project/vllm-neuron/blob/HEAD/examples/vllm_neuron/accuracy/) — simplified examples for accuracy tools

**How the tools are built:**

| Topic | What it explains |
| --- | --- |
| [Accuracy debugger design](accuracy_debugging_design.md) | Accuracy debugger design and report explanation |
| [Dataset evaluation](dataset_eval_design.md) | How task-level lm_eval dataset runners work |
| [Logit validation](logit_validation_design.md) | How teacher-forced logit comparison works |
| [KV cache analysis](kv_cache_analysis_design.md) | How KV cache extraction and comparison works |
| [Tensor capture](tensor_capture_design.md) | How intermediate tensors are captured from compiled graphs |
| [Tensor compare](tensor_compare_design.md) | Two-way and three-way tensor comparison algorithms |
| [Tensor replacement](tensor_replacement_design.md) | How tensors can be injected into compiled graphs for isolation |
| [Input snapshot](input_snapshot_design.md) | How NRT-boundary input tensors are captured for off-chip replay |
| [Module test guidelines](module_test_guidelines.md) | Writing per-module accuracy tests with appropriate thresholds |

:::{toctree}
:maxdepth: 1
:hidden:

accuracy_debugging_design
dataset_eval_design
kv_cache_analysis_design
logit_validation_design
module_test_guidelines
tensor_capture_design
tensor_compare_design
tensor_replacement_design
input_snapshot_design
:::
