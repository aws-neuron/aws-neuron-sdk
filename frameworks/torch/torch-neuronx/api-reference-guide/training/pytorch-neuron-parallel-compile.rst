.. _pytorch-neuronx-parallel-compile-cli:

PyTorch Neuron neuron_parallel_compile CLI (``torch-neuronx``)
==============================================================

PyTorch Neuron performs just-in-time compilation of graphs during
execution. At every step, a graph is traced. If the traced graph varies
from the previous executions, it is compiled by the neuron compiler. For
large models, the compilation time for each graph can be high. Moreover,
because of JIT, we would compile all these graphs sequentially, hence
incurring huge compilation penalty.

To reduce this compilation time during execution, the ``neuron_parallel_compile``
utility is provided as part of PyTorch Neuron installation. The
``neuron_parallel_compile`` will extract graphs from a trial run of your script,
perform parallel pre-compilation of the graphs, and populate the Neuron Cache
on disk or AWS S3 bucket with compiled graphs. Your trial run should be limited to a few steps
(eg.10-15), enough for the utility to extract the different graphs needed for
full execution. To run the utility:

``neuron_parallel_compile <run commands>``

Where ``<run commands>`` are the commands to run a short run (i.e. 10
steps) to trace training loops for pre-compilation. The example for
the run command is ``torchrun --nproc_per_node=2 <train script>``, where
train script accepts ``--steps_this_run`` option to limit number of run steps:

``neuron_parallel_compile torchrun --nproc_per_node=2 <train script> --steps_this_run=10``

NOTE: To avoid hang during ``neuron_parallel_compile`` run, please make sure to use xm.save
instead of torch.save to save checkpoints.

You may notice that the output from the model is invalid when you use
``neuron_parallel_compile``. This is because, when you initiate your training
run command with ``neuron_parallel_compile`` , the utility will run your command
with certrain env variables that would put your training script into graph
extraction mode. In this mode, no real execution is performed and the outputs
are invalid.

Once the ``neuron_parallel_compile`` finishes compilation of all graphs, it will copy
all the compilation results into the Neuron Cache.

Now if you run your script (without ``neuron_parallel_compile``), it will be faster
since the compiled graphs are already cached.

``torchrun --nproc_per_node=2 <train script>``

``Note``: Except for the option to limit number of run steps (such as ``--steps_this_run``),
the other options of ``<run commands>`` must match between the pre-compilation and
actual run. If this is not the case, you may see additional compilations during training
run because of new graphs getting generated, resulting in cache miss.

There may be additional compilations due to unreached execution paths (in case the
execution path is not reached in the first few steps of graph extraction), or changes
in parameters such as number of data parallel workers.

Two additional utility flags are provided:

``NEURON_PARALLEL_COMPILE_MAX_RETRIES`` :

-  Set the maximum number of retries when using ``neuron_parallel_compile`` tool.
   If set to N, the tool will try compilation N more time(s) if the first graph compilation
   failed. Example: Set NEURON_PARALLEL_COMPILE_MAX_RETRIES=1 when precompiling on
   trn1.2xlarge where there's limited host memory and CPU resources.
   Default is 0.

``NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE`` :

- When using neuron_parallel_compile, if you want to ignore the error in training script
  and compile the accumulated HLO graphs, you can do so by setting this environment variable.
  Example: If NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE=1 is set when using ``neuron_parallel_compile``,
  a crash in the training script would be ignored and the graphs collected upto the crash would be
  compiled.

``NEURON_COMPILE_CACHE_URL``:

-  Set the cache url when using ``neuron_parallel_compile`` tool.
   If starts with ``s3://``, it will use AWS S3 as cache backend. Otherwise it will use
   local disk cache. Default is ``/var/tmp/neuron-compile-cache``.


S3 as Neuron Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In ``neuron_parallel_compile``, user can either do ``NEURON_COMPILE_CACHE_URL=<cache_url>`` 
or ``NEURON_CC_FLAGS=="--cache_dir=<cache_url>"`` to set the cache URL (``--cache_dir`` is prioritized if both are set). The default cache location
is on disk as ``/var/tmp/neuron-compile-cache``. Now we support using **AWS S3** as remote cache backend besides local disk cache.
When ``<cache_url>`` starts with ``s3://``, it will put cache files into S3, which can be easily shared
between tasks running on different machines. **Note: you need to create S3 bucket before using S3 cache, and also have read/write permissions to the bucket.**

Seperate collection and compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For cases like finetuning, there could be multiple independent training tasks running on different nodes,
but shares a lot compilation graphs in common. ``neuron_parallel_compile`` provides commands to seperate 
graph collection and compilation, so users can collect all graphs across different training session in advance to avoid duplicated compilation.

Please make sure to use the same neuronx-cc compilier version when invoking these two commands.

``neuron_parallel_compile --command collect <run_script>`` will collect graphs with dry-run into cache.

``neuron_parallel_compile --command compile <run_script>`` will compile the collected graphs from cache, and store
compiled result (NEFFs) back into cache.

``neuron_parallel_compile --command clean`` will remove cached files.

**Note: if --command is not specified, will do collection and then compilation by default**.

Analyze operations support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The analyze command checks the support of operations within the training script by checking each operator against neuronx-cc.
It is only supported for PyTorch models. The output of the tool will be available as result.json within the output location.

``neuron_parallel_compile --command analyze python3 training_script.py``

Optional Arguments:

``--analyze-output ANALYZE_OUTPUT_LOCATION``
Only supported for --command analyze. Path to location where output will be persisted.
Default: cwd/model_analysis_result

``--analyze-verbosity {1,2}``
Only supported for --command analyze. Level of information to be included within the output.
1: add XLA operator information into the results.
2: add aten metadata into results.
Default: 2

The tutorial for ``analyze`` can be found :ref:`here <torch-analyze-for-training-tutorial>`