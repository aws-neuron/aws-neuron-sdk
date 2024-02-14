.. _pytorch-neuronx-parallel-compile-cli:

PyTorch NeuronX neuron_parallel_compile CLI
=============================================

PyTorch NeuronX performs just-in-time compilation of graphs during
execution. At every step, a graph is traced. If the traced graph varies
from the previous executions, it is compiled by the neuron compiler. For
large models, the compilation time for each graph can be high. Moreover,
because of JIT, we would compile all these graphs sequentially, hence
incurring huge compilation penalty.

To reduce this compilation time during execution, the ``neuron_parallel_compile``
utility is provided as part of PyTorch Neuron installation. The
``neuron_parallel_compile`` will extract graphs from a trial run of your script,
perform parallel pre-compilation of the graphs, and populate the :ref:`Neuron Persistent Cache <neuron-caching>`
on disk or in AWS S3 bucket with compiled graphs.
Your trial run should be limited to a few steps
(eg.10-15), enough for the utility to extract the different graphs needed for
full execution. To run the utility:

``neuron_parallel_compile <run commands>``

Where ``<run commands>`` are the commands to run a short run (i.e. 10
steps) to trace training loops for pre-compilation. The example for
the run command is ``torchrun --nproc_per_node=2 <train script>``, where
train script accepts ``--steps_this_run`` option to limit number of run steps:

``neuron_parallel_compile torchrun --nproc_per_node=2 <train script> --steps_this_run=10``

You may notice that the output from the model is invalid when you use
``neuron_parallel_compile``. This is because when you initiate your training
run command with ``neuron_parallel_compile``, the utility will run your command
with environment variables that puts your training script into graph
extraction mode. In this mode, no real execution is performed and the outputs
are invalid. You will also see outputs similar to the following about the compile cache path and the
extracted graphs:

.. code:: bash

   INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
   INFO ||NEURON_CC_WRAPPER||: Extracting graphs (/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_9219523464496887986+abb26765/model.hlo.pb) for ahead-of-time parallel compilation. No compilation was done.

After the trial execution ends and the graphs are extracted, ``neuron_parallel_compile`` would launch multiple compilation processes in parallel to compile all these graphs. Compiled graphs (NEFFs) are inserted into the Neuron Persistent Cache. You will also see outputs similar to the following about the compile cache path, the list of graphs (HLOs) to be compiled, and the running statistics of compiled graphs (count of remaining graphs, locked graphs, failed graphs, done compiled graphs).

.. code:: bash

    INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
    INFO ||NEURON_CACHE||: Current remaining items are 5, locked are 0, failed are 0, done are 0, total is 5
    INFO ||NEURON_PARALLEL_COMPILE||: master grab hlos to compile: ['/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_8068656800389078395+abb26765/model.hlo.pb', '/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_17109392703413819652+abb26765/model.hlo.pb', '/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_9219523464496887986+abb26765/model.hlo.pb', '/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_16969875447143373016+abb26765/model.hlo.pb', '/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_3000743782456078279+abb26765/model.hlo.pb']
    ...
    INFO ||NEURON_CACHE||: Current remaining items are 0, locked are 0, failed are 0, done are 5, total is 5

After all compilations are completed, a compilation summary is shown:

.. code:: bash

   INFO: 2023-08-24 20:21:11.000895:  161136  INFO ||NEURON_PARALLEL_COMPILE||: {
   INFO:     "compilation_summary": {
   INFO:         "true": 2
   INFO:     },
   INFO:     "compilation_report": {
   INFO:         "/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_1970132581169579119+abb26765/model.hlo.pb": {
   INFO:             "status": true,
   INFO:             "retry": 0
   INFO:         },
   INFO:         "/var/tmp/neuron-compile-cache/neuronxcc-2.0.0.22266a0+a69f71e55/MODULE_16141953836240613513+abb26765/model.hlo.pb": {
   INFO:             "status": true,
   INFO:             "retry": 0
   INFO:         }
   INFO:     }
   INFO: }
   INFO: 2023-08-24 20:21:11.000895:  161136  INFO ||NEURON_PARALLEL_COMPILE||: Total graphs: 2
   INFO: 2023-08-24 20:21:11.000895:  161136  INFO ||NEURON_PARALLEL_COMPILE||: Total successful compilations: 2
   INFO: 2023-08-24 20:21:11.000895:  161136  INFO ||NEURON_PARALLEL_COMPILE||: Total failed compilations: 0

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

Each precompilation command or actual script execution command above can be prefixed with ``NEURON_COMPILE_CACHE_URL=<cache URL>`` or ``NEURON_CC_FLAGS="--cache_dir=<cache URL>"`` to specify a different cache location than the default (with ``--cache_dir`` taking precedence over ``NEURON_COMPILE_CACHE_URL`` if both are specified). Alternatively, the cache URL can also be specify in Python code using:

.. code:: python

    os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + "--cache_dir=<cache URL>"

You need to specify the same cache URL for both the precompilation command (using ``neuron_parallel_compile``) and the actual script execution command if you want the previously compiled and cached graphs to be used for actual script execution.

The environment variables below are available to help modify ``neuron_parallel_compile`` behavior:

``NEURON_PARALLEL_COMPILE_MAX_RETRIES`` :

-  Set the maximum number of retries when using :ref:`Neuron Persistent Cache <neuron-caching>` or :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
   If set to N, the tool will try compilation N more time(s) if the first graph compilation
   failed. Example: Set NEURON_PARALLEL_COMPILE_MAX_RETRIES=1 when precompiling on
   trn1.2xlarge where there's limited host memory and CPU resources.
   Default is 0.

``NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE`` :

- When using :ref:`Neuron Persistent Cache <neuron-caching>` or :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>` , if you want to ignore the error in training script
  and compile the accumulated HLO graphs, you can do so by setting this environment variable.
  Example: If NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE=1 is set when using ``neuron_parallel_compile``,
  a crash in the training script would be ignored and the graphs collected up to the crash would be
  compiled.

``NEURON_COMPILE_CACHE_URL``:

-  Set the :ref:`Neuron Persistent Cache <neuron-caching>` URL or :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
   If starts with ``s3://``, it will use AWS S3 as cache backend. Otherwise it will use
   local disk cache. Default is ``/var/tmp/neuron-compile-cache``.
   If this is specified together with ``cache_dir=<cache_url>`` option via ``NEURON_CC_FLAGS``, the ``--cache_dir`` option takes precedence.


Debugging with Neuron Persistent Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A graph compilation can fail because of a compilation error or an environment issue (for example, compilation is interrupted by ctrl-C). The graph would be marked as failed and subsequent rerun would encounter message like below:

.. code:: bash

    INFO ||NCC_WRAPPER||: Got a cached failed neff at /var/tmp/neuron-compile-cache/neuronxcc-2.8.0.25+a3ad0f342/MODULE_12486829708343293975+d41d8cd9/model.neff. Will skip compilation, please set --retry_failed_compilation for recompilation. 

To retry compilation,
add ``--retry_failed_compilation`` in ``NEURON_CC_FLAGS`` environment variable. This will retry the compilation even if the graph was previously marked as failed compilation.

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --retry_failed_compilation'

See :ref:`Neuron Persistent Cache <neuron-caching>` for more information.

Separate collection and compilation commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For cases like finetuning, there could be multiple independent training tasks running on different nodes
and sharing many compilation graphs in common. ``neuron_parallel_compile`` provides commands to separate 
the graph collection and compilation phases, so users can collect all graphs across different training sessions in advance to avoid duplicate compilations.

To only collect the graphs from trial executions of training scripts into Neuron Persistent Cache:

.. code:: bash

    neuron_parallel_compile --command collect <run_script>

To compile the graph previously collected using ``collect`` command and store compiled result (NEFFs) back into Neuron Persistent Cache (make sure to use the same neuronx-cc compiler version as during the graph collection step):

.. code:: bash

    ``neuron_parallel_compile --command compile <run_script>``

Note: if ``--command`` is not specified, ``neuron_parallel_compile`` will do both collection and compilation phases by default.

Cache maintenance commands
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following commands are available to help maintain the cache.

.. warning::
   
    Make sure no running process is using the cache when you use ``clean`` or ``clear-locks`` command because it can cause cache errors.

To clean cached files:

.. code:: bash

    # WARNING: Make sure no running process is using the cache
    neuron_parallel_compile --command clean
    
To clear file locks left behind when a ``neuron_parallel_compile`` execution was interrupted:

.. code:: bash

    # WARNING: Make sure no running process is using the cache
    neuron_parallel_compile --command clear-locks

Each command above can be prefixed with ``NEURON_COMPILE_CACHE_URL=<cache URL>`` or ``NEURON_CC_FLAGS="--cache_dir=<cache URL>"`` to specify a different cache location than the default.

.. note::

   Currently there's no automatic maintenance of cache size either on disk or in S3. Please delete files (i.e. older compiler versions) as necessary to keep cache size within your limit.

Analyze operations support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The analyze command checks the support of operations within the training script by checking each operator against neuronx-cc.
It is only supported for PyTorch models. The output of the tool will be available as result.json within the output location.

.. code:: bash

    neuron_parallel_compile --command analyze python3 training_script.py

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
