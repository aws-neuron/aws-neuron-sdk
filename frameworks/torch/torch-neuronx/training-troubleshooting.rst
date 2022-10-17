.. _pytorch-neuron-traning-troubleshooting:

PyTorch Neuron (``torch-neuronx``) for Training Troubleshooting Guide
=====================================================================

.. contents:: Table of contents
   :local:
   :depth: 2


This document shows common issues users may encounter while using
PyTorch-Neuron and provides guidance how to resolve or work-around them.

General Troubleshooting
-----------------------

For XLA-related troubleshooting notes :ref:`How to debug models in PyTorch
Neuron on Trainium <pytorch-neurong-debug>`
and `PyTorch-XLA troubleshooting
guide <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md>`__.

If your multi-worker training run is interrupted, you may need to kill
all the python processes (WARNING: this kills all python processes and
reload the driver):

.. code:: bash

   killall -9 python
   killall -9 python3
   sudo rmmod neuron; sudo modprobe neuron

To turn on RT debug:

.. code:: python

   os.environ["NEURON_RT_LOG_LEVEL"] = "INFO"

To turn on Neuron NCCL debug:

.. code:: python

   os.environ["NCCL_DEBUG"] = "WARN"
   os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

Possible Error Conditions
-------------------------

Non-Fatal Error OpKernel ('op: "TPU*" device_type: "CPU"')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During execution using PyTorch Neuron, you may see these non-fatal error messages:

.. code:: bash

    E tensorflow/core/framework/op_kernel.cc:1676] OpKernel ('op: "TPURoundRobin" device_type: "CPU"') for unknown op: TPURoundRobin
    E tensorflow/core/framework/op_kernel.cc:1676] OpKernel ('op: "TpuHandleToProtoKey" device_type: "CPU"') for unknown op: TpuHandleToProtoKey

They don't affect operation of the PyTorch Neuron and can be ignored.

XLA runtime error: "Invalid argument: Cannot assign a device for operation"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    RuntimeError: tensorflow/compiler/xla/xla_client/xrt_computation_client.cc:490 : Check failed: session->session()->Run(session_work->feed_inputs, session_work->outputs_handles, &outputs) == ::tensorflow::Status::OK() (INVALID_ARGUMENT: Cannot assign a device for operation XRTAllocateFromTensor: {{node XRTAllocateFromTensor}} was explicitly assigned to /job:localservice/replica:0/task:0/device:TPU:0 but available devices are [ /job:localservice/replica:0/task:0/device:CPU:0, /job:localservice/replica:0/task:0/device:TPU_SYSTEM:0, /job:localservice/replica:0/task:0/device:XLA_CPU:0 ]. Make sure the device specification refers to a valid device.
	 [[XRTAllocateFromTensor]] vs. OK)
      *** Begin stack trace ***
         tensorflow::CurrentStackTrace()

         xla::util::MultiWait::Complete(std::function<void ()> const&)

         clone
      *** End stack trace ***

The above error indicates that the framework was not able to initialize the neuron runtime. If you get
the above error, check for the following:

1. No other process is taking the neuron cores. If yes, you may have to kill that process.

2. If no process is running, try reloading the driver using ``sudo rmmod neuron; sudo modprobe neuron``


Error: “Could not start gRPC server”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you get “Could not start gRPC server” error, please check if there
are any leftover python processes from a previous interrupted run and
terminate them before restarting run.

.. code:: bash

   E0207 17:22:12.592127280   30834 server_chttp2.cc:40]        {"created":"@1644254532.592081429","description":"No address added out of total 1 resolved","file":"external/com_github_grpc_grpc/src/core/ext/t
   ransport/chttp2/server/chttp2_server.cc","file_line":395,"referenced_errors":[{"created":"@1644254532.592078907","description":"Failed to add any wildcard listeners","file":"external/com_github_grpc_grpc/s
   rc/core/lib/iomgr/tcp_server_posix.cc","file_line":342,"referenced_errors":[{"created":"@1644254532.592072626","description":"Unable to configure socket","fd":10,"file":"external/com_github_grpc_grpc/src/c
   ore/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":216,"referenced_errors":[{"created":"@1644254532.592068939","description":"Address already in use","errno":98,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":189,"os_error":"Address already in use","syscall":"bind"}]},{"created":"@1644254532.592078512","description":"Unable to configure socket"
   ,"fd":10,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":216,"referenced_errors":[{"created":"@1644254532.592077123","description":"Address already in
    use","errno":98,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/tcp_server_utils_posix_common.cc","file_line":189,"os_error":"Address already in use","syscall":"bind"}]}]}]}
   2022-02-07 17:22:12.592170: E tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:545] Unknown: Could not start gRPC server


Failed compilation result in the cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All compilation results are by default saved in ``Neuron Persistent Cache``. If the Neuron Compiler
fails to compile a graph, we save the failed result in the cache. The reason for doing so is, if
the user tries to run the same script, we want the users to error out early rather than wait for
the compilation to progress and see an error at the later stage. However, there could be certain
cases under which a failed compilation may be do you some environment issues. One possible reason
of failure could be, during compilation the process went out of memory. This can happen if you are
running multiple processes in parallel such that not enough memory is available for compilation of
graph. Failure due to such reasons can be easily mitigated by re-running the compilation. In case,
you want to retry a failed compilation, you can do that by passing ``--retry_failed_compilation``
as follows:

.. code:: python

   os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --retry_failed_compilation'

This would retry the compilation and would replace a failed result in the cache with a
successful compilation result.


Compilation error: “Expect ap datatype to be of type float32 float16 bfloat16 uint8”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If an XLA example fails to run because of failed compilation and one of
the error messages is “Expect ap datatype to be of type float32 float16
bfloat16 uint8”, then please set the environment variable
``XLA_USE_32BIT_LONG=1`` in your script:

.. code:: python

    os.environ['XLA_USE_32BIT_LONG'] = '1'

.. code:: bash

   11/18/2021 04:51:25 PM WARNING 34567 [StaticProfiler]: matmul-based transposes inserted by penguin takes up 93.66 percent of all matmul computation
   terminate called after throwing an instance of 'std::runtime_error'
     what():  === BIR verification failed ===
   Reason: Expect ap datatype to be of type float32 float16 bfloat16 uint8
   Instruction: I-545-0
   Opcode: Matmult
   Input index: 0
   Argument AP:
   Access Pattern: [[1,8],[1,1],[1,1]]
   Offset: 0
   Memory Location: {compare.85-t604_i0}@SB<0,0>(8x2)#Internal DebugInfo: <compare.85||uint16||UNDEF||[8, 1, 1]>

Compilation error: "TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When compiling MRPC fine-tuning tutorial with ``bert-large-*`` and FP32 (no XLA_USE_BF16=1) for two workers or more, you will encounter compiler error that looks like ``Error message:  TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]`` followed by ``Error class:    KeyError``. Single worker fine-tuning is not affected. This issue will be fixed in an upcoming release.

.. code:: bash

    ERROR 103915 [neuronx-cc]: ***************************************************************
    ERROR 103915 [neuronx-cc]:  An Internal Compiler Error has occurred
    ERROR 103915 [neuronx-cc]: ***************************************************************
    ERROR 103915 [neuronx-cc]:
    ERROR 103915 [neuronx-cc]: Error message:  TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]
    ERROR 103915 [neuronx-cc]:
    ERROR 103915 [neuronx-cc]: Error class:    KeyError
    ERROR 103915 [neuronx-cc]: Error location: Unknown
    ERROR 103915 [neuronx-cc]: Command line:   /home/ec2-user/aws_neuron_venv_pytorch_p37/bin/neuronx-cc --target=trn1 compile --framework XLA /tmp/MODULE_1_SyncTensorsGraph.43535_10930462900538209641_ip-10-0-9-236.us-west-2.compute.internal-425495b5-100851-5eaa91287c491.hlo.pb --output /var/tmp/neuron-compile-cache/USER_neuroncc-2.1.0.76+2909d26a2/MODULE_10930462900538209641/MODULE_1_SyncTensorsGraph.43535_10930462900538209641_ip-10-0-9-236.us-west-2.compute.internal-425495b5-100851-5eaa91287c491/a3973086-78e0-4e16-b1f3-ce8e034fd4aa/MODULE_1_SyncTensorsGraph.43535_10930462900538209641_ip-10-0-9-236.us-west-2.compute.internal-425495b5-100851-5eaa91287c491.neff --model-type=transformer --verbose=35
    ERROR 103915 [neuronx-cc]:
    ERROR 103915 [neuronx-cc]: Internal details:
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/CommandDriver.py", line 226, in neuronxcc.driver.CommandDriver.CommandDriver.run
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/commands/CompileCommand.py", line 936, in neuronxcc.driver.commands.CompileCommand.CompileCommand.run
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/commands/CompileCommand.py", line 889, in neuronxcc.driver.commands.CompileCommand.CompileCommand.runPipeline
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/commands/CompileCommand.py", line 914, in neuronxcc.driver.commands.CompileCommand.CompileCommand.runPipeline
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/commands/CompileCommand.py", line 918, in neuronxcc.driver.commands.CompileCommand.CompileCommand.runPipeline
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/Job.py", line 294, in neuronxcc.driver.Job.SingleInputJob.run
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/Job.py", line 320, in neuronxcc.driver.Job.SingleInputJob.runOnState
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/Pipeline.py", line 30, in neuronxcc.driver.Pipeline.Pipeline.runSingleInput
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/Job.py", line 294, in neuronxcc.driver.Job.SingleInputJob.run
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/Job.py", line 320, in neuronxcc.driver.Job.SingleInputJob.runOnState
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/jobs/Frontend.py", line 556, in neuronxcc.driver.jobs.Frontend.Frontend.runSingleInput
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/driver/jobs/Frontend.py", line 357, in neuronxcc.driver.jobs.Frontend.Frontend.runXLAFrontend
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/starfish/penguin/Frontend.py", line 168, in neuronxcc.starfish.penguin.Frontend.tensorizeXla
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/starfish/penguin/Frontend.py", line 241, in neuronxcc.starfish.penguin.Frontend.tensorizeXlaImpl
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/starfish/penguin/Frontend.py", line 242, in neuronxcc.starfish.penguin.Frontend.tensorizeXlaImpl
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/starfish/penguin/Frontend.py", line 264, in neuronxcc.starfish.penguin.Frontend.tensorizeXlaImpl
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/starfish/penguin/Compile.py", line 129, in neuronxcc.starfish.penguin.Compile.compile_cu
    ERROR 103915 [neuronx-cc]:   File "neuronxcc/starfish/penguin/Compile.py", line 131, in neuronxcc.starfish.penguin.Compile.compile_cu
EEEERROR 103915 [neuronx-cc]:   File "neuronxcc/starfish/penguin/targets/tonga/passes/AllocateBlocks.py", line 83, in neuronxcc.starfish.penguin.targets.tonga.passes.AllocateBlocks.AllocateBlocks._allocate
    ERROR 103915 [neuronx-cc]:
    ERROR 103915 [neuronx-cc]: Version information:
    ERROR 103915 [neuronx-cc]:   NeuronX Compiler version 2.1.0.76+2909d26a2
    ERROR 103915 [neuronx-cc]:
    ERROR 103915 [neuronx-cc]:   HWM version 2.1.0.7-64eaede08
    ERROR 103915 [neuronx-cc]:   NEFF version Dynamic
    ERROR 103915 [neuronx-cc]:   TVM not available
    ERROR 103915 [neuronx-cc]:   NumPy version 1.18.2
    ERROR 103915 [neuronx-cc]:   MXNet not available
    ERROR 103915 [neuronx-cc]:
    ERROR 103915 [neuronx-cc]: Artifacts stored in: /home/ec2-user/transformers/examples/pytorch/text-classification/neuronxcc-wxp0mcjv

NeuronCore(s) not available - Requested:1 Available:0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you see "NeuronCore(s) not available" please terminate processes
that may be holding the NeuronCores and terminate any neuron-top
sessions that are running. Also check if someone else is using the
system. Then do "sudo rmmod neuron; sudo modprobe neuron" to reload the
driver.

.. code:: bash

   2021-Nov-15 15:21:28.0231 7245:7245 ERROR NRT:nrt_allocate_neuron_cores NeuronCore(s) not available - Requested:nc1-nc1 Available:0
   2021-11-15 15:21:28.231864: F ./tensorflow/compiler/xla/service/neuron/neuron_runtime.h:1037] Check failed: status == NRT_SUCCESS NEURONPOC : nrt_init failed. Status = 1

Often when you run multi-worker training, there can be many python
processes leftover after a run is interrupted. To kill all python
processes, run the follow (WARNING: this kills all python processes on
the system) then reload the driver:

.. code:: bash

   killall -9 python
   killall -9 python3
   sudo rmmod neuron; sudo modprobe neuron

TDRV error "TDRV:exec_consume_infer_status_notification"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see TDRV error "TDRV:exec_consume_infer_status_notification", try reloading the driver using ``sudo modprobe -r neuron; sudo modprobe neuron;``.

.. code:: bash

    2022-Mar-10 18:51:19.07392022-Mar-10 18:51:19.0739 17821:17931 ERROR  TDRV:exec_consume_infer_status_notifications  17822:18046 ERROR  TDRV:exec_consume_infer_status_notifications Unexpected number of CC notifications:  mod->cc_op_count=1, cc_start_cnt=0, cc_end_cnt=0Unexpected number of CC notifications:  mod->cc_op_count=1, cc_start_cnt=0, cc_end_cnt=0

    2022-Mar-10 18:51:19.07392022-Mar-10 18:51:19.0739 17821:17931 ERROR  TDRV:exec_consume_infer_status_notifications  17822:18046 ERROR  TDRV:exec_consume_infer_status_notifications (NON-FATAL, Ignoring) inference timeout (180000 ms) on Neuron Device 0 NC 0, waiting for cc status notifications.

    (NON-FATAL, Ignoring) inference timeout (180000 ms) on Neuron Device 0 NC 1, waiting for cc status notifications.


Could not open the ndX, close device failed, TDRV not initialized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see error messages stating “Could not open the ndX” (where X is
an integer from 0..15), please run ``neuron-ls`` and ensure that you are
able to see all 16 Neuron devices in the output. If one or more devices
are missing please report the issue to aws-neuron-support@amazon.com with the instance ID and a screen capture of ``neuron-ls`` output.

::

   2021-Nov-11 15:33:20.0161  7912:7912  ERROR  TDRV:tdrv_init_mla_phase1                    Could not open the nd0
   2021-Nov-11 15:33:20.0161  7912:7912  ERROR  TDRV:tdrv_destroy_one_mla                    close device failed
   2021-Nov-11 15:33:20.0161  7912:7912  ERROR  TDRV:tdrv_destroy                            TDRV not initialized
   2021-Nov-11 15:33:20.0161  7912:7912  ERROR   NRT:nrt_init                                Failed to initialize devices, error:1
   2021-11-11 15:33:20.161331: F ./tensorflow/compiler/xla/service/neuron/neuron_runtime.h:1033] Check failed: status == NRT_SUCCESS NEURONPOC : nrt_init failed. Status = 1

Multiworker execution hangs during NCCL init
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When your multi-worker execution hangs during NCCL init, you can try to
reserve the port used by environment variable ``NEURON_RT_ROOT_COMM_ID``
by (here we use host:port localhost:48620 as an example but you can use
any free port and root node’s host IP):

.. code:: bash

   sudo sysctl -w net.ipv4.ip_local_reserved_ports=48620

Then set the environment variable ``NEURON_RT_ROOT_COMM_ID`` in your
script:

.. code:: python

   os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:48620"

.. _nrt-init-error-one-or-more-engines-are-running-please-restart-device-by-reloading-driver:

NRT init error “One or more engines are running. Please restart device by reloading driver”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see an error stating “One or more engines are running. Please
restart device by reloading driver” please follow the instruction and
reload the driver using
“\ ``sudo modprobe -r neuron; sudo modprobe neuron;``\ ”.

.. code:: bash

   2021-Nov-15 20:23:27.0280 3793:3793 ERROR TDRV:tpb_eng_init_hals_v2 CRITICAL HW ERROR: One or more engines are running. Please restart device by reloading driver:
   sudo modprobe -r neuron; sudo modprobe neuron;
   2021-Nov-15 20:23:27.0280 3793:3793 ERROR TDRV:tdrv_init_one_mla_phase2 nd0 nc0 HAL init failed. error:1

NRT error “ERROR TDRV:kbl_model_add Attempting to load an incompatible model!”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see an NRT error “ERROR TDRV:kbl_model_add Attempting to load an
incompatible model!” this means that the compiler neuronx-cc used to
compile the model is too old. See installation instruction to update to
latest compiler.

NRT error "ERROR HAL:aws_hal_sprot_config_remap_entry SPROT remap destination address must be aligned size"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see an NRT error "ERROR HAL:aws_hal_sprot_config_remap_entry SPROT remap
destination address must be aligned size", please check the kernel version and upgrade it
to the distribution's latest kernel.

For example, on Ubuntu 18.04.6 LTS, the kernel version 4.15.0-66-generic is
known to cause this error when running MLP tutorial. This is due to a known
bug in the kernel in aligned memory allocation. To fix this issue, please
upgrade your kernel to latest version (i.e. 4.15.0-171-generic):

.. code:: shell

    uname -a
    sudo apt-get update
    sudo  apt-get upgrade
    sudo apt-get dist-upgrade

Please reboot after the upgrade.  Use "uname -a" to check kernel version again after reboot.

NCCL warning : "NCCL WARN Timeout waiting for RX (waited 120 sec) - retrying"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running multi-worker training, if a graph has collective communication operator like an
``all_reduce``, it requires all the workers involved in the collective communication to load the
graph in the runtime at approximately same time. If any of the worker doesn't load the graph
within a 120 sec window from the first model load by any of the worker, you would see warnings
like ``NCCL WARN Timeout waiting for RX (waited 120 sec) - retrying``. When you see such warnings
check for the following in the log messages:

1. One of the workers is compiling a graph: In multi-worker training, there is a chance that
each worker builds a slightly different graph. This would result in cache miss and can result
in compilation. Since the compilations during training run are serialized, the first worker
can compile and load the graph with collective communication. It would then wait for 120 secs
for other works to join. If they don't show up because they are compiling their own graphs,
first worker would start throwing a warning message as above. The warning in this case is
``non-fatal`` and would go away once all workers have compiled their respective graphs and then loaded
them. To identify this scenario, look for ``No candidate found under ....`` logs around the warning.
You should also see ``.....`` which indicates compilation is in progress.

2. Server on one of the nodes crashed: In distributed training across multiple nodes, if the server on one
node crashed, the workers on other nodes would keep waiting on model load and you would see above
``timeout`` logs on those nodes. To identify if the server crashed, check if you see the following
error on any of the nodes:

::

   `RPC failed with status = "UNAVAILABLE: Socket closed" and grpc_error_string = "{"created":"@1664146011.016500243","description":"Error received from peer ipv4:10.1.24.109:37379","file":"external/com_github_grpc_grpc/src/core/lib/surface/call.cc","file_line":1056,"grpc_message":"Socket closed","grpc_status":14}", maybe retrying the RPC`

If you see the above error, then it means there is a server crash and you need to cancel the
traning run.

RPC error: "RPC failed with status = 'UNAVAILABLE: Socket closed'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When you see the above error, it means that the xrt server crashed. When you see such an error, look for
the following:

1. Check for any error logs before the ``RPC error``. That should indicate the root cause of server crash.
   Note: The actual error log might be buried because of all the ``RPC error`` logs that swamp the logs.

2. Sometimes the server can crash because of host OOM. This can happen when we are loading and saving checkpoints.
   In such cases, you only see ``RPC errors`` and no other log. You can check if any instance is going out of memory
   by using tools like `dmesg <https://man7.org/linux/man-pages/man1/dmesg.1.html>`_

Error "Assertion \`listp->slotinfo[cnt].gen <= GL(dl_tls_generation)' failed" followed by 'RPC failed with status = "UNAVAILABLE: Connection reset by peer"'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The error "Assertion \`listp->slotinfo[cnt].gen <= GL(dl_tls_generation)' failed" is intermittent and occurs when using glibc 2.26. To find out the glibc version you have, you can run ``ldd --version``. The workaround is to use Ubuntu 20 where glibc is 2.27.

.. code:: bash

   INFO: Inconsistency detected by ld.so: ../elf/dl-tls.c: 488: _dl_allocate_tls_init: Assertion `listp->slotinfo[cnt].gen <= GL(dl_tls_generation)' failed!
   INFO: 2022-10-03 02:16:04.488054: W tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc:157] RPC failed with status = "UNAVAILABLE: Connection reset by peer" and grpc_error_string = "{"created":"@1664763364.487962663","description":"Error received from peer ipv4:10.0.9.150:41677","file":"external/com_github_grpc_grpc/src/core/lib/surface/call.cc","file_line":1056,"grpc_message":"Connection reset by peer","grpc_status":14}", maybe retrying the RPC

RPC connection error: "RPC failed with status = UNAVAILABLE: Connection reset by peer" not preceded by any error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This error may not be preceded by another error like shown in the previous section.
In this case, the RPC connection error usually happens when we do distributed training across multiple nodes. When you see such error, please
wait for a few minutes. It might be because some node is taking time to setup and hence the other node is not
able to connect to it just yet. Once, all nodes are up, training should resume.

Runtime errors "Missing infer_status notification" followed by "inference timeout"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you get a timeout error like below:

.. code:: bash

    ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:4)
    ERROR  TDRV:exec_consume_infer_status_notifications (FATAL-RT-UNDEFINED-STATE) inference timeout (600000 ms) on Neuron Device 4 NC 1, waiting for execution completion notification

It maybe due to long graph execution time causing synchronization delays
exceeding the default timeout. Please try increasing the timeout to
larger value using ``NEURON_RT_EXEC_TIMEOUT`` (unit in seconds) and
see if the problem is resolved.

Protobuf Error "TypeError: Descriptors cannot not be created directly."
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you install torch-neuronx after neuronx-cc, you may get the Protobuf error "TypeError: Descriptors cannot not be created directly.". To fix this, please reinstall neuronx-cc using "pip install --force-reinstall neuronx-cc".

.. code:: bash

    Traceback (most recent call last):
      File "./run_glue.py", line 570, in <module>
        main()
      File "./run_glue.py", line 478, in main
        data_collator=data_collator,
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/transformers/trainer.py", line 399, in __init__
        callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/transformers/trainer_callback.py", line 292, in __init__
        self.add_callback(cb)
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/transformers/trainer_callback.py", line 309, in add_callback
        cb = callback() if isinstance(callback, type) else callback
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/transformers/integrations.py", line 390, in __init__
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/torch/utils/tensorboard/__init__.py", line 10, in <module>
        from .writer import FileWriter, SummaryWriter  # noqa: F401
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/torch/utils/tensorboard/writer.py", line 9, in <module>
        from tensorboard.compat.proto.event_pb2 import SessionLog
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/tensorboard/compat/proto/event_pb2.py", line 17, in <module>
        from tensorboard.compat.proto import summary_pb2 as tensorboard_dot_compat_dot_proto_dot_summary__pb2
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/tensorboard/compat/proto/summary_pb2.py", line 17, in <module>
        from tensorboard.compat.proto import tensor_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__pb2
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/tensorboard/compat/proto/tensor_pb2.py", line 16, in <module>
        from tensorboard.compat.proto import resource_handle_pb2 as tensorboard_dot_compat_dot_proto_dot_resource__handle__pb2
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/tensorboard/compat/proto/resource_handle_pb2.py", line 16, in <module>
        from tensorboard.compat.proto import tensor_shape_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__shape__pb2
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/tensorboard/compat/proto/tensor_shape_pb2.py", line 42, in <module>
        serialized_options=None, file=DESCRIPTOR),
      File "/home/ec2-user/aws_neuron_venv_pytorch_p37_exp/lib64/python3.7/site-packages/google/protobuf/descriptor.py", line 560, in __new__
        _message.Message._CheckCalledFromGeneratedFile()
    TypeError: Descriptors cannot not be created directly.
    If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
    If you cannot immediately regenerate your protos, some other possible workarounds are:
     1. Downgrade the protobuf package to 3.20.x or lower.
     2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

TDRV error "Timestamp program stop timeout"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see TDRV error "Timestamp program stop timeout", i.e. when rerunning a training script after it was interrupted, try first reloading the driver using ``sudo modprobe -r neuron; sudo modprobe neuron;`` (make sure neuron-top and/or neuron-monitor are not running).

.. code:: bash

    2022-Aug-31 04:59:21.0546 117717:117717 ERROR  TDRV:tsync_wait_eng_stop                     nd0 nc0 Timestamp program stop timeout (1000 ms)
    2022-Aug-31 04:59:21.0546 117717:117717 ERROR  TDRV:tsync_wait_nc_stop                      nd0 nc0 Error while waiting for timestamp program to end on TPB eng 0
    2022-Aug-31 04:59:21.0546 117717:117717 ERROR  TDRV:tsync_timestamps_finish                 nd0 nc0 Failed to stop neuron core
    2022-Aug-31 04:59:21.0546 117717:117717 ERROR  TDRV:tdrv_tsync_timestamps                   nd0 nc0 Failed to end timestamp sync programs
    2022-Aug-31 04:59:22.0768 117717:117717 ERROR  TDRV:tdrv_destroy                            TDRV not initialized
    2022-Aug-31 04:59:22.0768 117717:117717 ERROR   NRT:nrt_init                                Failed to initialize devices, error:5

Runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when running MRPC fine-tuning tutorial with ``bert-base-*`` model, you will encounter runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703".
This issue will be fixed in an upcoming release.

.. code:: bash

    ERROR  TDRV:mem_ref_to_addr                         invalid offset in Coalesced_memloc_Coalesced_memloc_mhlo_multiply_1337_pftranspose_40839-t81854_i0_SpillSave2711--mhlo_multiply_1337_pftranspose_40839-t81854_i1_SpillSave2712_7832--mhlo_multiply_1367_pftranspose_40827-t54524_i0_SpillSave2714_10300, 12288 < (16896 + 768)
    ERROR  TDRV:drs_expand_data_desc_model              Failed to process dma block: 1703
    ERROR  TDRV:kbl_model_add                           create_data_refill_rings() error

Compiler error "module 'numpy' has no attribute 'asscalar'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you have a newer version of numpy in the Python environment, compilations may fail with the "error module 'numpy' has no attribute 'asscalar'".
Please note the neuronx-cc has the following dependency on numpy "numpy<=1.20.0,>=1.13.3". To workaround this error, please do "pip install --force-reinstall neuronx-cc" to reinstall neuronx-cc with the proper dependencies.

.. code:: base

   ERROR 227874 [neuronx-cc]: ***************************************************************
   ERROR 227874 [neuronx-cc]:  An Internal Compiler Error has occurred
   ERROR 227874 [neuronx-cc]: ***************************************************************
   ERROR 227874 [neuronx-cc]:
   ERROR 227874 [neuronx-cc]: Error message:  module 'numpy' has no attribute 'asscalar'
   ERROR 227874 [neuronx-cc]:
   ERROR 227874 [neuronx-cc]: Error class:    AttributeError
   ERROR 227874 [neuronx-cc]: Error location: Unknown
   ERROR 227874 [neuronx-cc]: Version information:
   ERROR 227874 [neuronx-cc]:   NeuronX Compiler version 2.1.0.76+2909d26a2
   ERROR 227874 [neuronx-cc]:
   ERROR 227874 [neuronx-cc]:   HWM version 2.1.0.7-64eaede08
   ERROR 227874 [neuronx-cc]:   NEFF version Dynamic
   ERROR 227874 [neuronx-cc]:   TVM not available
   ERROR 227874 [neuronx-cc]:   NumPy version 1.23.3
   ERROR 227874 [neuronx-cc]:   MXNet not available
   ERROR 227874 [neuronx-cc]:


Import error "import _XLAC ImportError: <>/site-packages/_XLAC.cpython-38-x86_64-linux-gnu.so: undefined symbol"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you encounter a PyTorch import error "import _XLAC ImportError: <>/site-packages/_XLAC.cpython-38-x86_64-linux-gnu.so: undefined symbol" during execution, please check:
    1. TensorFlow is not installed in the Python environment. If it is installed, please uninstall it.
    2. The installed PyTorch (torch) package major/minor versions match the installed torch-neuronx package's major/minor versions (ie. 1.11). If they don't match, please install the version of PyTorch that matches torch-neuronx.

.. code:: bash

    Traceback (most recent call last):
      File "/opt/ml/mlp_train.py", line 11, in <module>
        import torch_xla.core.xla_model as xm
      File "/usr/local/lib/python3.8/site-packages/torch_xla/__init__.py", line 117, in <module>
        import _XLAC
    ImportError: /usr/local/lib/python3.8/site-packages/_XLAC.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZNK3c1010TensorImpl7stridesEv
