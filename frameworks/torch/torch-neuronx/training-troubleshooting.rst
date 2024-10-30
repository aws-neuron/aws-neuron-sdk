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

For setting up EFA that is needed for multi-node training, please see :ref:`setup-trn1-multi-node-execution`


For XLA-related troubleshooting notes see :ref:`How to debug models in PyTorch
Neuron <pytorch-neuronx-debug>`
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

If some process crashed during training, you can enable core dumps using ``ulimit`` command:

.. code:: bash

   ulimit -S -c unlimited

To see the type of signals that would cause core dumps, see https://www.man7.org/linux/man-pages/man7/signal.7.html.

Note that core dumps take significant amount of storage, so make sure there is enough free disk space before enabling core dumps.

On Ubuntu, if Apport is not running, core dump file name is by default "core" in the local directory. To change file location and name format, modify ``/proc/sys/kernel/core_pattern`` (see https://www.kernel.org/doc/html/latest/admin-guide/sysctl/kernel.html#core-pattern for pattern info). For example, to dump to /tmp with executable filename and process ID:

.. code:: bash

   echo '/tmp/core.%e.%p' | sudo tee /proc/sys/kernel/core_pattern

For containers, install appropriate dependencies during docker build ("apt-get update && apt-get -y install build-essential gdb") and start the container with ``--ulimit core=-1`` to enable core dump and ``-v /tmp/:/tmp/`` to ensure core dumps to ``/tmp`` are preserved when container is stopped or deleted. Dependencies can also be installed after container is started.

On Ubuntu, core dumps can also handled by Apport which is disabled by default. To enable Apport, run ``sudo service apport start``. The ``/proc/sys/kernel/core_pattern`` is updated by Apport service. After a crash, look in ``/var/log/apport.log`` for the core dump file name, which should be in located in ``/var/lib/apport/coredump/``.

Once you have the core dump, you can use gdb to debug further (for Python applications, <executable> is ``python`` or ``python3``):

.. code:: bash

   gdb <executable> <core file>

If some process (i.e. XRT server) is killed due to out-of-memory on host (i.e. you see ``Out of memory: Killed process <PID>`` in ``/var/log/syslog`` or output of ``dmesg``), there won't be any core dump generated. However, you can change to it to kernel panic mode to trigger core dump by setting ``/proc/sys/vm/panic_on_oom`` to value of 1 on the host or from inside container.

On the host where you need ``sudo`` (this change will be reflected inside the container also):

.. code:: bash

   echo 1 | sudo tee /proc/sys/vm/panic_on_oom

From inside container where ``sudo`` doesn't work (this change will be reflected on the host also):

.. code:: bash
    
   echo 1 > /proc/sys/vm/panic_on_oom


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


Compilation errors when placing NeuronCache home directory on NFS/EFS/FSx mounted drive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, NeuronCache default root directory is /var/tmp which is local to the instance you are running on. You can modify the location of the NeuronCache root directory using ``NEURON_CC_FLAGS='--cache_dir=<root dir>'``.  However, when the NeuronCache directory is placed in a directory that is part of a NFS mounted drive shared among multiple instances, you may encounter file errors such as file not found, file corruption, or KeyError when running multi-instance training:

.. code:: bash

    KeyError: 'neff_cache2/neuron-compile-cache/USER_neuroncc-1.0.48875.0+7437fbf18/MODULE_7223055628515330524/MODULE_0_SyncTensorsGraph.14_7223055628515330524_compute1-dy-training-2-1-e859998e-3035-5df63dab5ce63'

This is a result of limitations to file locking on NFS. EFS/FSx also exhibit similar limitation. The workaround is to setup separate NeuronCache root directories for each worker instance, such as ``NEURON_CC_FLAGS="--cache_dir=$HOME/neuron_cache/bert/`hostname`"``, where the home directory is shared among worker instances as in ParallelCluster.

Consider the use case of a ParallelCluster with SLURM cluster management. The home directory of the head node is shared via NFS with worker instances. Also, SLURM would terminate the idle worker instances when the cluster is configured as dynamic auto-scaling cluster, and the default cache in the terminated worker instance's /var/tmp is deleted. So to persist the cache across runs separated by a cluster idle period, we use the workaround above to create separate NeuronCache root directories for each worker instance. For example, see `BERT ParallelCluster script <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/dp_bert_hf_pretrain/run_dp_bert_large_hf_pretrain_bf16_s128.sh#L42>`__.


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

TDRV error "TDRV:tdrv_one_tmpbuf_reserve  Number of ONE TMPBUF pages requested exceeded the max number of pages allowed (requested: <N>, max allowed: 16)."
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see the TDRV error "TDRV:tdrv_one_tmpbuf_reserve  Number of ONE TMPBUF pages requested exceeded the max number of pages allowed (requested: <N>, max allowed: 16)", it maybe due to model tensors requiring more device memory then available. A solution is to try training with a smaller data batch size.

.. code:: bash

    ERROR  TDRV:tdrv_one_tmpbuf_reserve                 Number of ONE TMPBUF pages requested exceeded the max number of pages allowed (requested: 28, max allowed: 16).
    ERROR  TDRV:copy_and_stage_mr                       Failed to reserve one tmpbuf memory
    ERROR  TDRV:kbl_model_add                           copy_and_stage_mr() error
    W tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc:157] RPC failed with status = "UNAVAILABLE: Socket closed" and grpc_error_string = "{"created":"@1669183391.155135683","description":"Error received from peer ipv4:172.31.58.24:43941","file":"external/com_github_grpc_grpc/src/core/lib/surface/call.cc","file_line":1056,"grpc_message":"Socket closed","grpc_status":14}", maybe retrying the RPC


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

Import errors 'generic_type: type "IrValue" is already registered!' or 'generic_type: type "XlaBuilder" is already registered!'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you encounter a PyTorch import error 'import _XLAC ... generic_type: type "IrValue" is already registered!' or 'import _XLAC ... generic_type: type "XlaBuilder" is already registered!', please check that TensorFlow and/or JAX are not installed in the Python environment. If they are installed, please uninstall them.

Import error "import _XLAC ImportError: <>/site-packages/_XLAC.cpython-38-x86_64-linux-gnu.so: undefined symbol"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you encounter a PyTorch import error "import _XLAC ImportError: <>/site-packages/_XLAC.cpython-38-x86_64-linux-gnu.so: undefined symbol" during execution, please check:
    1. TensorFlow and/or JAX are not installed in the Python environment. If they are installed, please uninstall them.
    2. The installed PyTorch (torch) package major/minor versions match the installed torch-neuronx package's major/minor versions (ie. 1.11). If they don't match, please install the version of PyTorch that matches torch-neuronx.

.. code:: bash

    Traceback (most recent call last):
      File "/opt/ml/mlp_train.py", line 11, in <module>
        import torch_xla.core.xla_model as xm
      File "/usr/local/lib/python3.8/site-packages/torch_xla/__init__.py", line 117, in <module>
        import _XLAC
    ImportError: /usr/local/lib/python3.8/site-packages/_XLAC.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZNK3c1010TensorImpl7stridesEv

NaNs seen with transformers version >= 4.21.0 when running HF BERT fine-tuning or pretraining with XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running HuggingFace BERT (any size) fine-tuning tutorial or pretraining tutorial with transformers version >= 4.21.0 and using XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1, you will see NaNs in the loss immediately at the first step. More details on the issue can be found at `pytorch/xla#4152 <https://github.com/pytorch/xla/issues/4152>`_. The workaround is to use 4.20.0 or earlier (the tutorials currently recommend version 4.15.0) or add ``transformers.modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16`` to the Python script.


.. _trn1_ubuntu_troubleshooting:

Network Connectivity Issue on trn1/trn1n 32xlarge with Ubuntu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**

Ubuntu distributions have network connectivity issues when multiple interfaces are connected to the same subnet. trn1/trn1n 32xlarge comes with 8/16 network interfaces. (To launch trn1/trn1n with 8/16 interfaces please follow :ref:`here <setup-trn1-multi-node-execution>`)

AWS publishes a package that installs a helper service to address the issue. This service runs at the startup, creates the appropriate netplan files, updates the netplan and the the instance networking and terminates.

Note that the following fix is only required on instances launched using generic Ubuntu AMIs.  Neuron AMIs and instances launched via ParalleCluster do not require the fix.

**Patch to fix networking on a multi-interface instance**

.. code:: bash

    wget -O /tmp/aws-ubuntu-eni-helper.deb 'https://github.com/aws-samples/aws-efa-nccl-baseami-pipeline/blob/master/nvidia-efa-ami_base/networking/aws-ubuntu-eni-helper_0.3-1_all.deb?raw=true'
    sudo apt install /tmp/aws-ubuntu-eni-helper.deb -y
    sudo systemctl enable aws-ubuntu-eni-helper.service
    sudo systemctl start aws-ubuntu-eni-helper.service


**How to apply the patch?**

The following steps could be followed to resolve this issue:

* Launch trn1.32xl from AWS console (starts with ``single interface``, does not suffer from the multi-interface issue)
* Apply the patch on this newly launched single-interface instance
* Create a new AMI from this instance
* Launch an 8 or 16 interface instance using that AMI.

.. note::
    The patch installs and enables the service but does not run it.  This is intentional.  The service will run at the startup when the AMI is used to launch a multi-interface instance. 

**FAQs**

.. note::
  Neuron DLAMI has the patch installed, users are always encouraged to launch the instances using the DLAMI which does not require any fix. Please refer to the :ref:`Set Up Guide <setup-guide-index>` to know how to launch an instance using DLAMI.



"Too many open files" when running training job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When running a large model training with several workers, it can result in errors like the following.

.. code:: bash

	2023-Jun-14 19:05:29.0312 4112959:4113326 [23] bootstrap.cc:106 CCOM WARN Call to accept failed : Too many open files
	2023-Jun-14 19:05:29.0312 4112959:4113263 [14] include/socket.h:438 CCOM WARN Net : Socket creation failed : Too many open files
	2023-Jun-14 19:05:29.0312 4112959:4113326 ERROR   ENC:ncclBootstrapRecv                       failed neuronBootstrapRecv request to NCCL
	2023-Jun-14 19:05:29.0312 4112959:4113249 [12] bootstrap.cc:106 CCOM WARN Call to accept failed : Too many open files
	2023-Jun-14 19:05:29.0312 4112959:4113263 ERROR   ENC:ncclBootstrapSend                       failed neuronBootstrapSend request to NCCL2023-Jun-14 19:05:29.03122023-Jun-14 19:05:29.0312 4112959:4113270 [15] bootstrap.cc:106 CCOM WARN Call to accept failed : Too many open files

This can result when the default OS limits is low. The hard and soft limits can be set on OS using the following commands or by manually opening and setting the limits.

.. code:: bash

	sudo sed -i 'H;1h;$!d;x;/hard  *nofile/!s/$/\n* hard nofile 65536/' /etc/security/limits.conf
	sudo sed -i 'H;1h;$!d;x;/soft  *nofile/!s/$/\n* soft nofile 65536/' /etc/security/limits.conf
	sudo sed -i 's/^#*\(\*\|\s*\*\)\s*soft\s*nofile\s*[0-9]\+$/\1 soft nofile 65536/' /etc/security/limits.conf
	sudo sed -i 's/^#*\(\*\|\s*\*\)\s*hard\s*nofile\s*[0-9]\+$/\1 hard nofile 65536/' /etc/security/limits.conf
	sudo sed -i 's/^#*\(\*\|\s*\*\)\s*soft\s*nofile\s*[0-9]\+$/\1 soft nofile 65536/' /etc/security/limits.d/01_efa.conf || true
	sudo sed -i 's/^#*\(\*\|\s*\*\)\s*hard\s*nofile\s*[0-9]\+$/\1 hard nofile 65536/' /etc/security/limits.d/01_efa.conf || true

The `01_efa.conf` file is created as part of the EFA installation and needs to be updated. If EFA driver is not installed the file `01_efa.conf` doesn't exist and the sed commands will fail with `No such file or directory`. If there are other files under `limits.d` with file limits they need to be updated as well.

"undefined symbol"
^^^^^^^^^^^^^^^^^^
To maintain compatibility with the packages vended publicly in Pypi, AWS Neuron python packages contain binary extensions that are compiled with the pre-2011 libstdc++ application binary interface (ABI). If a custom version of a package - such as `torch` - is compiled using a modern compiler, it can result in "undefined symbol" errors due to mismatches between the package and AWS Neuron package. 

To support this situation, we provide alternative versions of AWS Neuron packages that are compiled according to the newer 2011 ABI. For information on how to use these packages, see :ref:`pytorch-install-cxx11`.

