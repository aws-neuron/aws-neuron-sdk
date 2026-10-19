.. meta::
   :description: This topic guides you through your first time generating a Neuron runtime core dump when using the AWS Neuron SDK. 
   :date-modified: 09-23-2026

.. _runtime-core-dump-quickstart:

Quickstart: Generating a Neuron runtime core dump
==================================================

This topic guides you through your first time generating a Neuron runtime core dump. It will help you understand the process when using AWS Neuron during a runtime failure and debugging the state of the device. When you have completed it, you will have a core dump.

**This quickstart is for**: Advanced users

**Time to complete**: 15m

Prerequisites
---------------

* `Launch an EC2 instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`__
* Use the latest :doc:`AWS Neuron Multi-Framework DLAMI </deploy/environments/dlami>`
* `Connect to the EC2 instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html>`__
* Understand the  :doc:`AWS Neuron Kernel Interface </nki/get-started/index>`

Step 1: Setup the python virtual environment
---------------------------------------------

To run this example, you must create a Python virtual environment with the Neuron Compiler and the NKI package::

    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install neuronx-cc==2.* nki

Step 2: Implement a NKI kernel with an error
---------------------------------------------

To generate a core dump, you must run a model with a runtime error. The following script implements a NKI kernel with an out-of-bounds indirect memcopy: it gathers 64 rows from a 128-row input using the row indices ``0, 3, 6, ..., 189``, so every index of 128 or above is out of bounds. Save it to ``oob.py``::

    import nki
    import nki.isa as nisa
    import nki.language as nl
    import numpy as np

    @nki.jit
    def out_of_bounds(in_tensor):
        n, m = in_tensor.shape   # (128, 512)
        p = n // 2               # number of rows to gather

        # Row indices 0, 3, 6, ..., 3*(p-1). Indices >= n are out of range
        # on purpose to demonstrate the core dump.
        idx = nl.ndarray((p, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(idx, pattern=[[1, 1]], offset=0, channel_multiplier=3)

        out_tile = nl.ndarray((p, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
        nisa.memset(dst=out_tile, value=-1)

        # Indirect gather from HBM. oob_mode.error is the default; it is
        # spelled out here to make the intent explicit.
        nisa.dma_copy(
            dst=out_tile,
            src=in_tensor.ap(pattern=[[m, p], [1, m]], offset=0,
                             vector_offset=idx, indirect_dim=0),
            oob_mode=nisa.oob_mode.error,
        )

        output = nl.ndarray((p, m), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
        nl.store(output, out_tile)
        return output

    if __name__ == "__main__":
        in_tensor = (np.random.random_sample([128, 512]) * 100).astype(np.float32)
        output = out_of_bounds(in_tensor)

Step 3: Run the NKI kernel
---------------------------

Trigger the core dump by running the script in your virtual environment: ``python3 oob.py``.

If the NKI compiler cannot detect the target platform, set ``NEURON_PLATFORM_TARGET_OVERRIDE`` to match your instance, for example ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python3 oob.py`` (use ``trn3`` on Trn3 instances).

This leads to a runtime error and is accompanied with a ``nrt_infodump``::

    2025-Sep-19 18:57:20.782962  4444:4444  ERROR  TDRV:exec_process_custom_notification        nd0:nc0:h_model.id1001: Received notification generated at runtime: failed to run scatter/gather (indirect memory copy via vector DGE), due to out-of-bound access. model name = file.neff.
    2025-Sep-19 18:57:20.798030  4444:4444  ERROR  TDRV:exec_wait_round_robin                   [ND 0][NC 0] Out of bounds access on model file.neff
    2025-Sep-19 18:57:20.805570  4444:4444  ERROR  NMGR:dlr_infer                               Inference completed with err: 1006. mode->h_nn=1001, lnc=0
    2025-Sep-19 18:57:20.813269  4444:4444  ERROR   NRT:nrt_infodump                            Neuron runtime information - please include in any support request:
    2025-Sep-19 18:57:20.821272  4444:4444  ERROR   NRT:nrt_infodump                            ------------->8------------[ cut here ]------------>8-------------
    2025-Sep-19 18:57:20.829241  4444:4444  ERROR   NRT:nrt_infodump                            NRT version: 2.x.33931.0 (8be979e9fd075e9294c151d7cf03968058670d4c)
    2025-Sep-19 18:57:20.837226  4444:4444  ERROR   NRT:nrt_infodump                            Embedded FW version: 1.0.22039.0 (d5fbbb7781171a2d6dd5bf6bac8f71064308bb0a) loaded from "libnrtucode_extisa.so"
    2025-Sep-19 18:57:20.848129  4444:4444  ERROR   NRT:nrt_infodump                            CCOM version: 2.0.35440.0- (compat 78)
    2025-Sep-19 18:57:20.855228  4444:4444  ERROR   NRT:nrt_infodump                            NCFW version: 1.0.18253.0 (7c9806c58d468da2cd27d24d59ceaf8fa0d25e4a)
    2025-Sep-19 18:57:20.863255  4444:4444  ERROR   NRT:nrt_infodump                            Instance ID: i-0b514eadc4fec7de6
    2025-Sep-19 18:57:20.870138  4444:4444  ERROR   NRT:nrt_infodump                            Cluster ID: 0
    2025-Sep-19 18:57:20.876409  4444:4444  ERROR   NRT:nrt_infodump                            Kernel: Linux 5.10.240-218.959.amzn2int.x86_64 #1 SMP Thu Aug 7 19:38:22 UTC 2025
    2025-Sep-19 18:57:20.886375  4444:4444  ERROR   NRT:nrt_infodump                            Nodename: 9371096ea4a1
    2025-Sep-19 18:57:20.892956  4444:4444  ERROR   NRT:nrt_infodump                            Driver version: 2.x

    2025-Sep-19 18:57:20.901533  4444:4444  ERROR   NRT:nrt_infodump                            Failure: NRT_EXEC_OOB in nrt_execute()
    2025-Sep-19 18:57:20.908621  4444:4444  ERROR   NRT:nrt_infodump                            LNC: 0
    2025-Sep-19 18:57:20.914681  4444:4444  ERROR   NRT:nrt_infodump                            Visible cores: 0, 1
    2025-Sep-19 18:57:20.921135  4444:4444  ERROR   NRT:nrt_infodump                            Environment:
    2025-Sep-19 18:57:20.927398  4444:4444  ERROR   NRT:nrt_infodump                            -------------8<-----------[ cut to here ]-----------8<------------
    2025-Sep-19 18:57:21.484865  4444:4444  ERROR   NRT:nrt_execute_repeat                      Failed to execute model file.neff with status 1006

Confirmation
--------------

The core dump is generated under ``/tmp/neuron-core-dump/``::

    $ ls /tmp/neuron-core-dump/
    dt-20250917-194443-cid-0000000000000000
    $ ls /tmp/neuron-core-dump/dt-20250917-194443-cid-0000000000000000/
    i-0b514eadc4fec7de6-nd0-nc0-pid-897-tid-897-lid-0  i-0b514eadc4fec7de6-nrt-pid-897.log

The core dump creates two types of files:

* Dump of the hardware state
* Dump of the tail of Neuron runtime error logs

Next Steps
-----------

Now that you've completed this quickstart, take the core dump and dive into other topics that build off of and investigate it.

* :ref:`Explore a Neuron Runtime core dump <runtime-core-dump-deep-dive>`

