.. _mxnet_troubleshooting_guide:

Troubleshooting Guide for Neuron Apache MXNet (Incubating)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
   :local:
   :depth: 2


Inference Runtime Error
=======================

Out-of-memory error when calling Symbol API bind() too many times
-----------------------------------------------------------------

.. important ::

  ``NEURONCORE_GROUP_SIZES`` is being deprecated, if your application is using ``NEURONCORE_GROUP_SIZES`` please 
  see :ref:`neuron-migrating-apps-neuron-to-libnrt` for more details.

If you see out-of-memory error when using Symbol API's bind() function, please ensure that the bind() function is
called once for each desired model instance. For example, on inf1.xlarge, use Symbol API to create 4 parallel 
instances of a model that was compiled to 1 NeuronCore (--neuroncore-pipeline-cores=1), each is bound to an 
different mx.neuron(i) context where i is the NeuronCore Group index ranging from 0 to 3. Then use 4 threads to feed
the 4 instances in parallel. For example:

.. code:: python

    NUM_PARALLEL = 4
    os.environ['NEURONCORE_GROUP_SIZES'] = ','.join('1' for _ in range(NUM_PARALLEL))
       
    data_iter = []
    for i in range(NUM_PARALLEL):
        data_iter.append(mx.io.ImageRecordIter(
            path_imgrec=recfile_base, data_shape=(3, 224, 224), batch_size=1,            
            prefetch_buffer=1,
            num_parts=NUM_PARALLEL, part_index=i))

    sym, args, auxs = mx.model.load_checkpoint('resnet-50_compiled', 0)

    exec_list = []
    for i in range(NUM_PARALLEL):
        exec = sym.bind(ctx=mx.neuron(i), args=args, aux_states=auxs, grad_req='null')
        exec_list.append(exec)

    def single_thread_infer(i):
        for batch in data_iter[i]:
            img = batch.data[0]
            label = batch.label
            feed_dict = {'data': img}
            exe = exec_list[i]
            exe.copy_params_from(feed_dict)
            exe.forward()
            out = exe.outputs[0]

    future_list = []
    with futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL) as executor:
        for i in range(NUM_PARALLEL):
            future_list.append(executor.submit(single_thread_infer, i))


Inference crashed with MXNetError: InferShapeKeyword argument name xyz not found
--------------------------------------------------------------------------------

If you see MXNetError:

.. code:: bash

    mxnet.base.MXNetError: [11:55:39] src/c_api/c_api_symbolic.cc:508: InferShapeKeyword argument name xyz not found."

This is followed by a list of "Candidate arguments". This list shows all the input argument names that the model knows about, and 'xyz' is not in the list. To fix this, remove entry xyz from the feed dictionary.


Inference crashed at mx.nd.waitall() with MXNetError: Check failed: bin.dtype() == mshadow::kUint8
--------------------------------------------------------------------------------------------------

When executing Symbol API's forward function followed by mx.nd.waitall(), where MXNetError exception occurs with 'Check failed: bin.dtype() == mshadow::kUint8'.


Inference crashed with NRTD error 1002
--------------------------------------

During inference, the user may encounter an error with details "[NRTD:infer_wait] error: 1002":

.. code:: bash

    mxnet.base.MXNetError: [11:26:56] src/operator/subgraph/neuron/./neuron_util.h:1175: Check failed: rsp_wait.status().code() == 0 || rsp_wait.status().code() == 1003: Failed
    Infer Wait with Neuron-RTD Error. Neuron-RTD Status Code: 1002, details: "[NRTD:infer_wait] error: 1002
    "

Runtime errors are listed in :ref:`rtd-return-codes`. In particular, 1002 means that some invalid input has been submitted to infer, e.g. missing some of the input tensors, incorrect input tensor sizes. Please examine /var/log/syslog to see imore details on the error. For example, you may see:

.. code::

    Oct 30 19:13:39 ip-172-31-93-131 nrtd[1125]: [TDRV:io_queue_prepare_input_nonhugetlb] Unexpected input size, for data00, expected: 2097152, received: 33554432

This means that the input tensor size is larger than what the model was compiled for (i.e. the example input tensor shapes passed during compilation.


Multi-Model Server
==================


Failed to create NEURONCORE Group with GRPC Error. Status Error: 14, Error message: "Connect Failed"
----------------------------------------------------------------------------------------------------

NOTE: This error only applies to MXNet 1.5.

If the client is unable to start workers and you get a message that MMS is unable to create NeuronCore Group,
please check that Neuron RTD is running (neuron-rtd process).

.. code:: json

    {
    "code": 500,
    "type": "InternalServerException",
    "message": "Failed to start workers“
    }

.. code:: bash

    2019-10-23 19:56:23,187 [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [19:56:23] src/operator/subgraph/inferentia/./inferentia_util.h:218: Check failed: status.ok() Failed to create NeuronCore Group with GRPC Error. Status Error: 14, Error message: "Connect Failed"

Multiple MMS workers die with “Backend worker process die.” message
-------------------------------------------------------------------

.. important ::

  ``NEURONCORE_GROUP_SIZES`` is being deprecated, if your application is using ``NEURONCORE_GROUP_SIZES`` please 
  see :ref:`neuron-migrating-apps-neuron-to-libnrt` for more details.

If you run inference with MMS and get multiple messages “Backend worker process die", please ensure that the number of workers ("intial_workers") passed during load model is less than or equal to number of NeuronCores available divided by  number of NeuronCores required by model.

.. code:: bash

    com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Backend worker process die.
    com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Traceback (most recent call last):
    com.amazonaws.ml.mms.wlm.WorkerLifeCycle - File "/usr/local/lib/python3.6/site-packages/mxnet/symbol/symbol.py", line 1524, in simple_bind
    com.amazonaws.ml.mms.wlm.WorkerLifeCycle - ctypes.byref(exe_handle)))
    com.amazonaws.ml.mms.wlm.WorkerLifeCycle - File "/usr/local/lib/python3.6/site-packages/mxnet/base.py", line 252, in check_call
    com.amazonaws.ml.mms.wlm.WorkerLifeCycle - raise MXNetError(py_str(_LIB.MXGetLastError()))
    com.amazonaws.ml.mms.wlm.WorkerLifeCycle - mxnet.base.MXNetError: [00:26:32] src/operator/subgraph/neuron/./neuron_util.h:221: Check failed: 0 == create_eg_rsp.status().code() Failed to create NeuronCore Group with KRTD Error. KRTD Status Code: 4, details: ""

As indicated in :ref:`appnote-performance-tuning`, for greater flexibility user can use NEURONCORE_GROUP_SIZES to specify the groupings of NeuronCores into Neuron devices, each device consisting of one or more NeuronCores. Each worker would take a device. The total number of NeuronCores taken by all the workers should be less than or equal the total number of NeuronCores visible to neuron-rtd. This situation should be considered at full load (MMS scales up to max_workers). Additionally, to properly assign model to Neuron device, the environment NEURONCORE_GROUP_SIZES must be specified within the model server class (ie. mxnet_model_service.py in the example above). For example, add the following line within mxnet_model_service.py for model compiled to 1 NeuronCore:

.. code:: python

    os.environ['NEURONCORE_GROUP_SIZES'] = '1'

More information about max_worker limit setting can be found at `MMS Management API Documentation`_. For example, to run up to 4 workers in inf1.xlarge where 4 NeuronCores are available by default to Neuron-RTD, set max_workers to 4:

.. _MMS Management API Documentation: https://github.com/awslabs/multi-model-server/blob/master/docs/management_api.md#scale-workers

.. code:: bash

    curl -v -X PUT "http://localhost:8081/models/squeezenet_v1.1_compiled?min_worker=1?max_worker=4"

MMS throws a "mxnet.base.MXNetError: array::at" error
-----------------------------------------------------

If you see “mxnet.base.MXNetError: array::at” when running MMS please check that NDArray/Gluon API is not used as they are not supported in MXNet-Neuron.
If you would like to use NDArray or Gluon API, please upgrade to MXNet 1.8.

.. code:: bash

    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - array::at
    [INFO ] W-9000-squeezenet_v1.1_compiled com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 30
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Traceback (most recent call last):
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   File "/tmp/models/6606fa046f68a34df87f15362a7a2d9a49749878/model_handler.py", line 82, in handle
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     data = self.inference(data)
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   File "/tmp/models/6606fa046f68a34df87f15362a7a2d9a49749878/mxnet_model_service.py", line 153, in inference
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     d.wait_to_read()
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   File "/home/user/regression_venv_p3.6/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py", line 1819, in wait_to_read
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     check_call(_LIB.MXNDArrayWaitToRead(self.handle))
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   File "/home/user/regression_venv_p3.6/lib/python3.6/site-packages/mxnet/base.py", line 253, in check_call
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     raise MXNetError(py_str(_LIB.MXGetLastError()))
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - mxnet.base.MXNetError: array::at
    [INFO ] W-9000-squeezenet_v1.1_compiled-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Invoking custom service failed.

MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded
------------------------------------------------------------------------------------

NOTE: This issue is resolved in version 1.5.1.1.1.88.0 released 11/17/2020 and only applies for MXNet 1.5.

MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server. Restarting the model server may fail with "Failed to create NEURONCORE_GROUP" error:

.. code:: bash

    mxnet.base.MXNetError: [00:26:59] src/operator/subgraph/neuron/./neuron_util.h:348: Check failed:    0 == create_eg_rsp.status().code(): Failed to create NEURONCORE_GROUP with Neuron-RTD Error. Neuron-RTD Status Code: 9, details: ""

The workaround is to run “`/opt/aws/neuron/bin/neuron-cli reset`“ to clear Neuron RTD states after all models are unloaded and server is shut down before restarting the model server.

Pipeline mode is not able to execute inferences requests in parallel
--------------------------------------------------------------------

If you see that multiple executors in a neuron pipeline setup (one model compiled for more than one neuron-cores using `--neuroncore-pipeline-cores` option during compilation) are not running in parallel, please set the following MXNet's environment variables before inference to allow mxnet to execute the CPU ops in parallel. Otherwise it will be sequential and stall the executors.

``MXNET_CPU_WORKER_NTHREADS`` is used to do that. (https://mxnet.apache.org/versions/1.7.0/api/faq/env_var). Setting its value to ``__subgraph_opt_neuroncore__`` in the compiled model json will ensure that all the executors (threads) can be run in parallel.


Features only in MXNet-Neuron 1.5
---------------------------------
- Shared memory for IFMaps transfer to neuron runtime (has higher performance compared to GRPC mode)
- Neuron profiling using MXNet

Features only in MXNet-Neuron 1.8
---------------------------------
- Gluon API support
- Library mode neuron runtime