.. _api_nrt_async_sendrecv_h:

nrt_async_sendrecv.h
====================

Neuron Runtime Asynchronous Send/Receive API - Network communication between logical neuron cores.

**Source**: `src/libnrt/include/nrt/nrt_async_sendrecv.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h>`_

.. note::

   The Neuron Runtime Async APIs are currently in early release and may change across Neuron versions.

Functions
---------

nrt_async_sendrecv_init
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_async_sendrecv_init(int lnc);

Initialize asynchronous tensor send and receive on logical neuron core.

Logical neuron core ID is the absolute ID of the logical core on the host machine. The ID is unaffected by device remapping via docker and selection of visible logical cores.

**Parameters:**

* ``lnc`` [in] - Logical neuron core ID on the current server

**Returns:** NRT_SUCCESS if logical core has been initialized successfully, NRT_FAILURE for errors

**Note:** This function may only be called when runtime is initialized. This function must have a matching call to nrt_async_sendrecv_close() before nrt_close() is called.

**Source**: `nrt_async_sendrecv.h:48 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h#L48>`_

nrt_async_sendrecv_close
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_async_sendrecv_close(int lnc);

Closes asynchronous tensor send and receive of logical neuron core and cleans up resources.

**Parameters:**

* ``lnc`` [in] - Logical neuron core ID on the current server

**Returns:** NRT_SUCCESS if logical core has been closed successfully, NRT_FAILURE for errors

**Note:** After this function was invoked, all sendrecv communicators and requests associated with this logical neuron core are closed and cannot be accessed anymore.

**Source**: `nrt_async_sendrecv.h:64 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h#L64>`_

nrt_async_sendrecv_connect
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_async_sendrecv_connect(const char* peer_ip, int peer_lnc, int lnc, 
                                         nrt_async_sendrecv_comm_t** send_comm);

Create send communicator.

Before send communicator can be used to initiate sending a tensor, connection to receive communicator must be established. Use function nrt_async_sendrecv_test_comm() to test whether connection is established.

**Parameters:**

* ``peer_ip`` [in] - IP address of peer logical neuron core
* ``peer_lnc`` [in] - Logical neuron core ID on the peer server
* ``lnc`` [in] - Logical neuron core ID on the current server
* ``send_comm`` [out] - Pointer to send communicator

**Returns:** NRT_SUCCESS if logical core has been created successfully, NRT_RESOURCE if the number of created communicators exceeds the limit, NRT_FAILURE for other errors

**Note:** This function is thread-safe.

**Source**: `nrt_async_sendrecv.h:84 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h#L84>`_

nrt_async_sendrecv_accept
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_async_sendrecv_accept(const char* peer_ip, int peer_lnc, int lnc, 
                                        nrt_async_sendrecv_comm_t** recv_comm);

Create receive communicator.

Before receive communicator can be used to initiate receiving a tensor, connection to receive communicator must be established. Use function nrt_async_sendrecv_test_comm() to test whether connection is established.

**Parameters:**

* ``peer_ip`` [in] - IP address of peer logical neuron core
* ``peer_lnc`` [in] - Logical neuron core ID on the peer server
* ``lnc`` [in] - Logical neuron core ID on the current server
* ``recv_comm`` [out] - Pointer to receive communicator

**Returns:** NRT_SUCCESS if logical core has been created successfully, NRT_RESOURCE if the number of created communicators exceeds the limit, NRT_FAILURE for other errors

**Note:** This function is thread-safe.

**Source**: `nrt_async_sendrecv.h:104 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h#L104>`_

nrt_async_sendrecv_send_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_async_sendrecv_send_tensor(nrt_tensor_t* tensor, size_t offset, size_t length, 
                                             nrt_async_sendrecv_comm_t* send_comm, 
                                             nrt_async_sendrecv_request_t** request);

Asynchronously send a tensor.

This is a non-blocking function. This function is thread-safe. This function is only allowed to be invoked on a communicator that is successfully tested to be connected via call to nrt_async_sendrecv_test_comm().

**Parameters:**

* ``tensor`` [in] - Tensor to send from
* ``offset`` [in] - Offset into the tensor to send from
* ``length`` [in] - Number of bytes to send
* ``send_comm`` [in] - Send communicator
* ``request`` [out] - Pointer to send request

**Returns:** NRT_SUCCESS on success, NRT_INVALID_HANDLE if handle is invalid, NRT_RESOURCE if the number of pending requests exceeds the limit, NRT_FAILURE for other errors

**Source**: `nrt_async_sendrecv.h:135 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h#L135>`_

nrt_async_sendrecv_recv_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_async_sendrecv_recv_tensor(nrt_tensor_t* tensor, size_t offset, size_t length, 
                                             nrt_async_sendrecv_comm_t* recv_comm, 
                                             nrt_async_sendrecv_request_t** request);

Asynchronously receive a tensor.

This is a non-blocking function. This function is thread-safe. This function is only allowed to be invoked on a communicator that is successfully tested to be connected via call to nrt_async_sendrecv_test_comm().

**Parameters:**

* ``tensor`` [in] - Tensor to receive to
* ``offset`` [in] - Offset into the tensor to receive to
* ``length`` [in] - Number of bytes to read
* ``recv_comm`` [in] - Receive communicator
* ``request`` [out] - Pointer to receive request

**Returns:** NRT_SUCCESS on success, NRT_INVALID_HANDLE if handle is invalid, NRT_RESOURCE if the number of pending requests exceeds the limit, NRT_FAILURE for other errors

**Source**: `nrt_async_sendrecv.h:156 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h#L156>`_

nrt_async_sendrecv_test_request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_async_sendrecv_test_request(nrt_async_sendrecv_request_t* request, bool* done, size_t* size);

Test the completion status of an asynchronous request.

This function is thread-safe when invoked with different requests. This function is not allowed to be invoked concurrently by multiple threads with the same request at the same time.

**Parameters:**

* ``request`` [in] - Request to test
* ``done`` [out] - Whether the request has completed
* ``size`` [out] - Number of bytes sent/received

**Returns:** NRT_SUCCESS on success, NRT_INVALID_HANDLE if handle is invalid, NRT_TIMEOUT if the request fails to complete data transfer within time limit, NRT_FAILURE for other errors

**Source**: `nrt_async_sendrecv.h:174 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async_sendrecv.h#L174>`_
