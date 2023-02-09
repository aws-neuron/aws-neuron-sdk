.. _custom-ops-api-ref-guide:

Custom Operators API Reference Guide [Experimental]
===================================================

This page provides the documentation for the C++ API available to creators of Neuron custom C++ operators (see :ref:`neuron_c++customops`).

.. contents:: Table of contents
   :local:
   :depth: 1


Tensor Library
--------------

The tensor library used for Neuron custom C++ operators is based upon the PyTorch ATen tensor library. This includes the core Tensor class as well as select operations defined below. Users need to include the ``<torch/torch.h>`` header to access the tensor library. A small example of using the tensor library looks as follows.

.. code-block:: c++

    #include <torch/torch.h>
    ...
    torch::Tensor a = torch::zeros({32, 32, 3}, torch::kFloat);

Tensor Factory Functions
^^^^^^^^^^^^^^^^^^^^^^^^

The tensor factory functions provide different means for creating new tensors.

They each take in a ``size`` argument that specifies the size of each dimension of the tensor created (with the exception of ``eye``, which takes in two int64's and creates a strictly 2-dimensional identity matrix.)

``c10::TensorOptions`` allows the specification of optional properties for the tensor being created. Currently, only the ``dtype`` property has an effect on tensor construction, and it must be specified. Other properties, such as ``layout`` may be supported in the future.
The example above shows a common way to use factory functions.

The following dtypes are supported:

* torch::kFloat
* torch::kBFloat16
* torch::kHalf
* torch::kInt
* torch::kChar
* torch::kLong
* torch::kShort
* torch::kByte

.. cpp:function:: torch::Tensor empty(torch::IntArrayRef size, c10::TensorOptions options)

    Creates a tensor filled with uninitialized data, with the specified size and options. Slightly faster than other factory functions since it skips writing data to the tensor.

.. cpp:function:: torch::Tensor full(torch::IntArrayRef size, const Scalar & fill_value, c10::TensorOptions options)

    Creates a tensor filled with the specified ``fill_value``, with the specified size and options.

.. cpp:function:: torch::Tensor zeros(torch::IntArrayRef size, c10::TensorOptions options)

    Creates a tensor filled with zeros, with the specified size and options.

.. cpp:function:: torch::Tensor ones(torch::IntArrayRef size, c10::TensorOptions options)

    Creates a tensor filled with ones, with the specified size and options.

.. cpp:function:: torch::Tensor eye(int64_t n, int64_t m, c10::TensorOptions options)

    Creates a 2-D tensor with ones on the diagonal and zeros elsewhere.

Tensor Operation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tensor library provides commonly used operations defined below. The tensor operation functions do not support broadcasting; the shape of the operands must match if applicable. 

The library provides two styles of functions for each tensor operation. For functions ending with ``_out``, a tensor with the proper size must be provided to which the output is written. This is illustrated in the example below.

.. code-block:: c++

    torch::exp_out(t_out, t_in);

Alternatively, for functions that do not end in ``_out``, a new tensor that contains the results of the operation is allocated and returned as seen in the example below.

.. code-block:: c++

    torch::Tensor t_out = torch::exp(t_in);

.. warning:: 
    Only operations that are documented below are supported.

.. cpp:function:: torch::Tensor& abs_out(torch::Tensor &result, torch::Tensor &self)
.. cpp:function:: torch::Tensor abs(torch::Tensor& self)

    Computes the absolute value of each element in ``self``.

.. cpp:function:: torch::Tensor& ceil_out(torch::Tensor &result, torch::Tensor &self)
.. cpp:function:: torch::Tensor ceil(torch::Tensor &self)

    Computes the ceiling of the elements of ``self``, the smallest integer greater than or equal to each element.

.. cpp:function:: torch::Tensor& floor_out(torch::Tensor& result, torch::Tensor &self)
.. cpp:function:: torch::Tensor floor(torch::Tensor &self)

    Computes the floor of the elements of ``self``, the largest integer less than or equal to each element.

.. cpp:function:: torch::Tensor& sin_out(torch::Tensor& result, torch::Tensor& self)
.. cpp:function:: torch::Tensor sin(torch::Tensor& self)

    Computes the sine value of the elements of ``self``.

.. cpp:function:: torch::Tensor& cos_out(torch::Tensor& result, torch::Tensor& self)
.. cpp:function:: torch::Tensor cos(torch::Tensor& self)

    Computes the cosine value of the elements of ``self``.

.. cpp:function:: torch::Tensor& tan_out(torch::Tensor& result, torch::Tensor& self)
.. cpp:function:: torch::Tensor tan(torch::Tensor& self)

    Computes the tangent value of the elements of ``self``.

.. cpp:function:: torch::Tensor& log_out(torch::Tensor& result, torch::Tensor& self)
.. cpp:function:: torch::Tensor log(torch::Tensor& self)

    Computes the natural logarithm of the elements of ``self``.

.. cpp:function:: torch::Tensor& log2_out(torch::Tensor& result, torch::Tensor& self)
.. cpp:function:: torch::Tensor log2(torch::Tensor& self)

    Computes the base-2 logarithm of the elements of ``self``.

.. cpp:function:: torch::Tensor& log10_out(torch::Tensor& result, torch::Tensor& self)
.. cpp:function:: torch::Tensor log10(torch::Tensor& self)

    Computes the base-10 logarithm of the elements of ``self``.

.. cpp:function:: torch::Tensor& exp_out(torch::Tensor& result, torch::Tensor& self)
.. cpp:function:: torch::Tensor exp(torch::Tensor& self)

    Computes the exponential of the elements of ``self``.

.. cpp:function:: torch::Tensor& pow_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar & exponent)
.. cpp:function:: torch::Tensor& pow_out(torch::Tensor& result, const torch::Scalar& self, const torch::Tensor & exponent)
.. cpp:function:: torch::Tensor& pow_out(torch::Tensor& result, const torch::Tensor& self, const torch::Tensor & exponent)
.. cpp:function:: torch::Tensor pow(const torch::Tensor& self, const torch::Scalar & exponent)
.. cpp:function:: torch::Tensor pow(const torch::Scalar& self, const torch::Tensor & exponent)
.. cpp:function:: torch::Tensor pow(const torch::Tensor& self, const torch::Tensor & exponent)

    Takes the power of each element in ``self`` with ``exponent``. 

.. cpp:function:: torch::Tensor& clamp_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar& minval, const torch::Scalar& maxval)
.. cpp:function:: torch::Tensor clamp(const torch::Tensor& self, const torch::Scalar& minval, const torch::Scalar& maxval)

    Clamps all elements in ``self`` into the range ``[minval, maxval]``.

.. cpp:function:: torch::Tensor& add_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar &other, const torch::Scalar& alpha=1)
.. cpp:function:: torch::Tensor& add_out(torch::Tensor& result, const torch::Tensor& self, const torch::Tensor& other, const torch::Scalar& alpha=1)
.. cpp:function:: torch::Tensor add(const torch::Tensor& self, const torch::Scalar &other, const torch::Scalar& alpha=1)
.. cpp:function:: torch::Tensor add(const torch::Tensor& self, const torch::Tensor &other, const torch::Scalar& alpha=1)

    Adds ``other``, scaled by ``alpha``, to ``input``,
.. math:: 
    out = self + alpha \times other.

.. cpp:function:: torch::Tensor& sub_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar &other, const torch::Scalar& alpha=1)
.. cpp:function:: torch::Tensor& sub_out(torch::Tensor& result, const torch::Tensor& self, const torch::Tensor& other, const torch::Scalar& alpha=1)
.. cpp:function:: torch::Tensor sub(const torch::Tensor& self, const torch::Tensor &other, const torch::Scalar& alpha=1)

    Subtracts ``other``, scaled by ``alpha``, to ``input``,
.. math:: 
    out = self - alpha \times other.

.. cpp:function:: torch::Tensor& mul_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar &other)
.. cpp:function:: torch::Tensor& mul_out(torch::Tensor& result, const torch::Tensor& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor mul(const torch::Tensor& self, const torch::Scalar &other)
.. cpp:function:: torch::Tensor mul(const torch::Tensor& self, const torch::Tensor &other)

    Multiplies ``self`` by ``other``.

.. cpp:function:: torch::Tensor& div_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar &other)
.. cpp:function:: torch::Tensor& div_out(torch::Tensor& result, const torch::Tensor& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor div(const torch::Tensor& self, const torch::Scalar &other)
.. cpp:function:: torch::Tensor div(const torch::Tensor& self, const torch::Tensor &other)

    Divides ``self`` by ``other``.

.. note:: 
   For tensor-tensor bitwise operations, all the bitwise operations are elementwise between two tensors. For scalar-tensor bitwise operations, the scalar is casted to the datatype of the tensor before computing the bitwise operation.

.. cpp:function:: torch::Tensor& bitwise_and_out(torch::Tensor& result, const torch::Tensor& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor& bitwise_and_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar& other)
.. cpp:function:: torch::Tensor& bitwise_and_out(torch::Tensor& result, const torch::Scalar& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor bitwise_and(const torch::Tensor& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor bitwise_and(const torch::Tensor& self, const torch::Scalar& other)
.. cpp:function:: torch::Tensor bitwise_and(const torch::Scalar& self, const torch::Tensor& other)

    Computes the bitwise AND of ``self`` and ``other``. The input tensors must be of integral types.

.. cpp:function:: torch::Tensor& bitwise_or_out(torch::Tensor& result, const torch::Tensor& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor& bitwise_or_out(torch::Tensor& result, const torch::Tensor& self, const torch::Scalar& other)
.. cpp:function:: torch::Tensor& bitwise_or_out(torch::Tensor& result, const torch::Scalar& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor bitwise_or(const torch::Tensor& self, const torch::Tensor& other)
.. cpp:function:: torch::Tensor bitwise_or(const torch::Tensor& self, const torch::Scalar& other)
.. cpp:function:: torch::Tensor bitwise_or(const torch::Scalar& self, const torch::Tensor& other)

    Computes the bitwise OR of ``self`` and ``other``. The input tensors must be of integral types.

.. cpp:function:: torch::Tensor& bitwise_not_out(torch::Tensor& result, const torch::Tensor& self)
.. cpp:function:: torch::Tensor bitwise_not(torch::Tensor& result, const torch::Tensor& self)  

    Computes the bitwise NOT of ``self``. The input tensor must be of integral types. 

Class torch::Tensor
^^^^^^^^^^^^^^^^^^^

Constructors
""""""""""""

Users should not call the Tensor constructor directly but instead use one of the Tensor factory functions.

Member Functions
""""""""""""""""

.. cpp:function:: template<typename T, size_t N> TensorAccessor<T,N,true> accessor() const&

    Return a ``TensorAccessor`` for element-wise random access of a Tensor's elements. Scalar type and dimension template parameters must be specified. This const-qualified overload returns a read-only ``TensorAccessor``, preventing the user from writing to Tensor elements. See the Tensor Accessors section below for more details.

.. cpp:function::  template<typename T, size_t N> TensorAccessor<T,N,false> accessor() &

    Return a ``TensorAccessor`` for element-wise random access of a Tensor's elements. Scalar type and dimension template parameters must be specified. This non-const-qualified overload returns a ``TensorAccessor`` that can be used to both read and write to Tensor elements. See the Tensor Accessors section below for more details.

.. cpp:function:: template<typename T> TensorReadStreamAccessor<T> read_stream_accessor() const&

    Opens a streaming accessor for read on a tensor. Template parameter ``T`` is the scalar type of the tensor data. See Streaming Accessors section below for more details.

.. cpp:function:: template<typename T> TensorWriteStreamAccessor<T> write_stream_accessor() &

    Opens a streaming accessor for write on a tensor. Template parameter ``T`` is the scalar type of the tensor data. See Streaming Accessors section below for more details.

.. cpp:function:: CoherencyEnforcer::Policy get_accessor_coherence_policy() const

    Get the Tensor accessor coherence policy. See Coherence section below for more details.

.. cpp:function:: void set_accessor_coherence_policy(CoherencyEnforcer::Policy policy) const

    Set the Tensor accessor coherence policy. See Coherence section below for more details.

.. cpp:function:: TensorTcmAccessor<true> tcm_accessor() const&

    Opens a TCM accessor on a tensor. This const-qualified overload returns a read-only ``TensorTcmAccessor``, preventing the user from writing to Tensor elements. See TCM Accessor section below for more details.

.. cpp:function:: TensorTcmAccessor<false> tcm_accessor() &

    Opens a TCM accessor on a tensor. This non-const-qualified overload returns a ``TensorTcmAccessor`` that can be used to both read and write to Tensor elements. See TCM Accessor section below for more details.

.. cpp:function:: torch::Tensor& fill_(const torch::Scalar & value) const
    
    Fill a tensor with the specified value.

Tensor Operators
""""""""""""""""

.. cpp:function:: Tensor& operator=(const Tensor &x) &
.. cpp:function:: Tensor& operator=(Tensor &&x) &

    Assignment operators

Tensor Accessors
----------------

The standard tensor accessor provides element-wise random access to ``Tensor`` elements. They can be created by calling ``Tensor::accessor()``. It can be used similarly to the Pytorch ATen version (see https://pytorch.org/cppdocs/notes/tensor_basics.html#cpu-accessors). However, it is not as fast as other methods of accessing a ``Tensor``, such as the streaming accessor or TCM accessor.

Example Usage
^^^^^^^^^^^^^

Element-wise add of two 1D tensors using ``TensorAccessor``.

.. code-block:: c++

    torch::Tensor tensor_add_compute(const torch::Tensor& t1, const torch::Tensor& t2) {
        size_t num_elem = t1.numel();
        assert(t1.sizes() == t2.sizes());
        torch::Tensor t_out = torch::empty({num_elem}, torch::kFloat);

        auto t1_acc = t1.accessor<float, 1>();
        auto t2_acc = t2.accessor<float, 1>();
        auto t_out_acc = t_out.accessor<float, 1>();
        for (size_t i = 0; i < num_elem; i++) {
            t_out_acc[i] = t1_acc[i] + t2_acc[i];
        }
        return t_out;
    }

.. _custom-ops-ref-guide-mem-arch:

Memory Architecture
^^^^^^^^^^^^^^^^^^^

Tensor data is stored in NeuronCore memory. The various types of accessors enable users to access tensor data from their custom C++ operator code running on the GPSIMD engine.

.. image:: /neuron-customops/images/ncorev2_gpsimd_memory.png
    :width: 600

Streaming Accessors
-------------------

Streaming accessors provide the user the ability to access ``Tensor`` elements in sequential order, faster than the standard tensor accessor. There are two stream accessor classes, one for reading and one for writing. Users should not construct stream accessors directly, but should get them from a ``Tensor`` using ``Tensor::read_stream_accessor`` and ``Tensor::write_stream_accessor()``.

An active stream accessor is defined as a stream accessor that has been instantiated and not yet closed (via the ``close()`` method or by going out-of-scope).

The user is responsible for managing stream accessors concurrently accessing the same ``Tensor``. For safest usage, no stream accessor should be active while there is an active ``TensorWriteStreamAccessor`` on the same ``Tensor``. The user may either have multiple ``TensorReadStreamAccessors`` active on the same ``Tensor``, or only have a single ``TensorWriteStreamAccessor`` active on that ``Tensor``. Stream accessors should not be used concurrently with standard tensor accessors on the same ``Tensor``.

An unlimited number of active stream accessors (in total, across all ``Tensors``) are functionally supported, but only up to 4 active stream accessors will be performant. Additional stream accessors beyond the 4th will have performance similar to that of a standard tensor accessor.

Example Usage
^^^^^^^^^^^^^

Element-wise add of two tensors using ``TensorWriteStreamAccessor`` and ``TensorWriteStreamAccessor``.

.. code-block:: c++

    torch::Tensor tensor_add_compute(const torch::Tensor& t1, const torch::Tensor& t2) {
        assert(t1.sizes() == t2.sizes());
        torch::Tensor t_out = torch::empty(t1.sizes(), torch::kFloat);

        auto t1_rd_stm_acc = t1.read_stream_accessor<float>();
        auto t2_rd_stm_acc = t2.read_stream_accessor<float>();
        auto t_out_wr_stm_acc = t_out.write_stream_accessor<float>();
        for (int i = 0; i < t1.numel(); i++) {
            auto sum = t1_rd_stm_acc.read() + t2_rd_stm_acc.read();
            t_out_wr_stm_acc.write(sum);
        }
        return t_out;
    }

Class torch::TensorWriteStreamAccessor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:class:: template<typename T> class TensorReadStreamAccessor

    The class template parameter ``T`` is the scalar type of the tensor data.

Member Functions
""""""""""""""""

.. cpp:function:: T read()

    Reads from next element in the stream. User is responsible for knowing when to stop reading from ``TensorReadStreamAccessor``. Reading past the end of the stream or on a closed stream results in undefined behaviour.

.. cpp:function:: int close()

    Closes stream. Do not read from the stream after calling ``close()``.

Class torch::TensorWriteStreamAccessor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:class:: template<typename T> class torch::TensorWriteStreamAccessor

    The class template parameter ``T`` is the scalar type of the tensor data.

Member Functions
""""""""""""""""

.. cpp:function:: void write(T value)

    Writes to next element in the stream. Written value is not guaranteed to be written back to the Tensor's memory until the ``TensorWriteStreamAccessor`` goes out of scope, or the user explicitly calls ``close()``. User is responsible for knowing when to stop writing to a stream accessor. Writing past the end of the stream or on a closed stream results in undefined behaviour.

.. cpp:function:: int close()

    Closes stream. Flushes write data to the ``Tensor``'s memory. Do not write to the stream after calling ``close()``.

Coherence
^^^^^^^^^

Stream accessors cache ``Tensor`` data in GPSIMD tightly-coupled memory (TCM), but do not ensure their caches remain coherent. When exactly they read from or write back to NeuronCore memory is opaque to the user (except for ``close()`` which forces a write back).

The safest way to use them is to ensure that no stream accessor is active (instantiated and not yet closed) while there is an active write stream accessor on the same ``Tensor``. The user should either have multiple read stream accessors active on the same ``Tensor``, or only have a single write stream accessor active on that ``Tensor``.

The standard tensor accessors read/write NeuronCore memory directly. Therefore, tensor accessors can safely concurrently access the same ``Tensor``, but it is safest not to use them concurrently with stream accessors since NeuronCore memory isn't guaranteed to be coherent with the stream accessor caches.

These coarse-grained guidelines are best practices, but it is possible to ignore them with careful usage of the accessors (making sure elements are read before they are written to, elements written to are written back before being read again, etc).

The coherence policy of a ``Tensor`` determines what to do when there is potentially incoherent access by an accessor of that ``Tensor``. It can either cause an error, or allow it but print a warning, or do nothing. In the case of the latter two options, it is the user's responsibility to ensure they carefully use accessors coherently. Coherence policy for ``Tensors`` is ``torch::CoherencyEnforcer::Policy::COHERENT`` by default, but can be changed using ``Tensor::set_accessor_coherence_policy()``.

.. code-block:: c++

    // class torch::CoherencyEnforcer
    enum Policy {
        // Enforce a resource is acquired in a way that guarantees coherence
        // Causes an error if it encounters potentially incoherent access
        COHERENT,

        // Allows potentially incoherent access, but will print a warning
        INCOHERENT_VERBOSE,

        // Allows potentially incoherent access, no error or warnings
        INCOHERENT_QUIET
    };

TCM Accessor
------------

TCM accessors provide the fastest read and write performance. TCM accessors allow the user to manually manage copying data between larger, but slower-access NeuronCore memory to faster GPSIMD tightly-coupled memory (TCM). It may be beneficial to see the diagram under :ref:`custom-ops-ref-guide-mem-arch`. Create a ``TensorTcmAccessor`` from a ``Tensor`` by calling ``Tensor::tcm_accessor()``. Users can allocate and free TCM memory using ``tcm_malloc()`` and ``tcm_free()``. Users have access to a 16KB pool of TCM memory. Note the streaming accessors also allocate from this pool (4KB each). TCM accessors do not do any coherence checks.

Example Usage
^^^^^^^^^^^^^

Element-wise negate of a tensor using ``TensorTcmAccessor``.

.. code-block:: c++

    torch::Tensor tensor_negate_compute(const torch::Tensor& t_in) {
        size_t num_elem = t_in.numel();
        torch::Tensor t_out = torch::empty(t_in.sizes(), torch::kFloat);

        static constexpr size_t buffer_size = 1024;
        float *tcm_buffer = (float *)torch::neuron::tcm_malloc(sizeof(float) * buffer_size);

        if (tcm_buffer != nullptr) {
            // tcm_malloc allocated successfully, use TensorTcmAccessor
            auto t_in_tcm_acc = t_in.tcm_accessor();
            auto t_out_tcm_acc = t_out.tcm_accessor();
            for (size_t i = 0; i < num_elem; i += buffer_size) {
                size_t remaining_elem = num_elem - i;
                size_t copy_size = (remaining_elem > buffer_size) ? buffer_size : remaining_elem;

                t_in_tcm_acc.tensor_to_tcm<float>(tcm_buffer, i, copy_size);
                for (size_t j = 0; j < copy_size; j++) {
                    tcm_buffer[j] *= -1;
                }
                t_out_tcm_acc.tcm_to_tensor<float>(tcm_buffer, i, copy_size);
            }

            torch::neuron::tcm_free(tcm_buffer);
        } else {
            // Handle not enough memory...
        }

        return t_out;
    }

TCM Management Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: void * torch::neuron::tcm_malloc(size_t nbytes)

    Allocate ``nbytes`` bytes of memory from TCM and return pointer to this memory. Upon failure, returns null.

.. cpp:function:: void torch::neuron::tcm_free(void * ptr)

    Free memory that was allocated by ``tcm_malloc()``. Undefined behaviour if ``ptr`` was not returned from a previous call to ``tcm_malloc()``.

Class torch::TensorTcmAccessor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:class:: template<bool read_only> class torch::TensorTcmAccessor

    The ``read_only`` template parameter controls whether or not you can write to the accessor's ``Tensor``. A ``const Tensor`` will return a read-only ``TensorTcmAccessor`` from ``Tensor::tcm_accessor()``.

Member Functions
""""""""""""""""

.. cpp:function:: template<typename T> void tensor_to_tcm(T * tcm_ptr, size_t tensor_offset, size_t num_elem)

    Copy ``num_elem`` elements from the accessor's ``Tensor`` starting at the index ``tensor_offset`` to a TCM buffer starting at ``tcm_ptr``. Tensor indexing is performed as if the tensor was flattened. Template parameter ``T`` is the scalar type of the tensor data. The TCM buffer's size should be at least ``sizeof(T) * num_elem`` bytes.

.. cpp:function:: template<typename T> void tcm_to_tensor(T * tcm_ptr, size_t tensor_offset, size_t num_elem)

    Copy ``num_elem`` elements from a TCM buffer starting at ``tcm_ptr`` to the accessor's ``Tensor`` starting at the index ``tensor_offset``. Tensor indexing is performed as if the tensor was flattened. The TCM buffer's size should be at least ``sizeof(T) * num_elem`` bytes.


printf()
--------------

Custom C++ operators support the use of C++'s ``printf()`` to send information to the host's terminal. Using ``printf()`` is the recommended approach to functional debug. With it, the programmer can check the value of inputs, outputs, intermediate values, and control flow within their operator.

Usage
^^^^^

To use ``printf()`` within a Custom C++ operator, the programmer must set the following environment variables before running their model in order to receive the messages printed by their operator:

.. list-table:: Environment Variables
   :widths: 50 200 20 200 200
   :header-rows: 1



   * - Name
     - Description
     - Type
     - Value to Enable printf
     - Default Value
   * - ``NEURON_RT_LOG_LEVEL``
     - Runtime log verbose level
     - String
     - At least ``INFO``
     - See (https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-configurable-parameters.html?highlight=NEURON_RT_LOG_LEVEL#neuron-runtime-configuration) for more options.
   * - ``NEURON_RT_GPSIMD_STDOUT_QUEUE_SIZE_BYTES``
     - Size of the printf output buffer, in bytes
     - Integer
     - Any power of two that is equal to or less than ``2097152`` (2MB)
     - Recommend setting a value of ``2097152`` to maximize the size of printf's buffer. Setting a value of 0 disables printf.

Within a Custom C++ operator, ``printf()`` can be used as normal from within a C++ program. For more information, consult a reference such as (https://cplusplus.com/reference/cstdio/printf/)

Example
^^^^^^^

.. code-block:: c++

    #include <torch/torch.h>
    #include <stdio.h> // Contains printf()

    torch::Tensor tensor_negate_compute(const torch::Tensor& t_in) {
        size_t num_elem = t_in.numel();
        torch::Tensor t_out = torch::zeros({num_elem}, torch::kFloat);

        auto t_in_acc = t_in.accessor<float, 1>();
        auto t_out_acc = t_out.accessor<float, 1>();
        for (size_t i = 0; i < num_elem; i++) {
            float tmp = -1 * t_in_acc[i];
            printf("Assigning element %d to a value of %f\n", i, tmp);
            t_out_acc[i] = tmp;
        }
        return t_out;
    }

Print statements then appear on the host's terminal with a header message prepended:

::

    2023-Jan-26 00:25:02.0183  4057:4131   INFO  TDRV:pool_stdio_queue_consume_all_entries    Printing stdout from GPSIMD:
    Setting element 0 to value -1.000000
    Setting element 1 to value -2.000000
    Setting element 2 to value -3.000000
    Setting element 3 to value -4.000000
    Setting element 4 to value -5.000000
    Setting element 5 to value -6.000000
    Setting element 6 to value -7.000000
    Setting element 7 to value -8.000000


Limitations
^^^^^^^^^^^

* Performance: using ``printf()`` significantly degrades the operator's performance
    * The programmer can disable it by unsetting ``NEURON_RT_GPSIMD_STDOUT_QUEUE_SIZE_BYTES`` or setting it to 0
        * Disabling ``printf()`` is recommended if running the model in a performance-sensitive context
    * To maximize performance, the programmer should remove calls to ``printf()`` from within the operator
        * Even if disabled, calling the function incurs overhead
* Buffer size: output from ``printf()`` is buffered during model execution and read by the Neuron runtime after execution
    * The model can still execute successfully if the programmer overflows the buffer
    * Overflowing the buffer will cause the oldest data in it to be overwritten
* Print statements are processed and printed to the host's terminal at the end of model execution, not in real time
