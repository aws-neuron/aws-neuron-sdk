import numpy as np
import ml_dtypes

class FrameworkKernel: 
  r"""
  NKI kernels are represeted as XLA CustomCall instructions in HLO. This class
  facilitates the HLO generation for NKI kernels.

  For example, a kernel that read from the first two tensors, and write to its last
  argument in python,

  .. code-block:: python

    def example_kernel(in1, in2, out):
        # Actual kernel content omitted
        pass

  should be mapped to the following HLO instruction,

  .. code-block::

    %custom-call.2 = f32[16,8,128,512]{3,2,1,0} custom-call(
    f32[16,8,128,512]{3,2,1,0} %p2.2, f32[16,8,128,512]{3,2,1,0} %p1.2),
    custom_call_target="AwsNeuronCustomNativeKernel",
    api_version=API_VERSION_UNSPECIFIED,
    metadata={op_type="xla___op_NkiKernelCallImpl" op_name="xla___op_NkiKernelCallImpl"},
    backend_config= # ...omitted

  It is important to notice that, although in Python, NKI kernels use pass-by-reference
  semantics, the corresponding HLO instruction returns the output tensor.

  The field `api_version` is optional. The field `metadata` is optional debug information,
  developer could elect to pass `op_type` and `op_name`, the information will show up in
  the profile using `neuron-profiler`. The `custom_call_target` should always be
  "AwsNeuronCustomNativeKernel".

  Framework developers should inherit this class and implement the following methods.

  #. translate_to_neuron_dtype
  #. is_framework_tensor
  #. map_framework_tensor

  Then `backend_config` can be obtained by calling `dump_config(*args, **kwargs)`.

  As an example, suppose we have correctly implemented a PyTorch variant of this class, i.e.
  `PyTorchFrameWorkKernel(FrameworkKernel)`, then we can generate the `backend_config` for
  the HLO instruction example with the following.

  .. code-block:: python

    in1 = torch.rand((16, 8, 128, 512), dtype=torch.float32)
    in2 = torch.rand((16, 8, 128, 512), dtype=torch.float32)
    out = torch.rand((16, 8, 128, 512), dtype=torch.float32)
    kernel = PyTorchFrameworkKernel(func_name=example_kernel.__name__, func=example_kernel, grid=(16, 8))
    kernel.dump_config(in1, in2, out) # Dump config based on inputs
    # Omitted, config string specialized for (16, 8, 12, 512)
    in3 = torch.rand((16, 8, 64, 1024), dtype=torch.float32)
    in4 = torch.rand((16, 8, 64, 1024), dtype=torch.float32)
    out = torch.rand((16, 8, 64, 1024), dtype=torch.float32)
    kernel = PyTorchFrameworkKernel(func_name=example_kernel.__name__, func=example_kernel, grid=(16, 8))
    kernel.dump_config(in3, in4, out=out) # Dump config based on inputs
    # Omitted, config string specialized for (16, 8, 64, 1024)

  The kernel should be called for each set of different input tensor shapes configuration.
  """

  def dump_config(self, *args, **kwargs):
    r"""
    Returns the `backend_config`, the list of input names and the list of the output name,
    based on given arguments.

    If `self.enable_cache` is True, `dump_config` will try to retrieve the results
    from the cache using `args`, `kwargs` and the spmd launch grid and other
    kernel attributes as key to identify the identical backend_config.

    Otherwise, `dump_config` will always generate new backend_config.

    # NOTE: THis is still used by legacy framework code, dont change the signature
    """
    ...

  def is_framework_tensor(self, t):
    r"""
    Return true if and only if `t` should be treated as a tensor. Parameter that
    returns false must be constants known at compile time.

    As an example, for PyTorch,

    .. code-block:: python

      >>> is_framework_tensor(torch.rand((2, 3)))
      True
      >>> is_framework_tensor("this is not a tensor")
      False
    """
    ...

  def map_framework_tensor(self, t):
    r"""
    Take in a framework tensor, returns the shape of tensor and its type in a tuple. This function
    should only be called on t where `is_framework_tensor(t)` returns True.

    As an example, for PyTorch,

    .. code-block:: python

      >>> map_framework_tensor(torch.rand((2, 3), dtype=torch.bfloat16))
      (torch.Size([2, 3]), torch.bfloat16)
    """
    ...

  def translate_to_neuron_dtype(self, _dtype):
    r"""
    Translate a framework dtype to neuron specific dtype representation in numpy
     or neuron specific dtype.

    As an example, for PyTorch,

    .. code-block:: python

      >>> result = translate_to_neuron_dtype(torch.bfloat16)
      >>> result == neuronxcc.nki.language.bfloat16
      True
    """
    ...

class tensor: 
  r"""
  A tensor object represents a multidimensional, homogeneous array of fixed-size items
  """

  def assert_shape(self, shape):
    r"""
    Assert that the tensor has the given shape.

    :param shape: The expected shape.
    :return: The tensor.
    """
    ...

  def astype(self, dtype):
    r"""
    Copy of the tensor, cast to a specified type.

    :param dtype: The target dtype
    :return: the tensor with new type. Copy ALWAYS occur
    """
    ...

  def broadcast_to(self, shape):
    r"""
    Broadcast tensor to a new shape based on numpy broadcast rules.
    The tensor object must be a tile or can be implicitly converted to a tile.
    A tensor can be implicitly converted to a tile iff the partition dimension
    is the highest dimension.

    :param shape: The new shape
    :return:      Return a new view of the tensor, no copy will occur
    """
    ...

  @property
  def dtype(self):
    r"""
    Data type of the tensor.
    """
    ...

  def expand_dims(self, axis):
    r"""
    Gives a new shape to a tensor by adding a dimension of size 1 at the specified position.

    :param axis: the position of the new dimension.
    :return:      Return a new tensor with expanded shape
    """
    ...

  @property
  def itemsize(self):
    r"""
    Length of one tensor element in bytes.
    """
    ...

  @property
  def ndim(self):
    r"""
    Number of dimensions of the tensor.
    """
    ...

  def reshape(self, shape):
    r"""
    Gives a new shape to an array without changing its data.

    :param shape: The new shape
    :return:      Return a new view of the tensor, no copy will occur
    """
    ...

  @property
  def shape(self):
    r"""
    Shape of the tensor.
    """
    ...

  def view(self, dtype):
    r"""
    Return a new view of the tensor, reinterpret to a specified type.

    :return: A new tensor object refer to the original tensor data, NO copy will occur
    """
    ...

def baremetal(kernel=None, **kwargs):
  r"""
  Compile and run a NKI kernel on NeuronDevice without involving ML frameworks such as PyTorch and JAX.
  If you decorate your NKI kernel function with decorator ``@nki.baremetal(...)``, you may call the NKI kernel function
  directly just like any other Python function. You must run this API on a Trn/Inf instance with NeuronDevices
  (v2 or beyond) attached.

  .. note::

    The decorated function using ``nki.baremetal`` expects
    `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ as input/output
    tensors instead of ML framework tensor objects.

  This decorator compiles the NKI kernel into an executable on NeuronDevices (``NEFF``) and also
  collects an execution trace (``NTFF``) by running the ``NEFF`` on the local NeuronDevice. See
  :doc:`Profiling NKI kernels with Neuron Profile <../../neuron_profile_for_nki>` for more information on how to
  visualize the execution trace for profiling purposes.

  Since ``nki.baremetal`` runs the compiled NEFF without invoking any ML framework,
  it is the fastest way to compile and run any NKI kernel
  standalone on NeuronDevice. Therefore, this decorator is useful for quickly iterating an early implementation of
  a NKI kernel to reach functional correctness before porting it to the ML framework and injecting the kernel
  into the full ML model. To iterate over NKI kernel performance quickly, NKI also provides
  :doc:`nki.benchmark <../generated/nki.benchmark>`
  decorator which uses the same underlying mechanism as ``nki.baremetal`` but additionally collects latency statistics
  in different percentiles.

  :param save_neff_name: A file path to save your NEFF file. By default, this is unspecified, and the NEFF file
                         will be deleted automatically after execution.
  :param save_trace_name: A file path to save your NTFF file. By default, this is unspecified, and the NTFF file
                         will be deleted automatically after execution.
                         Known issue: if ``save_trace_name`` is specified, ``save_neff_name`` must be set to "file.neff".
  :param additional_compile_opt: Additional Neuron compiler flags to pass in
                                 when compiling the kernel.
  :param artifacts_dir: A directory path to save Neuron compiler artifacts. The directory must be empty before running
         the kernel. A non-empty directory would lead to a compilation error.
  :return: None

  .. code-block:: python
    :caption: An Example

    from neuronxcc.nki import baremetal
    import neuronxcc.nki.language as nl
    import numpy as np

    @baremetal(save_neff_name='file.neff', save_trace_name='profile.ntff')
    def nki_tensor_tensor_add(a_tensor, b_tensor):
      c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

      a = nl.load(a_tensor)
      b = nl.load(b_tensor)

      c = a + b

      nl.store(c_tensor, c)

      return c_tensor

    a = np.zeros([128, 1024], dtype=np.float32)
    b = np.random.random_sample([128, 1024]).astype(np.float32)
    c = nki_tensor_tensor_add(a, b)

    assert np.allclose(c, a + b)
  """
  ...

def benchmark(kernel=None, **kwargs):
  r"""
  Benchmark a NKI kernel on a NeuronDevice by using ``nki.benchmark`` as a decorator. You must run this API on a
  Trn/Inf instance with NeuronDevices (v2 or beyond) attached and also ``aws-neuronx-tools`` installed on the host using
  the following steps:

  .. code-block:: bash

    # on Ubuntu
    sudo apt-get install aws-neuronx-tools=2.* -y

    # on Amazon Linux
    sudo dnf install aws-neuronx-tools-2.* -y

  You may specify a path to save your NEFF file through input
  parameter ``save_neff_name`` and a path to save your NTFF file through ``save_trace_name``.
  See :doc:`Profiling NKI kernels with Neuron Profile <../../neuron_profile_for_nki>` for more information on how to
  visualize the execution trace for profiling purposes.

  .. note::

    Similar to ``nki.baremetal``, The decorated function using ``nki.benchmark`` expects
    `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ as input/output
    tensors instead of ML framework tensor objects.
  
  In additional to generating NEFF/NTFF files, this decorator also invokes ``neuron-bench`` to collect
  execution latency statistics of the NEFF file and prints the statistics to the console.

  ``neuron-bench`` is a tool that launches the NEFF file on a NeuronDevice in a loop to collect
  end-to-end latency statistics. You may specify the number of warm-up iterations to skip benchmarking in input
  parameter ``warmup``, and the number of benchmarking iterations in ``iters``. Currently, ``nki.benchmark`` only
  supports benchmarking on a single NeuronCore, since NKI not yet supports collective compute. Note, ``neuron-bench``
  measures not only the device latency but also the time taken to transfer data between host and device. However, the tool
  does not rely on any ML framework to launch the NEFF and therefore reports NEFF latency without any framework overhead.

  :param warmup: The number of iterations for warmup execution (10 by default).
  :param iters: The number of iterations for benchmarking (100 by default).
  :param save_neff_name: Save the compiled neff file if specify a name
                         (unspecified by default).
  :param save_trace_name: Save the trace (profile) file if specified a name
                          (unspecified by default); at the moment, it requires
                          that the `save_neff_name` is unspecified or specified
                          as 'file.neff'.
  :param additional_compile_opt: Additional Neuron compiler flags to pass in
                                 when compiling the kernel.
  :return: A function object that wraps the decorating function. A property ``benchmark_result.nc_latency`` is
           available after invocation.
           ``get_latency_percentile(int)`` of the property returns the specified percentile latency in microsecond(us).
           Available percentiles: [0, 1, 10, 25, 50, 90, 99, 100]

  .. code-block:: python
    :caption: An Example

    from neuronxcc.nki import benchmark
    import neuronxcc.nki.language as nl
    import numpy as np

    @benchmark(warmup=10, iters = 100, save_neff_name='file.neff', save_trace_name='profile.ntff')
    def nki_tensor_tensor_add(a_tensor, b_tensor):
      c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

      a = nl.load(a_tensor)
      b = nl.load(b_tensor)

      c = a + b

      nl.store(c_tensor, c)

      return c_tensor

    a = np.zeros([128, 1024], dtype=np.float32)
    b = np.random.random_sample([128, 1024]).astype(np.float32)
    c = nki_tensor_tensor_add(a, b)

    metrics = nki_tensor_tensor_add.benchmark_result.nc_latency
    print("latency.p50 = " + str(metrics.get_latency_percentile(50)))
    print("latency.p99 = " + str(metrics.get_latency_percentile(99)))

  .. note::

    ``nki.benchmark`` does not use the actual inputs passed into the benchmarked function when running the 
    neff file. For instance, in the above example, the output c tensor is undefined and should not be used 
    for numerical accuracy checks.
  """
  ...

def jit(func=None, mode="auto", **kwargs):
  r"""
  This decorator compiles a function to run on NeuronDevices.

  This decorator tries to automatically detect the current framework and compile
  the function as a custom operator of the current framework. To bypass the
  framework detection logic, you may specify the ``mode`` parameter explicitly.

  :param func:               The function that define the custom op
  :param mode:               The compilation mode, possible values: "jax", "torchxla",
                             "baremetal", "benchmark", "simulation" and "auto"

  .. code-block:: python
    :caption: An Example

    from neuronxcc import nki
    import neuronxcc.nki.language as nl

    @nki.jit
    def nki_tensor_tensor_add(a_tensor, b_tensor):
      c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

      a = nl.load(a_tensor)
      b = nl.load(b_tensor)

      c = a + b

      nl.store(c_tensor, c)

      return c_tensor

  """
  ...

def profile(func=None, **kwargs):
  r"""
  Profile a NKI kernel on a NeuronDevice by using ``nki.profile`` as a decorator. 

  .. note::

    Similar to ``nki.baremetal``, The decorated function using ``nki.benchmark`` expects
    `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ as input/output
    tensors instead of ML framework tensor objects.

  :param working_directory: A path to working directory where profile artifacts are saved,
                            This must be specified and must also be an absolute path.
  :param save_neff_name: Name of the saved neff file if specified
                         (file.neff by default).
  :param save_trace_name: Name of the saved trace (profile) file if specified
                          (profile.ntff by default)
  :param additional_compile_opt: Additional Neuron compiler flags to pass in
                                 when compiling the kernel.
  :param overwrite: Overwrite existing profile artifacts if set to True.
                    Default is False.
  :param profile_nth: Profiles the `profile_nth` execution.
                      Default is 1.
  :return: None

  .. code-block:: python
    :caption: An Example

    from neuronxcc import nki
    import neuronxcc.nki.language as nl

    @nki.profile(working_directory="/home/ubuntu/profiles", save_neff_name='file.neff', save_trace_name='profile.ntff')
    def nki_tensor_tensor_add(a_tensor, b_tensor):
      c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

      a = nl.load(a_tensor)
      b = nl.load(b_tensor)

      c = a + b

      nl.store(c_tensor, c)

      return c_tensor

  ``nki.profile`` will save file.neff, profile.ntff, along with json files containing a profile summary
  inside of the working_directory.

  See `Profiling NKI kernels with Neuron Profile <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/neuron_profile_for_nki.html#neuron-profile-for-nki>`_ 
  for more information on how to visualize the execution trace for profiling purposes.
  
  In addition, more information about `neuron-profile` can be found in its 
  `documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html>`_.

  .. note::
	  
	     ``nki.profile`` does not use the actual inputs passed into the profiled function when running the 
	     neff file. For instance, in the above example, the output c tensor is undefined and should not be used 
	     for numerical accuracy checks. The input tensors are used mainly to specify the shape of inputs.

  """
  ...

def simulate_kernel(kernel, *args, **kwargs):
  r"""
  Simulate a nki kernel on CPU using a built-in simulator in Neuron Compiler.
  This simulation mode is especially useful for inspecting intermediate tensor
  values using :doc:`nki.language.device_print <nki.language.device_print>`
  (see code example below).

  .. note::

    All input and output tensors to the kernel must be
    `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ when
    using this ``simulate_kernel`` API.

  To run the kernel on a NeuronCore instead, please refer to
  :doc:`Getting Started with NKI <../../getting_started>`.

  :param kernel: The kernel to be simulated
  :param args:   The args of the kernel
  :param kwargs: The kwargs of the kernel
  :return:

  Examples:

  .. nki_example:: ../../test/test_nki_simulate_kernel.py
   :language: python
  """
  ...

