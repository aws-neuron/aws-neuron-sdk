.. _pytorch-neuronx-install-cxx11:

Install with support for C++11 ABI
==================================

.. warning::

    The intended user of this guide is using a custom built version of
    ``torch`` and ``torch-xla`` or compiling a non-python application which must be built using
    the C++11 ABI.

    *Most applications do not require this specialized distribution.*

    For regular installation instructions see: :ref:`Fresh install <pytorch-neuronx-install>`

The standard ``torch-neuronx`` packages (which are normally installed according
to the :ref:`Fresh install <pytorch-neuronx-install>` guide) are compiled with
the pre-C++11 ABI and linked against the pre-C++11 ``libtorch``. These
compilation options ensure that the ``torch-neuronx`` ABI matches the *publicly*
released version of the ``torch`` and ``torch-xla`` packages that are installed from the default
PyPI index.

To support applications with specific ABI requirements, Neuron distributes
packages which are linked against the C++11 version of
``libtorch``. These ``torch-neuronx`` packages are built using the
``-D_GLIBCXX_USE_CXX11_ABI=1`` compilation flag. 

.. note::

    The ``libneuronxla`` packages are already built with both pre-C++11 ABI and C++11 ABI symbols so the same PIP package can be used for C++11 ABI applications.

The only difference between these packages and the standard packages
is the torch plugin library contained within the package. This is the
``libtorchneuron.so`` library located in the ``torch_neuronx/lib/`` package
directory. All other libraries and python files within the packages are
identical. This means that these C++11-compatible packages are drop-in
replacements in environments that are incompatible with the standard releases of
``torch-neuronx``. The behavior is identical whether compiling models, executing
inferences or running training.

Installation
^^^^^^^^^^^^

All versions of the library are available to download from the following pip
index:

::

    https://pip.repos.neuron.amazonaws.com/cxx11


To install a wheel, it is recommended to use the ``--no-deps`` flag since
versions of ``torch`` and ``torch-xla`` compiled using the C++11 ABI are not distributed on this
index.

::

    pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com/cxx11 torch-neuronx --no-deps


Specific versions of ``torch-neuronx`` with C++11 ABI support can be installed
just like standard versions of ``torch-neuronx``.

::

    pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com/cxx11 "torch-neuronx==2.1.*" --no-deps
    pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com/cxx11 "torch-neuronx==2.5,*" --no-deps

.. important::

    This pip index does not include a distribution of ``torch`` and ``torch-xla`` compiled with
    the new C++11 ABI. The intent of this index is *only* to provide Neuron SDK
    wheels. See :ref:`pytorch-neuronx-cxx11-building-torch-xla`.

    The version of ``torch`` and ``torch-xla`` that are distributed on the default PyPI index is
    compiled with the old pre-C++11 ABI.

    If a C++11 ``torch-neuronx`` package is installed *with* dependencies
    using the *default* PyPI index, then the installed version of ``torch`` and ``torch-xla`` will
    be using the pre-C++11 ABI and ``torch-neuronx`` will be using the C++11
    ABI. This ABI mismatch will lead to ``undefined symbol`` errors in both Python usage and at link
    time for non-Python applications.


.. _pytorch-neuronx-cxx11-building-torch-xla:

Building ``torch`` and ``torch-xla`` with C++11 ABI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The instructions for building ``torch`` from source is at https://github.com/pytorch/pytorch#from-source

The instructions for building ``torch-xla`` from source is at https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md

The following are simplified instructions (subject to change):

Setting the build environment:

.. code:: bash

   sudo apt install cmake
   pip install yapf==0.30.0
   wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
   sudo cp bazelisk-linux-amd64 /usr/local/bin/bazel

Build ``torch`` (CPU only) and ``torch-xla`` wheels for version 2.5:

.. code:: bash

   git clone --recursive https://github.com/pytorch/pytorch --branch v2.5.1
   cd pytorch/
   git clone --recursive https://github.com/pytorch/xla.git --branch r2.5.1
   _GLIBCXX_USE_CXX11_ABI=1 python setup.py bdist_wheel
   # pip wheel will be present in ./dist
   cd xla/
   CXX_ABI=1 python setup.py bdist_wheel
   # pip wheel will be present in ./dist

Build ``torch`` (CPU only) and ``torch-xla`` wheels for version 2.1:

.. code:: bash

   git clone --recursive https://github.com/pytorch/pytorch --branch v2.1.2
   cd pytorch/
   git clone --recursive https://github.com/pytorch/xla.git --branch r2.1_aws_neuron
   _GLIBCXX_USE_CXX11_ABI=1 python setup.py bdist_wheel
   # pip wheel will be present in ./dist
   cd xla/
   # Release 2.21 TORCH_XLA_VERSION=2.1.6
   TORCH_XLA_VERSION=2.1.6 CXX_ABI=1 python setup.py bdist_wheel
   # pip wheel will be present in ./dist


FAQ
^^^

When should I use a C++11 torch-neuronx wheel?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributions compiled with the new C++11 ABI should only be used in the
following cases:

1. You have built your own version of ``torch`` and ``torch-xla`` which uses the new C++11 ABI and
   need a corresponding version of ``torch-neuronx`` that is compatible.
2. You are compiling an application against a ``libtorch``
   which uses the C++11 ABI and would like to include
   ``libtorchneuron.so`` as well. Torch distributes these C++11 ``libtorch``
   libraries with a ``libtorch-cxx11`` prefix.

    Example:

    ::

        https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip


Can I download a library/header zip file similar to the torch distribution?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently ``torch-neuron`` does not distribute a bundled library ``.zip`` with
only library/header files.

The recommended alternative when compiling ``libtorchneuron.so`` into a
non-python application is to install the ``torch-neuron`` wheel using ``pip``
according to the installation instructions. Then use the ``libtorchneuron.so``
library from within the python ``site-packages`` directory.

A second alternative to isolate the package contents from a python environment
is to download the wheel and unpack the contents:

.. code:: bash

    pip download --extra-index-url=https://pip.repos.neuron.amazonaws.com/cxx11 torch-neuronx --no-deps
    wheel unpack torch_neuronx-*.whl

If the exact version of the ``torch-neuronx`` package is known and no
Python/Pip is available in the build environment, an alternative is to fetch the
package file directly and ``unzip`` the wheel:

.. code::

    wget https://pip.repos.neuron.amazonaws.com/cxx11/torch-neuronx/torch_neuronx-<VERSION>-py3-none-any.whl
    unzip torch_neuronx-<VERSION>-py3-none-any.whl


.. _pytorch-neuronx-cxx11-versioning:

How can I know which ABI torch-neuron is using?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Packages which use the pre-C++11 ABI have no local identifier and use the
following version scheme:

::

    <torch version>.<neuron version>

Packages which use the C++11 ABI have a ``+cxx11`` local identifier and use
following version scheme:

::

    <torch version>.<neuron version>+cxx11


This allows the ABI to be validated in the by inspecting the local identifier
(or version suffix).

Example:
::

    2.1.5.2.4.0+cxx11
    2.5.1.2.4.0+cxx11


How can I know which ABI torch is using?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``torch`` python package provides an API at the that allows you to check if
the underlying ``libtorch`` was compiled with the C++11 ABI:

.. code:: python

    import torch
    torch.compiled_with_cxx11_abi()  # True/False

Currently ``torch-neuronx`` does not have an equivalent API. If the C++11 ABI was
used, it will be visible in the version string (See :ref:`pytorch-neuronx-cxx11-versioning`).


Troubleshooting
^^^^^^^^^^^^^^^

What Python errors could I see if I mix ABI versions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a version of ``torch`` compiled with the C++11 ABI will trigger an error
in the python interpreter when importing a version of ``torch-neuronx`` using
the old (pre-C++11) ABI from the standard index. This will manifest as an
error when the ``import torch_neuronx`` statement is executed.

::

    Traceback (most recent call last):
      File "/python3.9/site-packages/torch_neuron/__init__.py", line 64, in <module>
        _register_extension()
      File "/python3.9/site-packages/torch_neuron/__init__.py", line 60, in _register_extension
        torch.ops.load_library(neuron_op_filename)
      File "/python3.9/site-packages/torch/_ops.py", line 110, in load_library
        ctypes.CDLL(path)
      File "/python3.9/ctypes/__init__.py", line 364, in __init__
        self._handle = _dlopen(self._name, mode)
    OSError: /python3.9/site-packages/torch_neuron/lib/libtorchneuron.so: undefined symbol: _ZN5torch6detail10class_baseC2ERKSsS3_SsRKSt9type_infoS6_


Similarly, when using the standard pre-C++11 versions of ``torch/torch-xla`` with the C++11
version of ``torch-neuronx``, an error would also occur at import.

::

    Traceback (most recent call last):
      File "/python3.9/site-packages/torch_neuron/__init__.py", line 79, in <module>
        _register_extension()
      File "/python3.9/site-packages/torch_neuron/__init__.py", line 75, in _register_extension
        torch.ops.load_library(neuron_op_filename)
      File "/python3.9/site-packages/torch/_ops.py", line 110, in load_library
        ctypes.CDLL(path)
      File "/python3.9/ctypes/__init__.py", line 364, in __init__
        self._handle = _dlopen(self._name, mode)
    OSError: /python3.9/site-packages/torch_neuron/lib/libtorchneuron.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE


In either of these cases, the remedy is to ensure that the ABI of the ``torch`` and ``torch-xla``
distribution matches the ABI of the ``torch-neuronx`` distribution.

What compiler/linking errors could I see if I mix ABI versions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you link an application which uses the old (pre-C++11) ABI
``libtorchneuron.so`` with a C++11 version of ``torch``, this will trigger a
link error.

::

    libtorchneuron.so: undefined reference to `torch::detail::class_base::class_base(std::string const&, std::string const&, std::string, std::type_info const&, std::type_info const&)'
    libtorchneuron.so: undefined reference to `c10::Error::Error(c10::SourceLocation, std::string)'
    libtorchneuron.so: undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::string const&)'
    libtorchneuron.so: undefined reference to `c10::ClassType::getMethod(std::string const&) const'
    libtorchneuron.so: undefined reference to `c10::ivalue::ConstantString::create(std::string)'
    libtorchneuron.so: undefined reference to `c10::DeviceTypeName(c10::DeviceType, bool)'
    libtorchneuron.so: undefined reference to `torch::jit::parseSchema(std::string const&)'
    libtorchneuron.so: undefined reference to `unsigned short caffe2::TypeMeta::_typeMetaData<std::string>()'
    libtorchneuron.so: undefined reference to `c10::Warning::warn(c10::SourceLocation const&, std::string const&, bool)'
    libtorchneuron.so: undefined reference to `torch::jit::parseSchemaOrName(std::string const&)'
    libtorchneuron.so: undefined reference to `c10::Symbol::fromQualString(std::string const&)'
    libtorchneuron.so: undefined reference to `c10::Error::Error(std::string, std::string, void const*)'
    libtorchneuron.so: undefined reference to `c10::detail::infer_schema::make_function_schema(std::string&&, std::string&&, c10::ArrayRef<c10::detail::infer_schema::ArgumentDef>, c10::ArrayRef<c10::detail::infer_schema::ArgumentDef>)'
    libtorchneuron.so: undefined reference to `c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&)'
    libtorchneuron.so: undefined reference to `torch::jit::canonicalSchemaString(c10::FunctionSchema const&)'


Similarly, an error will also occur in the opposite scenario where the
C++11 ``libtorchneuron.so`` library is used with the pre-C++11 ``libtorch``:

::

    libtorchneuron.so: undefined reference to `c10::ivalue::ConstantString::create(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)'
    libtorchneuron.so: undefined reference to `torch::jit::parseSchemaOrName(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
    libtorchneuron.so: undefined reference to `c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)'
    libtorchneuron.so: undefined reference to `c10::Error::Error(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, void const*)'
    libtorchneuron.so: undefined reference to `torch::jit::canonicalSchemaString[abi:cxx11](c10::FunctionSchema const&)'
    libtorchneuron.so: undefined reference to `torch::detail::class_base::class_base(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::type_info const&, std::type_info const&)'
    libtorchneuron.so: undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
    libtorchneuron.so: undefined reference to `c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
    libtorchneuron.so: undefined reference to `c10::detail::infer_schema::make_function_schema(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&&, c10::ArrayRef<c10::detail::infer_schema::ArgumentDef>, c10::ArrayRef<c10::detail::infer_schema::ArgumentDef>)'
    libtorchneuron.so: undefined reference to `torch::jit::parseSchema(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
    libtorchneuron.so: undefined reference to `c10::DeviceTypeName[abi:cxx11](c10::DeviceType, bool)'
    libtorchneuron.so: undefined reference to `c10::Symbol::fromQualString(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
    libtorchneuron.so: undefined reference to `unsigned short caffe2::TypeMeta::_typeMetaData<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >()'
    libtorchneuron.so: undefined reference to `c10::ClassType::getMethod(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) const'
    libtorchneuron.so: undefined reference to `c10::Warning::warn(c10::SourceLocation const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, bool)'


In either of these cases, the remedy is to ensure that the ABI of the
``libtorch`` distribution matches the ABI of the ``libtorchneuron.so``
distribution.

The ``torch`` and ``torch-xla`` ABI must match the ``torch-neuron`` ABI or an ``undefined symbol`` error will occur.
