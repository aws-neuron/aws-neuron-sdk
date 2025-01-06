.. _pytorch-install-cxx11:

Install with support for cxx11 ABI
==================================

.. warning::

    The intended user of this guide is using a custom built version of
    ``torch`` or compiling a non-python application which must be built using
    the cxx11 ABI.

    *Most applications do not require this specialized distribution.*

    For regular installation instructions see: :ref:`Fresh install <install-neuron-pytorch>`

The standard ``torch-neuron`` packages (which are normally installed according
to the :ref:`Fresh install <install-neuron-pytorch>` guide) are compiled with
the pre-cxx11 ABI and linked against the pre-cxx11 ``libtorch``. These
compilation options ensure that the ``torch-neuron`` ABI matches the *publicly*
released version of the ``torch`` package that is installed from the default
PyPI index.

To support applications with specific ABI requirements, Neuron distributes
packages which are linked against the cxx11 version of
``libtorch``. These ``torch-neuron`` packages are built using the
``-D_GLIBCXX_USE_CXX11_ABI=1`` compilation flag.

The only difference between these packages and the standard packages
is the torch plugin library contained within the package. This is the
``libtorchneuron.so`` library located in the ``torch_neuron/lib/`` package
directory. All other libraries and python files within the packages are
identical. This means that these cxx11-compatible packages are drop-in
replacements in environments that are incompatible with the standard releases of
``torch-neuron``. Behavior is identical whether compiling models or executing
inferences.

Installation
^^^^^^^^^^^^

All versions of the library are available to download from the following pip
index:

::

    https://pip.repos.neuron.amazonaws.com/cxx11


To install a wheel, it is recommended to use the ``--no-deps`` flag since
versions of ``torch`` compiled using the cxx11 ABI are not distributed on this
index.

::

    pip install --index-url=https://pip.repos.neuron.amazonaws.com/cxx11 torch-neuron --no-deps


Specific versions of ``torch-neuron`` with cxx11 ABI support can be installed
just like standard versions of ``torch-neuron``.

::

    pip install --index-url=https://pip.repos.neuron.amazonaws.com/cxx11 "torch-neuron>=1.8" --no-deps
    pip install --index-url=https://pip.repos.neuron.amazonaws.com/cxx11 "torch-neuron==1.9.1" --no-deps
    pip install --index-url=https://pip.repos.neuron.amazonaws.com/cxx11 "torch-neuron<1.10" --no-deps

.. important::

    This pip index does not include a distribution of ``torch`` compiled with
    the new cxx11 ABI. The intent of this index is *only* to provide Neuron SDK
    wheels.

    The version of ``torch`` that is distributed on the default PyPI index is
    compiled with the old pre-cxx11 ABI.

    If a cxx11 ``torch-neuron`` package is installed *with* dependencies
    using the *default* PyPI index, then the installed version of ``torch`` will
    be using the pre-cxx11 ABI and ``torch-neuron`` will be using the cxx11
    ABI. This ABI mismatch will lead to errors in both python usage and at link
    time for non-python applications.

FAQ
^^^

When should I use a cxx11 torch-neuron wheel?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributions compiled with the new cxx11 ABI should only be used in the
following cases:

    1. You have built your own version of ``torch`` which uses the new cxx11 ABI and
       need a corresponding version of ``torch-neuron`` that is compatible.
    2. You are compiling an application against a ``libtorch``
       which uses the cxx11 ABI and would like to include
       ``libtorchneuron.so`` as well. Torch distributes these cxx11 ``libtorch``
       libraries with a ``libtorch-cxx11`` prefix.

        Example:

        ::

            https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcpu.zip


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

    pip download --index-url=https://pip.repos.neuron.amazonaws.com/cxx11 torch-neuron --no-deps
    wheel unpack torch_neuron-*.whl

If the exact version of the ``torch-neuron`` package is known and no
python/pip is available in the build environment, an alternative to is fetch the
package file directly and ``unzip`` the wheel:

.. code::

    wget https://pip.repos.neuron.amazonaws.com/cxx11/torch-neuron/torch_neuron-<VERSION>-py3-none-any.whl
    unzip torch_neuron-<VERSION>-py3-none-any.whl


.. _pytorch-cxx11-versioning:

How can I know which ABI torch-neuron is using?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Packages which use the pre-cxx11 ABI have no local identifier and use the
following version scheme:

::

    <torch version>.<neuron version>

Packages which use the cxx11 ABI have a ``+cxx11`` local identifier and use
following version scheme:

::

    <torch version>.<neuron version>+cxx11


This allows the ABI to be validated in the by inspecting the local identifier
(or version suffix).

Example:
::

    1.8.1.0.0.0.0+cxx11
    1.9.1.0.0.0.0+cxx11
    1.10.2.0.0.0.0+cxx11


How can I know which ABI torch is using?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``torch`` python package provides an API at the that allows you to check if
the underlying ``libtorch`` was compiled with the cxx11 ABI:

.. code:: python

    import torch
    torch.compiled_with_cxx11_abi()  # True/False

Currently ``torch-neuron`` does not have an equivalent API. If the cxx11 ABI was
used, it will be visible in the version string (See :ref:`pytorch-cxx11-versioning`).


Troubleshooting
^^^^^^^^^^^^^^^

What python errors could I see if I mix ABI versions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a version of ``torch`` compiled with the cxx11 ABI will trigger an error
in the python interpreter when importing a version of ``torch-neuron`` using
the old (pre-cxx11) ABI from the standard index. This will manifest as an
error when the ``import torch_neuron`` statement is executed.

::

    Traceback (most recent call last):
      File "/python3.7/site-packages/torch_neuron/__init__.py", line 64, in <module>
        _register_extension()
      File "/python3.7/site-packages/torch_neuron/__init__.py", line 60, in _register_extension
        torch.ops.load_library(neuron_op_filename)
      File "/python3.7/site-packages/torch/_ops.py", line 110, in load_library
        ctypes.CDLL(path)
      File "/python3.7/ctypes/__init__.py", line 364, in __init__
        self._handle = _dlopen(self._name, mode)
    OSError: /python3.7/site-packages/torch_neuron/lib/libtorchneuron.so: undefined symbol: _ZN5torch6detail10class_baseC2ERKSsS3_SsRKSt9type_infoS6_


Similarly if using the standard pre-cxx11 version of ``torch`` with the cxx11
version of ``torch-neuron`` will also cause an error upon import.

::

    Traceback (most recent call last):
      File "/python3.7/site-packages/torch_neuron/__init__.py", line 79, in <module>
        _register_extension()
      File "/python3.7/site-packages/torch_neuron/__init__.py", line 75, in _register_extension
        torch.ops.load_library(neuron_op_filename)
      File "/python3.7/site-packages/torch/_ops.py", line 110, in load_library
        ctypes.CDLL(path)
      File "/python3.7/ctypes/__init__.py", line 364, in __init__
        self._handle = _dlopen(self._name, mode)
    OSError: /python3.7/site-packages/torch_neuron/lib/libtorchneuron.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE


In either of these cases, the remedy is to ensure that the ABI of the ``torch``
distribution matches the ABI of the ``torch-neuron`` distribution.

What compiler/linking errors could I see if I mix ABI versions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you link an application which uses the old (pre-cxx11) ABI
``libtorchneuron.so`` with a cxx11 version of ``torch``, this will trigger a
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
cxx11 ``libtorchneuron.so`` library is used with the pre-cxx11 ``libtorch``:

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

The ``torch`` ABI must match the ``torch-neuron`` ABI or an error will occur.
