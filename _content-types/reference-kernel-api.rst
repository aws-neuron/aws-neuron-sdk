.. meta::
    :description: API reference for the {kernel-name} kernel included in the NKI Library .
    :date-modified: MM/DD/YYYY

.. currentmodule:: {kernel namespace}.{kernel module path}

RMSNorm-Quant Kernel API Reference
==================================

This topic provides the API reference for the ``{kernel name}`` kernel. The kernel performs optional RMS normalization followed by quantization to ``fp8``.

The kernel supports:

* {feature 1}
* {feature 2}
* {feature 3}
* ... {more features as needed}

Background
-----------

The ``{kernel}`` kernel ... {description of kernel functionality based in sources}

For detailed information about the mathematical operations and implementation details, refer to the :doc:`{kernel name} Kernel Design Specification </nki/library/specs/{kernel-spec-doc-file-link}>`.

API Reference
--------------

{kernel argument class name}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: {kernel argument class name}

   {kernel name} Kernel arguments.

   .. py:attribute:: {attribute-1}
      :type: {attribute-1-type}

      {description from docstring}

   .. py:attribute:: {attribute-1}
      :type: {attribute-1-type}

      {description from docstring}

    {more attributes as needed}

   .. py:method:: {method syntax} -> {return type}

      {description from docstring}

   .. py:method:: {method syntax} -> {return type}

      {description from docstring}

   **Raises**:

   * **{exception-1}** – {when exception is raised}
   * **{exception-1}** – {when exception is raised}

{kernel API function name in code}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: rmsnorm_quant_kernel(hidden: nt.tensor, ln_w: nt.tensor, kargs: RmsNormQuantKernelArgs)

   {definition of method used to instantiate or invoke kernel here, from source docstrings}
   
   {params and types with descriptions from source docstrings}

Implementation Details
-----------------------

The kernel implementation includes several key optimizations:

1. **{optimization-or-feature}**: {description}
2. **{optimization-or-feature}**: {description}
3. **{optimization-or-feature}**: {description}

Example
--------

The following is a simple example of how to use the ``{kernel}`` kernel:

.. code-block:: python

   # Code here -- need usage example in pedagogical style.

See Also
--------

* :doc:`{kernel} </nki/library/specs/{link-to-kernel-spec}>`

