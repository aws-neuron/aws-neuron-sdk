.. meta::
   :description: Learn about the different NKI beta designations in the AWS Neuron SDK.
   :date-modified: 12-02-2025

.. _nki-beta-versions:

About NKI Beta Versions
=========================

This page provides details on the beta versions of the Neuron Kernel Interface (NKI) and its ongoing evolution.


Why is NKI considered in “Beta” development?
---------------------------------------------

NKI is still in active development. While we have made leaps and bounds in improving the overall language, compiler, libraries, and tools supporting NKI, we still have some polish that needs to be applied before we can call NKI GA. 

NKI Beta 2 Features
--------------------

First and foremost, NKI Beta 2 introduces a large number of changes to the NKI language. In Beta 2, we have sought to constrain the language to the minimum set that is known to be required to build high performance kernels for popular models today. Constraining the language thus allows us to thoughtfully and intentionally grow the language in the future in an additive fashion without making breaking changes.

Beta 2 also introduces a whole new compiler front end rewritten from the ground up to accommodate this constrained language. We have implemented an LL(k) parser that provides parsing and semantic errors up front, which drastically improves the development experience. Additionally, we have also introduced a number of new APIs that give developers more control over both allocations and scheduling to make the language even more powerful.

To learn more about the features in Beta 2, see the overview documentation here: :doc:`About NKI </nki/get-started/about/index>`.

To use the Beta 2 language and compiler, import the new ``nki.*`` namespace in your code and annotate your top-level kernel function with ``@nki.jit``. 

NKI Beta 1 Features
--------------------

While we are in the process of revising the language, we want to ensure continuity for customers with existing kernels already in the wild. As a result, the Beta 2 compiler also comes with a compatibility mode, allowing all Beta 1 kernels to continue working. :ref:`We are, however, ending support for the Beta 1 language and compiler <announce-eos-nki-beta-1>`. This means that the Beta 2 compiler is the final compiler version where the Beta 1 language will be accepted; subsequent releases will not include support for the Beta 1 language and APIs.

To utilize the Beta 1 language and compiler, continue importing the legacy ``neuronxcc.nki.*`` namespace into your code and annotate your top-level kernel function with ``@nki.jit``. We are however, deprecating the Beta 1 language and compiler.

Also, a single file cannot utilize both versions of the language. While it is possible to have a model that uses both versions (Beta 1 and Beta 2) of the language, this is not an officially supported scenario.

NKI Beta Support Information
-----------------------------

NKI Beta 1 is now officially deprecated and will receive minimal support. The majority of support cases will become requests to migrate kernels to the Beta 2 version of the language before proceeding. For details on migrating your NKI kernels from Beta 1 to Beta 2, :doc:`/nki/get-started/nki-language-guide`. 

For support with the Beta 2 language and compiler, file a `GitHub issue <https://github.com/aws-neuron/aws-neuron-sdk/issues>`__ and provide us the details of your experience or issue. Other contact details can be found here: :ref:`contact-us`.
