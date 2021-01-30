.. _conda-pytorch-release-notes:

PyTorch-Neuron Conda Package Release notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This document lists the release notes for the Neuron Conda-Pytorch
package.

[1.7.1.1.2.3.0]
^^^^^^^^^^^^^^^
Date: 1/30/2021

Included Neuron Packages
------------------------

neuron-cc-1.2.2.0

torch_neuron-1.7.1.1.2.3.0

Resolved Issues
---------------

Resolved the segmentation fault when enabling profiling using NEURON_PROFILE=<directory> environment variable for inference within a PyTorch-Neuron Conda environment (https://github.com/aws/aws-neuron-sdk/issues/230). This fix will be available in the next DLAMI release.


[1.5.1.1.2.3.0]
^^^^^^^^^^^^^^^
Date: 1/30/2021

Included Neuron Packages
------------------------

neuron-cc-1.2.1.0

torch_neuron-1.5.1.1.2.3.0

Resolved Issues
---------------

Resolved the segmentation fault when enabling profiling using NEURON_PROFILE=<directory> environment variable for inference within a PyTorch-Neuron Conda environment (https://github.com/aws/aws-neuron-sdk/issues/230). This fix will be available in the next DLAMI release.


[1.7.1.1.1.7.0]
^^^^^^^^^^^^^^^

Date: 12/23/2020

Included Neuron Packages
------------------------

neuron-cc-1.1.7.0

torch_neuron-1.7.1.1.1.7.0

Known Issues
------------

When enabling profiling using NEURON_PROFILE=<directory> environment variable for inference within a PyTorch-Neuron
Conda environment (such as the DLAMI aws_neuron_pytorch_p36 environment), running inference would result in segmentation
fault (https://github.com/aws/aws-neuron-sdk/issues/230). The workaround is to reinstall the PyTorch package of the
same version as installed. For example, if the installed PyTorch version is 1.7.1, please do:

.. code:: bash

    pip install --no-deps --force-reinstall torch==1.7.1

Similarly, if the installed PyTorch version is 1.5.1,

.. code:: bash

    pip install --no-deps --force-reinstall torch==1.5.1

[1.5.1.1.1.7.0]
^^^^^^^^^^^^^^^

Date: 12/22/2020

Included Neuron Packages
------------------------

neuron-cc-1.1.7.0

torch_neuron-1.5.1.1.1.7.0

[1.5.1.1.0.1978.0]
^^^^^^^^^^^^^^^^^^

Date: 11/17/2020

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.24045.0 <neuron-cc-rn>`

:ref:`torch_neuron-1.5.1.1.0.1978.0 <pytorch-neuron-rn>`

Known Issues
------------

-  Conda environment aws_neuron_pytorch_p36 of Conda DLAMI v36 cannot be
   updated to this latest (1.5.1.1.0.1978.0) PyTorch-Neuron Conda
   package using "conda update torch-neuron" command. To use the latest
   PyTorch-Neuron Conda package, please create a new Conda environment
   and install PyTorch-Neuron Conda package there using "conda install
   -c https://conda.repos.neuron.amazonaws.com torch-neuron". This issue
   is fixed in Conda DLAMI v37.

-  Conda environment aws_neuron_pytorch_p36 of Conda DLAMI v30 to v35
   can be updated using the following commands:

.. code:: bash

   conda install --force torch-neuron=1.5.1.1.0.1978.0
   conda install --force numpy=1.18.1

.. _1511017210_2010170:

[1.5.1.1.0.1721.0_2.0.1017.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 09/22/2020

.. _included-neuron-packages-1:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.20600.0 <neuron-cc-rn>`

:ref:`torch_neuron-1.0.1721.0 <pytorch-neuron-rn>`

Resolved Issues
---------------

When TorchVision is updated to version >= 0.5, running Neuron
compilation would crash with "Segmentation fault (core dumped)" error.

Known Issues
------------

-  When TorchVision is updated to version >= 0.5, running Neuron
   compilation would crash with "Segmentation fault (core dumped)"
   error. This issue is resolved with version
   1.5.1.1.0.1721.0_2.0.1017.0 of PyTorch-Neuron Conda package
   (9/22/2020 release).
-  When running PyTorch script in latest Torch-Neuron conda environment,
   you may see errors "AttributeError: module 'numpy' has no attribute
   'integer'" and "ModuleNotFoundError: No module named
   'numpy.core._multiarray_umath'". This is due to older version of
   numpy. Please update numpy to version 1.18 using the command "conda
   install --force numpy=1.18.1".
-  Due to changes to PyTorch-Neuron Conda package content in this
   release, updating from aws_neuron_pytorch_p36 of Conda DLAMI (v35 or
   earlier) would require the following to update:

.. code:: bash

   conda install --force torch-neuron=1.5.1.1.0.1721.0
   conda install --force numpy=1.18.1

.. _151102980_208800:

[1.5.1.1.0.298.0_2.0.880.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 08/08/2020

.. _included-neuron-packages-1:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.18001.0 <neuron-cc-10180010>`

:ref:`torch_neuron-1.0.1532.0 <neuron-torch-1015320>`

torch_neuron_base-1.5.1.1.0.298.0

.. _151102580_208710:

[1.5.1.1.0.258.0_2.0.871.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 08/05/2020

.. _included-neuron-packages-2:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.17937.0 <neuron-cc-10179370>`

:ref:`torch_neuron-1.0.1522.0 <neuron-torch-1015220>`

torch_neuron_base-1.5.1.1.0.258.0

.. _151102510_207830:

[1.5.1.1.0.251.0_2.0.783.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 07/16/2020

Now supporting Python 3.7 Conda packages in addition to Python 3.6 Conda
packages.

.. _included-neuron-packages-3:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.16861.0 <neuron-cc-10168610>`

:ref:`torch_neuron-1.0.1386.0 <neuron-torch-1013860>`

torch_neuron_base-1.5.1.1.0.251.0

.. _130102150-206330:

[1.3.0.1.0.215.0-2.0.633.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date 6/11/2020

.. _included-neuron-packages-4:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.15275.0 <neuron-cc-10152750>`

:ref:`torch_neuron-1.0.1168.0 <neuron-torch-1011680>`

torch_neuron_base-1.3.0.1.0.215.0

.. _130101700-203490:

[1.3.0.1.0.170.0-2.0.349.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date 5/11/2020

.. _included-neuron-packages-5:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.12696.0 <neuron-cc-10126960>`

:ref:`torch_neuron-1.0.1001.0 <neuron-torch-1010010>`

torch_neuron_base-1.3.0.1.0.170.0

.. _13010900_20620:

[1.3.0.1.0.90.0_2.0.62.0]
^^^^^^^^^^^^^^^^^^^^^^^^^

Date 3/26/2020

.. _included-neuron-packages-6:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.9410.0 <neuron-cc-1094100>`

:ref:`torch_neuron-1.0.825.0 <neuron-torch-108250>`

torch_neuron_base-1.3.0.1.0.90.0

.. _13010900-109180:

[1.3.0.1.0.90.0-1.0.918.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 2/27/2020

.. _included-neuron-packages-7:

Included Neuron Packages
------------------------

:ref:`neuron_cc-1.0.7878.0 <neuron-cc-1078780>`

:ref:`torch_neuron-1.0.763.0 <neuron-torch-107630>`

torch_neuron_base-1.3.0.1.0.90.0

Known Issues and Limitations
----------------------------

:ref:`conda-tensorflow-release-notes`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _13010410-107370:

[1.3.0.1.0.41.0-1.0.737.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 1/27/2020

.. _included-neuron-packages-8:

Included Neuron Packages
------------------------

:ref:`neuron-cc-1.0.6801.0 <neuron-cc-1068010>`

:ref:`torch-neuron-1.0.672.0 <neuron-torch-106720>`

torch-neuron-base-1.3.0.1.0.41.0

.. _known-issues-and-limitations-1:

Known Issues and Limitations
----------------------------
