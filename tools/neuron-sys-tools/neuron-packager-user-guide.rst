.. _neuron-packager-ug:

Neuron Packager User Guide
==========================

.. contents:: Table of contents
    :local:
    :depth: 2

Overview
--------

``neuron-packager`` is a tool to inspect the contents of a NEFF.

Using neuron-packager
---------------------

.. rubric:: neuron-packager CLI

.. program:: neuron-packager

.. option:: neuron-packager [options] [subcommand] [subcommand-options]

    Available subcommands include ``create``, ``info``, ``optimize``, and ``unpack``.
    See individual subcommand for corresponding options.

    - :option:`-v, --version`: show version and exit

.. option:: neuron-packager create [create-options]

    Packages a NEFF from a tarball containing all NEFF files.

    - :option:`-i, --input` (string): input NEFF tarball

    - :option:`-v, --tool-version` (string) default=<tool version>: packaging version number

    - :option:`-k, --neff-version` (string) default=0.4: NEFF version number (Maj.Min)

    - :option:`-r, --header-version` (int) default=2: NEFF header version

    - :option:`-n, --name` (string): name of the compiled model

    - :option:`-t, --num-nc` (int) default=1: number of NeuronnCores required to run the graph

    - :option:`-o, --output` (string) default=NN: output file name (<output>.neff)

    - :option:`-e, --enable-feature` (string): supported neff features - choose any combination of
      [ ``bin-weights`` | ``collectives-offset`` | ``custom-ops`` | ``coalesced-cc``] by passing this flag multiple times

        - **bin-weights**: supports binary weight files instead of numpy files

        - **collectives-offset**: supports specifying an offset for collective operations in the compiled Neuron instruction binary

        - **custom-ops**: supports custom operators

        - **coalesced-cc**: supports coalesced collective compute operations in NEFF interface

.. option:: neuron-packager info [info-options]

    Displays the NEFF header as well as information on NeuronCore subgraphs and CPU operators

    - :option:`-w, --show-weights`: show weights

    - :option:`-s, --show-spills`: show spills

    - :option:`--json-output`: dump JSON formatted output

.. option:: neuron-packager optimize [optimize-options] [neff-files...]

    Optimizes the NEFF for faster loading

    - :option:`--keep-debug`: keep debug information and prettified JSONs

.. option:: neuron-packager unpack [unpack-options] [neff-file]

    Unpacks the given NEFF file

    - :option:`-o, --output` (string) default=basename of NEFF file: output directory

Examples
--------

The examples below use a compiled NEFF from the ``torch-neuronx`` MLP tutorial.  For more information,
please check out :ref:`neuronx-mlp-training-tutorial`.

The ``info`` subcommand displays information about the compiled model, such as the number of NeuronCores necessary to run and the inputs
and outputs of each NeuronCore subgraph and CPU operator (if applicable)

::

  $ neuron-packager info MODULE_0_SyncTensorsGraph.305_16554925436865022292_ip-172-31-55-249-6c54106d-25758-5f5ddf7b170ab.neff
  NEFF Header:
        Package Version:               2
        Header Size:                   1024      (bytes)
        Data Size:                     48924     (bytes)
        Major Version:                 1
        Minor Version:                 0
        Build Version:
        Number of Neuron cores:        1
        Hash:                          ec4b1b1fa8919a9be2c176fd63269511
        UUID:                          c3275f90b87d11ed80b60e6b4183ae7f
        Network Name:                  compiler_cache/neuron-compile-cache/USER_neuroncc-2.4.0.21+b7621be18/MODULE_16554925436865022292/MODULE_0_SyncTensorsGraph.305_16554925436865022292_ip-172-31-55-249-6c54106d-25758-5f5ddf7b170ab/2afbe37c-93ff-4f66-88d0-0e8e5d98e497/MODULE_0_SyncTensorsGraph
        Enabled Features:              N/A


  NEFF Nodes:
      NODE      Executor    Name    Variable    Size    Type    Format     Shape    DataType    TimeSeries
         9    NeuronCore    sg00
                                      input0       4      IN         N       [1]     float32
                                      input1      20      IN         N       [5]     float32
                                      input2     200      IN        NC    [5,10]     float32
                                      input3      40      IN        NC    [1,10]     float32
                                      input4     100      IN        NC     [5,5]     float32
                                      input5      20      IN         N       [5]     float32
                                      input6      40      IN        NC     [2,5]     float32
                                      input7       8      IN         N       [2]     float32
                                      input8       8      IN         N       [2]       int32
                                     output0     200     OUT        NC    [5,10]     float32    false
                                     output1      20     OUT         N       [5]     float32    false
                                    output10      40     OUT        NC     [2,5]     float32    false
                                    output11      20     OUT         N       [5]     float32    false
                                    output12     100     OUT        NC     [5,5]     float32    false
                                    output13      20     OUT         N       [5]     float32    false
                                    output14     200     OUT        NC    [5,10]     float32    false
                                     output2     100     OUT        NC     [5,5]     float32    false
                                     output3      20     OUT         N       [5]     float32    false
                                     output4      40     OUT        NC     [2,5]     float32    false
                                     output5       8     OUT         N       [2]     float32    false
                                     output6      40     OUT        NC    [1,10]     float32    false
                                     output7       8     OUT         N       [2]       int32    false
                                     output8       4     OUT         N       [1]     float32    false
                                     output9       8     OUT         N       [2]     float32    false

To inspect the contents of the NEFF, use the ``unpack`` subcommand.

::

  $ neuron-packager unpack MODULE_0_SyncTensorsGraph.305_16554925436865022292_ip-172-31-55-249-6c54106d-25758-5f5ddf7b170ab.neff
  Unpacking NEFF in "MODULE_0_SyncTensorsGraph.305_16554925436865022292_ip-172-31-55-249-6c54106d-25758-5f5ddf7b170ab" directory...
  
  $ ls -l MODULE_0_SyncTensorsGraph.305_16554925436865022292_ip-172-31-55-249-6c54106d-25758-5f5ddf7b170ab/
  total 84
  drwxr-xr-x 2 ubuntu ubuntu  4096 Mar  1 22:35 debug_info
  -rw-rw-r-- 1 ubuntu ubuntu   249 Mar  1 22:35 hlo_stats.json
  -rw-rw-r-- 1 ubuntu ubuntu  1205 Mar  1 22:35 info.json
  -rw-rw-r-- 1 ubuntu ubuntu   161 Mar  1 22:35 kelf-0.json
  -rw-rw-r-- 1 ubuntu ubuntu   366 Mar  1 22:35 metrics.json
  -rw-rw-r-- 1 ubuntu ubuntu 10082 Mar  1 22:35 neff.json
  -rw-r--r-- 1 ubuntu ubuntu 48924 Mar  1 22:35 neff.tgz
  drwxr-xr-x 2 ubuntu ubuntu  4096 Mar  1 22:35 sg00

The top level directory contains the high level information about the model, such as inputs and outputs for 
NeuronCore subgraphs and CPU operators.  Each NeuronCore subgraph has it's own subdirectory containing Neuron machine
instructions, tensor information, model parameters, and other components that support the subgraph's execution.

Re-packaging the neff can be done through the ``create`` subcommand.  It takes a tarball of the NEFF contents and appends a header to it.
After unpacking the NEFF, this tarball will already be present as ``neff.tgz``.  For NEFF versions greater than 2.0, feature bits can be used
to indicate which features must be supported by the Neuron runtime in order to be executed.  Any incompatible NEFFs will be rejected when attempting
to load the model.

..

  $ neuron-packager create -i neff.tgz
  Successfully generated: NN.neff

The ``optimize`` subcommand takes an input NEFF and replaces it with a version optimized for model load time.  When invoked,
any weights will be combined into a single file when possible, debug information will be removed, and all other necessary files will
be modified to reflect the previous changes.  In addition, the resulting NEFF will not be compressed, so the NEFF size may
increase.

::

$ neuron-packager optimize opt.neff
Successfully generated: opt.neff 

.. note::

  Since ``optimize`` removes debug information, some Neuron tools output may be missing information.
  For example, ``neuron-packager info`` will still display tensor sizes, but the shapes will be unknown.
  To keep the debug info, use the ``--keep-debug`` option.