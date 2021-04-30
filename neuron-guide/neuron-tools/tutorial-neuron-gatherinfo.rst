.. _neuron_gatherinfo:

Using Neuron GatherInfo Tool to collect debug and support information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
========

The Neuron GatherInfo tool ``neuron-gatherinfo.py`` can assist in
automating the collection and packaging of information from Neuron SDK
tools that is useful to both user and AWS for issue resolution. The tool
gathers log files and other system information. If being used to supply
that info to AWS, the tool will redact proprietary and confidential
information. The GatherInfo tool is supplied in source code form -
available here: :neuron-gatherinfo-tree:`Neuron Gatherinfo <neuron-gatherinfo/>`

The tool enables developers to gather compiler and inference/runtime
logs. Additionally, the common usage is from within one of the supported
ML frameworks that have been integrated with Neuron, and information can
be captured from those compile/runtime environments using the
frameworks.

Steps Overview:
~~~~~~~~~~~~~~~

1. Obtain a copy of neuron-gatherinfo.py from
   :neuron-tools-tree:`Neuron Gatherinfo <neuron-gatherinfo/>`
2. Install into a location in your $PATH or into a location from where
   you can launch the script
3. Use with compile and/or runtime environments

Neuron-CC information gathering
-------------------------------

Step 1: Re-run the compile steps for your workload with increased verbosity or debug levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  For TensorFlow-Neuron, change the Python code as shown. Note that
   ‘compiler-workdir’ is expected to be an empty directory to prevent
   files from other runs from interfering with the information
   gathering. The call to the compile function has to be augmented with
   the **verbose** and the \**compiler_workdir \**arguments. In
   addition, please capture the stdout messages into a file (for
   example, by redirecting the stdout to a file)

::

   tfn.saved_model.compile(model_dir, compiled_model_dir, compiler_args=['--verbose', '2', '--pipeline', 'compile',  'SaveTemps'], compiler_workdir='./compiler-workdir')

-  For Neuron Apache MXNet (Incubating), add compiler arguments as shown below and run the
   compilation process from an empty workdir:

::

   import mxnet as mx
   import os

   from packaging import version
   mxnet_version = version.parse(mx.__version__)
   if mxnet_version >= version.parse("1.8"):
      import mx_neuron as neuron
   else: 
      from mxnet.contrib import neuron

   ...
   os.environ['SUBGRAPH_INFO'] = '1'
   compile_args = { '--verbose' : 2, '--pipeline' : 'compile', 'flags' : ['SaveTemps'] }
   csym, cargs, cauxs = neuron.compile(sym, args, auxs, inputs=inputs, **compile_args)

.. _step-2-run-neuron-gatherinfopy-to-gather-information-to-share:

Step 2: Run neuron-gatherinfo.py to gather information to share
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output result will be a tar.gz file.

Neuron Runtime information gathering
------------------------------------

Step 1: EXECUTE inference steps for your workload with increased verbosity or debug levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case of runtime information, the tool **neuron-dump.py** is used
by \**neuron-gatherinfo.py \**to gather that information. Make sure that
you have the neuron tools package (aws-neuron-tools) installed.

.. _step-2-run-neuron-gatherinfopy-to-gather-information-to-share-1:

Step 2: Run neuron-gatherinfo.py to gather information to share
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output result will be a tar.gz file.

Tool Usage Reference
====================

Run neuron-gatherinfo.py using the “—help“ option:

::

   bash $ ~/bin/neuron-gatherinfo.py --help
   usage: neuron-gatherinfo.py [-h] [--additionalfileordir ADDFLDIR] [-c CCDIR]
                               [-i] [-f FILTERFILE] [-m] -o OUTDIR [-r RTDIR] -s
                               STDOUT [-v]

       Usage: /home/user/bin/neuron-gatherinfo.py [options]
       This program is used to gather information from this system for analysis
       and debugging


   optional arguments:
     -h, --help            show this help message and exit
     --additionalfileordir ADDFLDIR
                           Additional file or directory that the user wants to
                           provide in the archive. The user can sanitize this
                           file or directory before sharing
     -c CCDIR, --compileroutdir CCDIR
                           Location of the neuron-cc generated files
     -i, --include         By default, only the lines containing (grep) patterns
                           like 'nrtd|neuron|kernel:' from the syslog are copied.
                           Other lines are excluded. Using this option allows the
                           timestamp section of other lines to be included. The
                           rest of the contents of the line itself are elided.
                           Providing the timestamp section may provide time
                           continuity while viewing the copied syslog file
     -f FILTERFILE, --filter FILTERFILE
     -m, --modeldata       By using this option, the entire compiler work
                           directory's contents will be included (excluding the
                           .pb files, unless an additional option is used). This
                           would include model information, etc. The files that
                           are included, by default, are these: graph_def.neuron-
                           cc.log, all_metrics.csv, hh-tr-operand-
                           tensortensor.json
     -o OUTDIR, --out OUTDIR
                           The output directory where all the files and other
                           information will be stored. The output will be stored
                           as an archive as well as the actual directory where
                           all the contents are copied. This will allow a simple
                           audit of the files, if necessary. *** N O T E ***:
                           Make sure that this directory has enough space to hold
                           the files and resulting archive
     -r RTDIR, --runtimeoutdir RTDIR
                           Location of the neuron runtime generated files
     -s STDOUT, --stdout STDOUT
                           The file where the stdout of the compiler run was
                           saved
     -v, --verbose         Verbose mode displays commands executed and any
                           additional information which may be useful in
                           debugging the tool itself

Examples
========

Example 1: no ML model information gathered (default behavior)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the tool will archive just the default information
gathering:

::

   bash $ sudo ~/bin/neuron-gatherinfo.py   -o compile-and-run-info-for-debugging-no-model-info  -i --verbose  -s stdout-from-compile_resnet50.out -c compiler-workdir

   Running cmd: lscpu and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo/report-lscpu.txt
   Running cmd: lshw and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo/report-lshw.txt
   Running cmd: lspci | grep -i Amazon and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo/report-lspci.txt
   Running cmd: neuron-cc --version and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo/report-neuron-cc.txt
   Running cmd: neuron-ls and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo/report-neuron-ls.txt
   <SNIP>
       ******
       Archive created at:
           /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo.tar.gz
       From directory:
           /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo
       ******


.. _example-2--model-ml-information-gathered-using-the-modeldata-option:

Example 2 : model ML information gathered using the “—modeldata” option
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the tool will archive the compiler work directory in
addition to the default information gathering

::

   bash $ sudo ~/bin/neuron-gatherinfo.py   -o compile-and-run-info-for-debugging  -i --verbose  -s stdout-from-compile_resnet50.out -c compiler-workdir --modeldata

   <SNIP>
   Running cmd: lscpu and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging/neuron-gatherinfo/report-lscpu.txt
   Running cmd: lshw and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging/neuron-gatherinfo/report-lshw.txt
   Running cmd: lspci | grep -i Amazon and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging/neuron-gatherinfo/report-lspci.txt
   Running cmd: neuron-cc --version and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo/report-neuron-cc.txt
   Running cmd: neuron-ls and capturing output in file: /home/user/tutorials-3/compile-and-run-info-for-debugging-no-model-info/neuron-gatherinfo/report-neuron-ls.txt
   <SNIP>

       ******
       Archive created at:
           /home/user/tutorials-3/compile-and-run-info-for-debugging/neuron-gatherinfo.tar.gz
       From directory:
           /home/user/tutorials-3/compile-and-run-info-for-debugging/neuron-gatherinfo
       ******


       **************************
       Based on your command line option, we're also packaging these files:

           graph_def.neuron-cc.log
           all_metrics.csv
           hh-tr-operand-tensortensor.json

       And this directory: /home/user/tutorials-3/compiler-workdir

       **************************

