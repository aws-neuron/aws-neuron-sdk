.. _neuron-tools:

Neuron Tools
============

Neuron provides debugging and profiling tools with the visualization support of the TensorBoard plugin. The Neuron helper tools assist in best practices for model onboarding and performance optimizations. The debugging and profiling tools provide monitoring of runtime and performance metrics insights.

About Neuron tools
------------------

Neuron tools are essential utilities that provide deep insights into model performance, system utilization, and hardware behavior on AWS Neuron devices. The toolset spans multiple categories to support the complete ML development lifecycle:

**Profiling Tools**: Advanced performance analysis capabilities including detailed execution traces, memory usage patterns, and optimization recommendations through Neuron Profiler and TensorBoard integration.

**System Tools**: Real-time monitoring and diagnostic utilities for tracking device utilization, process management, and hardware health across Neuron instances.

**Performance Tools**: Benchmarking and evaluation frameworks like NeuronPerf for measuring model performance, comparing configurations, and validating optimization strategies.

**Helper Tools**: Utility functions for model validation, system information gathering, and troubleshooting common deployment issues.

.. dropdown::  System Tools 
        :class-title: sphinx-design-class-title-med
        :class-body: sphinx-design-class-body-small
        :animate: fade-in

        * :ref:`neuron-monitor-ug`
        * :ref:`neuron-top-ug`
        * :ref:`neuron-ls-ug`
        * :ref:`neuron-sysfs-ug`
        * :ref:`nccom-test`
        * :ref:`What's New <neuron-tools-rn>`

.. dropdown::  TensorBoard Plugin for Neuron
        :class-title: sphinx-design-class-title-med
        :class-body: sphinx-design-class-body-small
        :animate: fade-in

        * :ref:`neuronx-plugin-tensorboard`
        * :ref:`neuron-plugin-tensorboard`
        * :ref:`What's New <neuron-tensorboard-rn>`

.. toctree:: 
    :maxdepth: 1
    :hidden:
       
    /tools/helper-tools/index

.. dropdown::  Helper Tools 
        :class-title: sphinx-design-class-title-med
        :class-body: sphinx-design-class-body-small
        :animate: fade-in

        * :ref:`neuron_check_model`
        * :ref:`neuron_gatherinfo`

.. toctree:: 
    :maxdepth: 1
    :hidden:

    /tools/neuronperf/index

.. dropdown::  Performance and Benchmarks Tools 
        :class-title: sphinx-design-class-title-med
        :class-body: sphinx-design-class-body-small
        :animate: fade-in

        * :ref:`neuronperf`
        * :ref:`nccom-test`
        * :ref:`neuron-profile-ug`
        * :ref:`neuron-profiler-2-0-guide`

                    
.. dropdown::  Tutorials 
        :class-title: sphinx-design-class-title-med
        :class-body: sphinx-design-class-body-small
        :animate: fade-in  

        .. tab-set:: 

            .. tab-item:: TensorBoard

                * :ref:`neuronx-plugin-tensorboard`
                * :ref:`tb_track_training_minst`
                * :ref:`torch-neuronx-profiling-with-tb`

            .. tab-item:: System Tools

                * :ref:`track-system-monitor`





