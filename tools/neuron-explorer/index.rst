.. meta::
   :description: Neuron Explorer documentation for performance profiling, debugging, and optimization of ML workloads on AWS Trainium and Inferentia.
   :date-modified: 12/02/2025

.. _neuron-explorer-home:

Neuron Explorer
=================

.. important::

    Neuron Explorer is in active development! At this time, it does not support system level profiling. For a stable user experience and system profiling, see :ref:`Neuron Profiler 2.0 <neuron-profiler-2-0-guide>` and :ref:`Neuron Profiler <neuron-profile-ug>`. 
    
Neuron Explorer is a suite of tools designed to support ML engineers throughout their development journey on AWS Trainium. Neuron Explorer helps developers maintain context, iterate efficiently, and focus on building and optimizing high-performance models. Developers can access Neuron Explorer from CLI, UI, or inside their IDE through VSCode integration.

Advanced Profiling Viewers
----------------------------

.. note::

    Neuron Explorer will replace Neuron Profiler and Neuron Profiler 2.0 in a future release. Please see :ref:`neuron-explorer-faq` for more details.

Neuron Explorer includes improvements over prior profiling workflows supported by Neuron Profiler and Neuron Profiler 2.0. Neuron Explorer enables ML performance engineers to trace execution from source code down to hardware operations, enabling detailed analysis of model behavior at every layer of the stack. The suite of tools supports both single-node and distributed applications, allowing developers to analyze workloads at scale. 

Getting Started
---------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Get Started
      :link: get-started
      :link-type: doc
      :class-card: sd-border-1

      Set up Neuron Explorer, launch the web UI, and configure SSH tunneling for secure access to profiling data.

   .. grid-item-card:: Launch Profiles via CLI, UI, or IDE
      :link: how-to-profile-workload
      :link-type: doc
      :class-card: sd-border-1

      Learn how to capture and launch the Neuron Explorer UI, use the Profile Manager, and view results in VSCode.

Visualization and Analysis
---------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Device Viewer
      :link: overview-device-profiles
      :link-type: doc
      :class-card: sd-border-1

      Explore hardware-level execution with timeline view, operator table, event details, annotations, dependency highlighting, search, and more analysis features.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Hierarchy Viewer
      :link: overview-hierarchy-view
      :link-type: doc
      :class-card: sd-border-1

      Visualize the entire execution from model layers down to hardware execution, supporting interactivity with device viewer and source code linking.

   .. grid-item-card:: Source Code Viewer
      :link: how-to-link-view-source-code
      :link-type: doc
      :class-card: sd-border-1

      Navigate between NKI and PyTorch source code and profile data with bidirectional linking and highlighting.

   .. grid-item-card:: Summary Viewer
      :link: overview-summary-page
      :link-type: doc
      :class-card: sd-border-1

      Get streamlined performance insights and optimization recommendations with high-level metrics and visualizations.

   .. grid-item-card:: AI Recommendation Viewer
      :link: overview-ai-recommendations
      :link-type: doc
      :class-card: sd-border-1

      Get AI powered bottleneck analysis and optmization recommendations for NKI profiles.

Tutorials
----------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Profile a NKI Kernel
      :link: /nki/deep-dives/use-neuron-profile
      :link-type: doc
      :class-card: sd-border-1

      Learn how to profile a NKI kernel with Neuron Explorer.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Multi-node Training
      :link: /tools/tutorials/multinode-training-model-profiling
      :link-type: doc
      :class-card: sd-border-1

      Profile multi-node training jobs with SLURM scheduling and visualize distributed workload performance.

   .. grid-item-card:: vLLM Performance
      :link: /tools/tutorials/performance-profiling-vllm
      :link-type: doc
      :class-card: sd-border-1

      Capture and analyze system-level and device-level profiles for vLLM inference workloads on Trainium.


.. _download-neuron-explorer-vsix:

Download the Neuron Explorer Visual Studio Code Extension
---------------------------------------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-card: sd-border-2

      **Get the Neuron Explorer VSCode Extension**
      ^^^
      :download:`Neuron Explorer Visual Studio Code Extension </tools/neuron-explorer/downloads/neuron.explorer-2.27.0.vsix>`

Once downloaded, open the command palette by pressing **CMD+Shift+P** (MacOS) or **Ctrl+Shift+P** (Windows), type ``> Extensions: Install from VSIX...`` and press **Enter**. When you are prompted to select a file, select ``neuronXray-external-v1.1.0.vsix`` and then the **Install** button (or press **Enter**) to install the extension.


.. _neuron-explorer-faq:

Neuron Explorer FAQ
--------------------

What can I expect from the Neuron Explorer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this time, Neuron Explorer features an enhanced device profiling experience. In future releases, Neuron Explorer will expand to provide support for the entire ML development journey on Trainium, with additional system and device level profiling viewers and features, debugging capabilities, IDE tooling, and enhanced recommendation and analysis tools.

What is the difference between device-level and system-level profiling?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Device-level profiling captures hardware execution data from NeuronCores, including compute engine instructions, DMA operations, and hardware utilization. Use device-level profiling to analyze hardware performance, identify compute or memory bottlenecks, and optimize kernel implementations.

System-level profiling captures software execution data, including framework operations, Neuron Runtime API calls, CPU utilization, and memory usage. Use system-level profiling to analyze framework overhead, identify CPU bottlenecks, and debug runtime issues.

For comprehensive performance analysis, you must consider both profiling levels to understand the complete picture from application code to hardware execution.

Should I continue using Neuron Profiler or migrate to Neuron Explorer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use **Neuron Profiler or Profiler 2.0** if you need both device-level and system-level profiling in a single workflow. These are the current default tools and provide the most comprehensive profiling experience with a stable, proven interface.

Use **Neuron Explorer** if your analysis focuses on hardware-level performance and you want enhanced capabilities such as hierarchical profiling, bidirectional code linking, or AI-powered recommendations. Neuron Explorer is particularly effective for NKI kernel development, hardware bottleneck analysis, and iterative optimization workflows that benefit from IDE integration and faster performance.

How do I see end to end profile for my workload with the latest features for both system and device profiling?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron Explorer** currently provides next generation device-level profiling features. For latest system-level profiling support, use **Neuron Profiler 2.0** until Neuron Explorer includes this capability.

For guidance on how to use the **Neuron Explorer** for device profiling and **Neuron Profiler 2.0** for system profiling, see :ref:`tutorials <neuron-tools-tutorials>`.

Is Neuron Explorer going to replace Neuron Profiler and Neuron Profiler 2.0? When will this happen?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, Neuron Profiler and Profiler 2.0 are fully supported as the default tools. Neuron Explorer is in public beta with device-level profiling capabilities.

Neuron Explorer will become the default profiling tool once system-level profiling is integrated. Neuron Profiler and Profiler 2.0 will remain supported until Neuron Explorer enters GA classification. When Neuron Profiler and Profiler 2.0 enter end-of-support, they will no longer receive updates or technical support, though they will remain accessible through the ``neuron-profile`` package in previous releases. Users should plan to migrate to Neuron Explorer before the end-of-support date.

Are my existing profiles compatible with Neuron Explorer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Neuron Explorer is backwards compatible with profile data captured using Neuron Profiler or Profiler 2.0. Existing profile files must be reprocessed before viewing in Neuron Explorer, but you do not need to recapture them. See :ref:`new-neuron-profiler-setup`.

.. toctree::
   :hidden:
   :maxdepth: 1

   Get Started <get-started>
   Launch Profiles via UI, CLI, IDE <how-to-profile-workload>
   Device Viewer <overview-device-profiles>
   Hierarchy Viewer <overview-hierarchy-view>
   Source Code Viewer <how-to-link-view-source-code>
   Summary Viewer <overview-summary-page>
   AI Recommendation Viewer <overview-ai-recommendations>
