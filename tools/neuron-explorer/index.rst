.. meta::
   :description: Neuron Explorer documentation for performance profiling, debugging, and optimization of ML workloads on AWS Trainium and Inferentia.
   :date-modified: 12/02/2025

.. _neuron-explorer-home:

Neuron Explorer
=================

.. important::

    Neuron Explorer is the recommended profiling tool for AWS Neuron workloads. It provides end-to-end profiling support along with the latest features and an improved user experience. 
    
    **Note:** Neuron will end support for :ref:`Neuron Profiler 2.0 <neuron-profiler-2-0-guide>` and :ref:`Neuron Profiler <neuron-profile-ug>` in Neuron 2.29 release. Users are encouraged to migrate to Neuron Explorer. Please see :doc:`migration-faq` and :ref:`neuron-explorer-faq` for more details.
    
Neuron Explorer is a suite of tools designed to support ML engineers throughout their development journey on AWS Trainium. Neuron Explorer helps developers maintain context, iterate efficiently, and focus on building and optimizing high-performance models. Developers can access Neuron Explorer from CLI, UI, or inside their IDE through VSCode integration.

Profiling Viewers
--------------------

Neuron Explorer enables ML performance engineers to trace execution from source code down to hardware operations, enabling detailed analysis of model behavior at every layer of the stack. The suite of tools supports both single-node and distributed applications, allowing developers to analyze workloads at scale. 

Getting Started
---------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Get Started
      :link: get-started
      :link-type: doc
      :class-card: sd-border-1

      Set up Neuron Explorer, launch the web UI, and configure SSH tunneling for secure access to profiling data.

   .. grid-item-card:: Capture and View Profiles
      :link: how-to-profile-workload
      :link-type: doc
      :class-card: sd-border-1

      Learn how to capture and view profiles in the Neuron Explorer UI or directly in your IDE via VSCode Integration.

Visualization and Analysis
---------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Device Trace Viewer
      :link: overview-device-profiles
      :link-type: doc
      :class-card: sd-border-1

      Explore hardware-level execution with timeline view, operator table, event details, annotations, dependency highlighting, search, and more analysis features.

   .. grid-item-card:: System Trace Viewer
      :link: overview-system-profiles
      :link-type: doc
      :class-card: sd-border-1

      Explore system-level execution with timeline view and more analysis features.


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

   .. grid-item-card:: Database Viewer
      :link: overview-database-viewer
      :link-type: doc
      :class-card: sd-border-1

      Develop your own analyses, examine profiling data stored in database tables, or run ad-hoc queries during performance analysis. 

   .. grid-item-card:: Tensor Viewer
      :link: overview-tensor-viewer
      :link-type: doc
      :class-card: sd-border-1

      Viewing tensor information including names, sizes, shapes, and memory usage details.

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

Additional Resources
--------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Viewing Profiles with Perfetto
      :link: view-perfetto
      :link-type: doc
      :class-card: sd-border-1

      Learn how to view Neuron Explorer profiles using the Perfetto UI for trace analysis.

.. _download-neuron-explorer-vsix:

Download the Neuron Explorer Visual Studio Code Extension
---------------------------------------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-card: sd-border-2

      **Get the Neuron Explorer VSCode Extension**
      ^^^
      :download:`Neuron Explorer Visual Studio Code Extension </tools/neuron-explorer/downloads/aws-neuron.neuron-explorer-2.28.0.vsix>`

Once downloaded, open the command palette by pressing **CMD+Shift+P** (MacOS) or **Ctrl+Shift+P** (Windows), type ``> Extensions: Install from VSIX...`` and press **Enter**. When you are prompted to select a file, select ``aws-neuron.neuron-explorer-2.28.0.vsix`` and then the **Install** button (or press **Enter**) to install the extension.

.. _neuron-explorer-faq:

Neuron Explorer FAQ
-------------------

What can I expect from the Neuron Explorer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron Explorer provides a comprehensive profiling experience with both device-level and system-level profiling support. Neuron Explorer features an enhanced profiling experience with hierarchical profiling, bidirectional code linking, AI-powered recommendations, IDE integration, and more. In future releases, Neuron Explorer will continue to expand with additional profiling viewers and features, debugging capabilities, and enhanced recommendation and analysis tools to support the entire ML development journey on Trainium.

What is the difference between device-level and system-level profiling?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Device-level profiling captures hardware execution data from NeuronCores, including compute engine instructions, DMA operations, and hardware utilization. Use device-level profiling to analyze hardware performance, identify compute or memory bottlenecks, and optimize kernel implementations.

System-level profiling captures software execution data, including framework operations, Neuron Runtime API calls, CPU utilization, and memory usage. Use system-level profiling to analyze framework overhead, identify CPU bottlenecks, and debug runtime issues.

Is Neuron Explorer going to replace Neuron Profiler and Neuron Profiler 2.0?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Neuron Explorer is the recommended profiling tool and replaces both Neuron Profiler and Profiler 2.0.

Neuron Profiler and Profiler 2.0 are supported for one final release. In Neuron 2.29 release, they will enter end-of-support and will no longer receive updates or technical support, though they will remain accessible through the ``neuron-profile`` package in previous releases. Users should migrate to Neuron Explorer now.

Are my existing profiles compatible with Neuron Explorer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Neuron Explorer is backwards compatible with profile data captured using Neuron Profiler or Profiler 2.0. Existing profile files must be reprocessed before viewing in Neuron Explorer, but you do not need to recapture them. See :ref:`new-neuron-profiler-setup`.

For detailed migration guidance, including CLI command mappings and feature comparisons, see the :doc:`migration-faq`.


.. toctree::
   :hidden:
   :maxdepth: 1

   Get Started <get-started>
   Neuron Profiler to Neuron Explorer Migration Guide <migration-faq>
   Capture and View Profiles <how-to-profile-workload>
   Device Trace Viewer <overview-device-profiles>
   System Trace Viewer <overview-system-profiles>
   Hierarchy Viewer <overview-hierarchy-view>
   Source Code Viewer <how-to-link-view-source-code>
   Summary Viewer <overview-summary-page>
   Database Viewer <overview-database-viewer>
   Tensor Viewer <overview-tensor-viewer>
   AI Recommendation Viewer <overview-ai-recommendations>
   View Profiles with Perfetto <view-perfetto>
   
