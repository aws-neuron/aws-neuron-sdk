.. _neuron-tools:

.. meta::
   :description: Developer tools for profiling, monitoring, and analyzing machine learning workloads on AWS Neuron devices.
   :keywords: AWS Neuron, developer tools, profiler, monitoring, analysis, TensorBoard, visualization, debugging, optimization
   :date-modified: 12/02/2025

Developer Tools
================

AWS Neuron provides a comprehensive suite of developer tools for optimizing, monitoring, and debugging machine learning workloads on AWS Inferentia and Trainium accelerators. These tools enable developers to gain deep insights into model performance, system utilization, and hardware behavior to maximize the efficiency of ML applications running on Neuron-enabled instances.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Neuron Agentic Development
      :link: /tools/neuron-agentic-development/index
      :link-type: doc
      :class-header: sd-bg-primary sd-text-white

      AI agents and skills for developing on Trainium and Inferentia. Write NKI kernels, debug compilation errors, profile performance, and port HuggingFace models to NxD Inference using natural language.

   .. grid-item-card:: Neuron Explorer
      :link: /tools/neuron-explorer/index
      :link-type: doc
      :class-header: sd-bg-primary sd-text-white

      Neuron Explorer is a suite of tools designed to support ML engineers throughout their development journey on AWS Trainium, from model development through debugging, profiling, analysis, and optimization.

   .. grid-item-card:: System Tools
      :link: /tools/neuron-sys-tools/index
      :link-type: doc
      :class-header: sd-bg-primary sd-text-white
        
      Command-line utilities for monitoring, debugging, and managing AWS Neuron devices, including neuron-monitor, neuron-top, neuron-ls, and more.

   .. grid-item-card:: Third Party Tools
      :link: /tools/third-party-solutions
      :link-type: doc
      :class-header: sd-bg-primary sd-text-white
        
      Third-party tools and integrations that support the AWS Neuron development experience, including monitoring, visualization, and optimization solutions.

..
   .. grid-item-card:: AP Visualizer
      :link: ap-visualizer/ap-visualizer.html
      :link-type: url
      :class-header: sd-bg-primary sd-text-white
        
      Visualize access patterns of tensors on Neuron devices.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Tutorials
      :link: /tools/tutorials/index
      :link-type: doc
      :class-header: sd-bg-secondary sd-text-white

      Tutorials for how to utilize all Neuron Tools.

   .. grid-item-card:: Release Notes
      :link: /release-notes/components/dev-tools
      :link-type: doc
      :class-header: sd-bg-secondary sd-text-white

      Latest updates, new features, and improvements to Neuron Tools and Neuron Explorer.

.. toctree::
   :maxdepth: 1
   :hidden:

   Neuron Agentic Development </tools/neuron-agentic-development/index>
   System Tools </tools/neuron-sys-tools/index>
   Third-party Tools </tools/third-party-solutions>
   Tutorials </tools/tutorials/index>
   Release Notes </release-notes/components/dev-tools>
