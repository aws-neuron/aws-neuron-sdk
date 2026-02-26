.. _neuron-2-27-0-tools:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Developer Tools component, version 2.27.0. Release date: 12/19/2025.

AWS Neuron SDK 2.27.0: Developer Tools Release Notes
====================================================

**Date of release**: December 19, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.27.0 release notes home <neuron-2-27-0-whatsnew>`

What's New
----------

Introducing Neuron Explorer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Introduces :doc:`Neuron Explorer </tools/neuron-explorer/index>`, a suite of tools designed to support ML engineers throughout their development journey on AWS Trainium. This release adds device profiling support, with new tools like :doc:`Hierarchy Viewer </tools/neuron-explorer/overview-hierarchy-view>`, :doc:`AI Recommendation Viewer </tools/neuron-explorer/overview-ai-recommendations>`, :doc:`Source Code Viewer </tools/neuron-explorer/how-to-link-view-source-code>`, and :doc:`Summary Viewer </tools/neuron-explorer/overview-summary-page>`.
* Introduced Neuron Explorer UI, CLI, and IDE integration via VSCode

Neuron Explorer includes device profiling support with the following tools:

* :doc:`Hierarchy Viewer </tools/neuron-explorer/overview-hierarchy-view>` — Visualizes the hierarchical structure of your model, allowing you to understand how different components interact and contribute to overall performance
* :doc:`AI Recommendation Viewer </tools/neuron-explorer/overview-ai-recommendations>` — Provides AI-driven recommendations for optimizing your model based on profiling data, helping you identify bottlenecks and areas for improvement
* :doc:`Source Code Viewer </tools/neuron-explorer/how-to-link-view-source-code>` — Links profiling data back to your source code, enabling you to quickly identify and address performance issues in your codebase
* :doc:`Summary Viewer </tools/neuron-explorer/overview-summary-page>` — Offers a high-level overview of your model's performance metrics, resource utilization, and optimization opportunities

* Trn3 support for ``neuron-monitor``, ``neuron-top``, ``neuron-ls``, and ``nccom-test``.

New Tutorials
^^^^^^^^^^^^^

Neuron 2.27.0 introduces :doc:`Neuron Explorer </tools/neuron-explorer/index>`, a suite of tools designed to support ML engineers throughout their development journey on AWS Trainium. Neuron Explorer provides insights into model performance, resource utilization, and optimization opportunities, helping developers to fine-tune their models for optimal performance on Trainium instances.

This release introduces enhanced in-UI performance, simplified setup, and key features for device profiling:  

- **Hierarchy Viewer**: Visualizes the hierarchical structure of your model, allowing you to understand how different components interact and contribute to overall performance.
- **AI Recommendation Viewer**: Provides AI-driven recommendations for optimizing your model based on profiling data, helping you identify bottlenecks and areas for improvement.
- **Source Code Viewer**: Links profiling data back to your source code, enabling you to quickly identify and address performance issues in your codebase.
- **Summary Viewer**: Offers a high-level overview of your model's performance metrics, resource utilization, and optimization opportunities.
- Added tutorials: :doc:`How to Profile a NKI Kernel </nki/deep-dives/use-neuron-profile>`, :doc:`Profiling Multi-Node Training Jobs </tools/neuron-explorer/how-to-profile-workload>`, and :doc:`Profiling a vLLM Inference Workload </tools/tutorials/performance-profiling-vllm>`

.. note::

   Neuron Explorer is in active development! At this time, it does not support system level profiling. For a stable user experience and system profiling, see :ref:`Neuron Profiler 2.0 <neuron-profiler-2-0-guide>` and :ref:`Neuron Profiler <neuron-profile-ug>`.
