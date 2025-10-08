
.. image:: /images/q-logo.png
       :scale: 30%
       :alt: Amazon Q
       :align: left
       :target: https://aws.amazon.com/q/

.. _amazon-q-dev:

Ask Q Developer
==================

Use Q Developer as your Neuron Expert for general Neuron technical guidance and to jumpstart your NKI kernel developement.

.. card:: Ask Q through Chat
            :link: https://aws.amazon.com/q/

.. card:: Ask Q in your IDE
            :link: https://marketplace.visualstudio.com/items?itemName=AmazonWebServices.amazon-q-vscode

.. card:: Guidelines for Quality Results
            :link: amazon-q-dev-guidelines
            :link-type: ref

.. _amazon-q-dev-guidelines:

Guidelines for Quality Results
------------------------------

1. Be Specific: Clearly state the task, desired output, and any
   constraints.
2. Provide Context: Mention specific versions, strategies, and any relevant performance requirements.
3. Request Complete Code: Ask for full implementations including
   imports, decorators, and main functions. Remember to always review and test the generated code before using it in
   production.
4. Ask for Explanations: Request comments or separate explanations for
   complex parts of the code.
5. Iterate: If the initial response isn’t satisfactory, refine your
   prompt based on the output. If you encounter issues or inaccuracies, consider rephrasing your
   prompt or breaking down complex tasks into smaller, more specific
   questions.
6. Fact check: Use Q as a starting point and supplement its output with official documentation, AWS NKI Samples repository, and your own expertise.

Example Prompts
~~~~~~~~~~~~~~~~~

.. note::
   Amazon Q Developer support for Neuron is currently in
   Beta. Therefore, Q may not always produce optimal or fully accurate results.

1. “Explain the key features and benefits of AWS Neuron Kernel Interface (NKI).”
2. "How do different parallelism strategies (data, pipeline, tensor) affect training performance on Neuron?"
3. “What are the best practices for optimizing matrix multiplication operations using Neuron Kernel Interface (NKI)?”
4. “Provide complete Neuron Kernel Interface (NKI) code for a matrix multiplication kernel, including imports, decorators, and explanations of key optimizations. Focus on efficient tiling and data movement strategies.”
