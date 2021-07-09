.. _neuron-gettingstarted:

QuickStart
============
|image|

 
.. |image| image:: /images/neuron-devflow.jpg
   :width: 500
   :alt: Neuron developer flow
   
A typical Neuron developer flow includes compilation phase and then deployment (inference) on inf1 instance/s.

To quickly start developing with Neuron:

1. Setup your environment to compile and deploy on Inf1 instance/s:

   * :ref:`ec2-then-ec2-setenv`
    
2. Run a tutorial from one of the leading machine learning frameworks supported by Neuron:

   * :ref:`pytorch-tutorials`
   * :ref:`tensorflow-tutorials`
   * :ref:`mxnet-tutorials`

3. Explore more flows to develop with Neuron:

   * :ref:`neuron-devflows`

4. Learn more about Neuron

   * :ref:`neuron-fundamentals`
   * :ref:`neuron-appnotes`
   * :ref:`neuron-models`
   * :ref:`neuron-roadmap-and-support`


Customers can train their models anywhere and easily migrate their ML applications to Neuron and run their high-performance production predictions with Inferentia. Once a model is trained to the required accuracy, model is compiled to an optimized binary form, referred to as a Neuron Executable File Format (NEFF), and loaded by the Neuron runtime driver to execute inference input requests on the Inferentia chips. Developers have the option to train their models in fp16 or keep training in 32-bit floating point for best accuracy and Neuron will auto-cast the 32-bit trained model to run at speed of 16-bit using bfloat16.
