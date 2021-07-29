.. _tensorflow-neuron:

TensorFlow Neuron
=================

Neuron is integrated into TensorFlow, and provides you with a familiar environment to run inference using Inferentia based instances.

Neuron supports both Tensorflow 2.x, an eager-execution-based deep learning framework,
and TensorFlow 1.x, a static-graph-based deep learning framework. Currently,
the supported TensorFlow versions are 2.5.0, 2.4.2, 2.3.3, 2.2.3, 2.1.4, and 1.15.5.

**General**
 * :ref:`tensorflow-setup-env`
 * :ref:`Tutorials <tensorflow-tutorials>`

**TensorFlow 2.x**
   * :ref:`Neuron Tracing API <tensorflow-ref-neuron-tracing-api>`
   * :ref:`Accelerated Python APIs and graph operators <tensorflow-ref-neuron-accelerated-ops>`
   * :ref:`FAQ <tf2_faq>`

**TensorFlow 1.x**
   * :ref:`Neuron Compilation API <tensorflow-ref-neuron-compile-api>`
   * :ref:`Supported operators <neuron-cc-ops-tensorflow>`
   * :ref:`FAQ <tf1_faq>`

**Release Notes**
   * :ref:`tf_whatsnew`

.. toctree::
   :maxdepth: 1
   :hidden:

   ./env-setup
   Tutorials <./tutorials/index>
   TensorFlow 2.x Neuron Tracing API <./api-tracing-python-api>
   TensorFlow 2.x Accelerated Python APIs and graph operators <./tensorflow2-accelerated-ops>
   TensorFlow 1.x Neuron Compilation API <./api-compilation-python-api>
   TensorFlow 1.x Supported operators </release-notes/neuron-cc-ops/neuron-cc-ops-tensorflow>
   ./tf1_faq
   ./tf2_faq
   rn
