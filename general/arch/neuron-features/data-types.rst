.. _neuron-data-types:

Data Types
==========

.. contents:: Table of contents
   :local:
   :depth: 2

Introduction
------------

:ref:`Inferentia <neurondevice_inferentia>` and :ref:`Trainium <neurondevice_trainium>` NeuronDevices include different NeuronCore versions, which support different data-types. This section describes what data-types are supported in each NeuronCore version, for details about NeuronCore versions see :ref:`neuron_hw_arch`.

NeuronCore v1 Data Types
------------------------

Neuron Data-Types
^^^^^^^^^^^^^^^^^

Neuron enables developers to choose from multiple data-types. The
supported data-types are FP32, FP16, and BF16. Developers can
train their models on their platform of choice (e.g. EC2 P3 instances),
and then easily move their trained models to EC2 Inf1 for execution.

.. raw:: html

  <style type="text/css">table, td, th { border: 1px solid black; padding: 5px; }
  </style>
  <table style="table-layout: fixed; width: 50%; border-spacing:0px;">
  	<tbody>
  		<tr>
  			<th width="20%">Data Type</th>
  			<th width="10%">S</th>
         <th colspan="8">Range</th>
  			<th colspan="23">Precision</th>
  		</tr>
  		<tr>
  			<td>FP32</td>
  			<td bgcolor="#ad3bff">1</td>
         <td bgcolor="#AFEFA9" colspan="8">8 bits</td>
  			<td bgcolor="#FAC49E" colspan="23">23 bits</td>
  		</tr>
  		<tr>
  			<td>BF16</td>
  			<td bgcolor="#ad3bff">1</td>
         <td bgcolor="#AFEFA9" colspan="8">8 bits</td>
  			<td style="border-right: 0px" colspan="13" />
  			<td colspan="3" />
  			<td bgcolor="#FAC49E" colspan="7">7 bits</td>
  		</tr>
  		<tr>
  			<td>FP16</td>
  			<td bgcolor="#ad3bff">1</td>
         <td colspan="3" />
         <td bgcolor="#AFEFA9" colspan="5">5 bits</td>
         <td colspan="13" />
  			<td bgcolor="#FAC49E" colspan="10">10 bits</td>
  		</tr>
  	</tbody>
  </table>
  <p/>

FP16/BF16 models
~~~~~~~~~~~~~~~~

Models natively trained in FP16/BF16 will be executed in their trained
data-types. This is a straightforward migration from the training
platform to Inf1.

FP32 models
~~~~~~~~~~~

Neuron SDK supports **automatic model conversion** from FP32 to BF16 by
default. This capability allows developers to train their models using
FP32 format for the highest accuracy, and achieve performance benefits
without having to worry about low-precision training (e.g. no need for
loss-scaling during training). ML models are typically robust to FP32 to
BF16 conversion, with minimal to no impact on accuracy. The conversion
accuracy is model dependent; therefore, users are encouraged to
benchmark the accuracy of the auto-converted model against the original
FP32 trained model.

When the compiler is supplied with an unmodified FP32 model input it
will automatically compile the model to run as BF16 on Inferentia. During
inference the FP32 input data will be auto-converted internally by
Inferentia to BF16 and the output will be converted back to FP32
data-type. For explicit FP16 inferencing, either use an FP16 trained
model, or use an external tool (like AMP) to make the explicit
conversions.

.. _neuron-data-types-v2:

NeuronCore v2 Data Types
------------------------

The NeuronCore v2 supports the following data types:

* 32 and 16-bit Floating Point (FP32 / FP16)
* TensorFloat-32 (TF32)
* Brain Floating Point (BFloat16)
* 8-bit Floating point with configurable range and precision (cFP8)
* Unsigned 8-bit integer (UINT8)

.. note:: Neuron Compiler support for cFP8 and UINT8 is planned for a future Neuron SDK release. For INT8, see `Neuron Compiler: Enable Neuron INT8 support <https://github.com/aws/aws-neuron-sdk/issues/36>`_ for details.

The layout for these is as follows:

.. raw:: html

  <style type="text/css">table, td, th { border: 1px solid black; padding: 5px; }
  </style>
  <table style="table-layout: fixed; width: 50%; border-spacing:0px;">
  	<tbody>
  		<tr>
  			<th width="20%">Data Type</th>
  			<th width="10%">S</th>
         <th colspan="8">Range</th>
  			<th colspan="23">Precision</th>
  		</tr>
  		<tr>
  			<td>FP32</td>
  			<td bgcolor="#ad3bff">1</td>
  			<td bgcolor="#AFEFA9" colspan="8">8 bits</td>
         <td bgcolor="#FAC49E" colspan="23">23 bits</td>
  		</tr>
  		<tr>
  			<td>TF32</td>
  			<td bgcolor="#ad3bff">1</td>
         <td bgcolor="#AFEFA9" colspan="8">8 bits</td>
  			<td colspan="13" />
  			<td bgcolor="#FAC49E" colspan="10">10 bits</td>
  		</tr>
  		<tr>
  			<td>BF16</td>
  			<td bgcolor="#ad3bff">1</td>
         <td bgcolor="#AFEFA9" colspan="8">8 bits</td>
  			<td style="border-right: 0px" colspan="13" />
  			<td colspan="3" />
  			<td bgcolor="#FAC49E" colspan="7">7 bits</td>
  		</tr>
  		<tr>
  			<td>FP16</td>
  			<td bgcolor="#ad3bff">1</td>
         <td colspan="3" />
  			<td bgcolor="#AFEFA9" colspan="5">5 bits</td>
  			<td colspan="13" />
  			<td bgcolor="#FAC49E" colspan="10">10 bits</td>
      </tr>
      <tr>
  			<td>FP8_e5m2</td>
  			<td bgcolor="#ad3bff">1</td>
         <td colspan="3" />
  			<td bgcolor="#AFEFA9" colspan="5">5 bits</td>
         <td style="border-right: 0px" colspan="18" />
         <td colspan="3" />
  			<td bgcolor="#FAC49E" colspan="2">2 bits</td>
  		</tr>
      <tr>
  			<td>FP8_e4m3</td>
  			<td bgcolor="#ad3bff">1</td>
         <td style="border-right: 0px" colspan="3" />
         <td colspan="1" />
  			<td bgcolor="#AFEFA9" colspan="4">4 bits</td>
         <td style="border-right: 0px" colspan="20" />
  			<td bgcolor="#FAC49E" colspan="3">3 bits</td>
  		</tr>
      <tr>
  			<td>FP8_e3m4</td>
  			<td bgcolor="#ad3bff">1</td>
         <td style="border-right: 0px" colspan="4" />
         <td colspan="1" />
  			<td bgcolor="#AFEFA9" colspan="3">3 bits</td>
         <td style="border-right: 0px" colspan="19" />
  			<td bgcolor="#FAC49E" colspan="4">4 bits</td>
  		</tr>
      <tr>
  			<td>UINT8</td>
  			<td colspan="1" />
  			<td bgcolor="#AFEFA9" colspan="8">8 bits</td>
         <td colspan="23" />
  		</tr>
  </table>
  <p/>



Model Type Conversion
^^^^^^^^^^^^^^^^^^^^^

The Neuron SDK supports automatic model conversion from FP32 to BF16 by default. This capability allows developers to train their models using FP32 format for the highest accuracy, and then achieve run-time performance benefits without having to worry about low-precision training (e.g. no need for loss-scaling during training). ML models are typically robust to FP32 to BF16 conversion, with minimal to no impact on accuracy. Since conversion accuracy is model dependent, users are encouraged to benchmark the accuracy of the auto-converted model against the original FP32 trained model.

See :ref:`Mixed Precision and Performance-accuracy Tuning for Training<neuronx-cc-training-mixed-precision>` for more details on supported data types and their properties.

The Neuron compiler offers the :option:`--auto-cast` and :option:`--auto-cast-type` options to specify automatic casting of FP32 tensors to other data types to address performance and accuracy tradeoffs. See the :ref:`Neuron Compiler CLI Reference Guide<neuron-compiler-cli-reference-guide>` for a description of these options.


NeuronCore v2 Rounding Modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because floating point values are represented by a finite number of bits, they cannot represent all real numbers accurately. Floating point calculations that exceed their defined data type size are rounded. The NeuronCore v2 performs a Round-to-Nearest (RNE) algorithm with ties to Even by default. It also provides a new Stochastic Rounding mode. When Stochastic Rounding is enabled, the hardware will round the floating point value up or down using a proportional probability. This could lead to improved model convergence. Use the environment variable NEURON_RT_STOCHASTIC_ROUNDING_EN to select a rounding mode.
