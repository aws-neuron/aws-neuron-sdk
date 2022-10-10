.. _neuron-rounding-modes:

Neuron Rounding Modes
=====================

.. contents:: Table of contents
	:local:
	:depth: 1


.. _neuron-rounding-mode-rne:

Round Nearest, ties to Even (RNE)
---------------------------------

When the exact result of a floating point operation cannot be exactly
represented as a floating point value, it must be rounded. The IEEE
754-2008 standard defines the default rounding mode to be ‘Round
Nearest, ties to Even’ (RNE for short). Under this scheme, numbers are
rounded to the nearest representable value, and in case of a ‘tie’ (i.e.
the number is exactly between the two nearest representable values)
numbers will be rounded to the nearest even number.

All NeuronCore generations support the RNE rounding scheme, which is the
most commonly used rounding scheme for Machine Learning workloads. Below
is an illustration of the RNE rounding scheme: 

.. image:: /images/rne1.png
    :width: 700

.. image:: /images/rne2.png
    :width: 700

.. image:: /images/rne3.png
    :width: 700

.. _neuron-rounding-mode-sr:


Stochastic Rounding (SR)
------------------------

One downside of the RNE rounding scheme (and other rounding schemes
described in the IEEE 754-2008 standard), is that when adding floating
point values of significantly different magnitudes, rounding can squash
small values and prevent them from accumulating over time. 

To improve this, starting from the second generation of the NeuronCore
(NeuronCore-v2), customers can choose between the RNE rounding scheme
described above, and a second rounding scheme called ‘Stochastic
Rounding’ (SR for short). Stochastic rounding prevents the computation
precision-loss described above, by performing the rounding operations in
a probabilistic manner, according to the relative distance from the two
nearest representable values, as illustrated below: 

.. image:: /images/sr.png
    :width: 700


By performing the rounding in a probabilistic manner, this scheme allows
for small increments to accumulate over time, even when added to numbers
of significantly higher magnitude, which leads to more precise results
when performing large floating point computations (as done for machine
learning).


Quick Tests 
-----------

As an example, we examine the code-snippet below:

::

   import torch
   import torch_xla
   import torch_xla.core.xla_model as xm
   device = xm.xla_device()
   
   a = torch.tensor(1024.0).half().to(device)
   
   for i in range(2048) :
      a = (a + 0.5)
      xm.mark_step()
   
   print(a)


This code shows that rounding can significantly impact the calculation’s precision over time.
To use standard RNE rounding, use the environment variable ``NEURON_RT_STOCHASTIC_ROUNDING_EN=0``.
To enable stochastic rounding, use the environment variable ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``.

NOTE: Stochastic rounding mode is enabled by default in PyTorch-Neuron when XLA_USE_BF16=1.

The first test continues to show 1024 due to RNE rounding after each addition, and the second test shows result that is mostly in line with expectation.

::

   $ NEURON_RT_STOCHASTIC_ROUNDING_EN=0 python3 rounding_mode_test.py
   
   tensor(1024., device='xla:1', dtype=torch.float16)
   
   $ NEURON_RT_STOCHASTIC_ROUNDING_EN=1 python3 rounding_mode_test.py
   
   tensor(2056., device='xla:1', dtype=torch.float16)

