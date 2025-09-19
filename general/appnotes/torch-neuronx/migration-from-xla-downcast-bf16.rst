.. _migration_from_xla_downcast_bf16:

Migration From ``XLA_USE_BF16``/``XLA_DOWNCAST_BF16``
=====================================================

Introduction
------------

The environmental variables ``XLA_USE_BF16`` and ``XLA_DOWNCAST_BF16`` were created to provide an easy cast-to-bf16 option before automatic mixed-precision or ``model.to(torch.bfloat16)`` as available in Torch-XLA. Now that both automatic mixed precision and ``model.to(torch.bfloat16)`` are available in Torch-XLA,  ``XLA_USE_BF16`` and ``XLA_DOWNCAST_BF16`` are redundant and can be replaced with these options as a more familiar experience as on other platforms such as CPUs and GPUs. Using them in Torch-XLA 2.5+ would cause warnings to be displayed about their end-of-support. While they are still functional, their functionality will be removed in a future release (Torch-XLA 2.8) so the recommended changes below are available as replacement.

NeuronX Distributed Training has been updated to use some of the options below. Please see :ref:`standard_mixed_precision` for more information.

The changes recommended below can best be made to scripts running with Torch-XLA 2.5+. The same recommendations are also available in :ref:`pytorch-neuronx-programming-guide`.

.. note::

    This guide recommends the options below as replacement for ``XLA_USE_BF16`` and ``XLA_DOWNCAST_BF16``. Do not set ``XLA_USE_BF16=1`` or ``XLA_DOWNCAST_BF16=1`` when using the options below on Neuron devices. Using them will override the per-operator precision settings provided by the options and thus cause more operators to execute in bfloat16.

Full BF16 with stochastic rounding enabled
------------------------------------------

Previously, on torch-neuronx 2.1 and earlier, the environmental variables ``XLA_USE_BF16`` or ``XLA_DOWNCAST_BF16`` provided full casting to BF16 with stochastic rounding enabled by default. These environmental variables are deprecated in torch-neuronx 2.5, although still functional with warnings. To replace ``XLA_USE_BF16`` or ``XLA_DOWNCAST_BF16`` with stochastic rounding on Neuron, set ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1`` and use the ``torch.nn.Module.to`` method to cast model floating-point parameters and buffers to data-type BF16 as follows:

.. code:: python

    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"

    # model is created
    model.to(torch.bfloat16)

Stochastic rounding is needed to enable faster convergence for full BF16 model.

If the loss is to be kept in FP32, initialize it with ``dtype=torch.float`` as follows:

.. code:: python

    running_loss = torch.zeros(1, dtype=torch.float).to(device)

Similarly, if the optimizer states are to be kept in FP32, convert the gradients to FP32 before optimizer computations:

.. code:: python

    grad = p.grad.data.float()

For a full example, please see the :ref:`PyTorch Neuron BERT Pretraining Tutorial (Data-Parallel) <hf-bert-pretraining-tutorial>`, which has been updated to use ``torch.nn.Module.to`` instead of ``XLA_DOWNCAST_BF16``.

BF16 in GPU-compatible mode without stochastic rounding enabled
---------------------------------------------------------------

Full BF16 training in GPU-compatible mode would enable faster convergence without the need for stochastic rounding, but would require a FP32 copy of weights/parameters to be saved and used in the optimizer. To enable BF16 in GPU-compatible mode without stochastic rounding enabled, use the ``torch.nn.Module.to`` method to cast model floating-point parameters and buffers to data-type bfloat16 as follows without setting ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``:

.. code:: python

    # model is created
    model.to(torch.bfloat16)

In the initializer of the optimizer, for example AdamW, you can add code like the following code snippet to make a FP32 copy of weights:

.. code:: python

        # keep a copy of weights in highprec
        self.param_groups_highprec = []
        for group in self.param_groups:
            params = group['params']
            param_groups_highprec = [p.data.float() for p in params]
            self.param_groups_highprec.append({'params': param_groups_highprec})

From then, you can use the usual gradients but updating the FP32 copy of weights instead:

.. code:: python

        for group, group_highprec in zip(self.param_groups, self.param_groups_highprec):
            for p, p_highprec in zip(group['params'], group_highprec['params']):
                # convert gradients to FP32 before computing exponential average
                grad = p.grad.data.float()

                # compute the exponential average and denominator using grad
                ...

                # Update FP32 copy of weights
                p_highprec.data.addcdiv_(exponential_avg, denominator, value=-step_size)


In the :ref:`PyTorch Neuron BERT Pretraining Tutorial (Data-Parallel) <hf-bert-pretraining-tutorial>`, this mode can be enabled by pasing ``--optimizer=AdamW_FP32ParamsCopy`` option to ``dp_bert_large_hf_pretrain_hdf5.py`` and setting ``NEURON_RT_STOCHASTIC_ROUNDING_EN=0`` (or leave it unset).

BF16 automatic mixed precision using PyTorch Autocast
-----------------------------------------------------

By default, the compiler automatically casts internal FP32 operations to
BF16. You can disable this and allow PyTorch's BF16 automatic mixed precision function (``torch.autocast``) to
do the casting of certain operations to operate in BF16.

To enable PyTorch's BF16 mixed-precision, first turn off the Neuron
compiler auto-cast:

.. code:: python

   os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"

Next, per recommendation from official PyTorch `torch.autocast documentation <https://pytorch.org/docs/stable/amp.html#autocasting>`__, place only
the forward-pass of the training step in the ``torch.autocast`` scope with ``xla`` device type:

.. code:: python

   with torch.autocast(dtype=torch.bfloat16, device_type='xla'):
       # forward pass

The device type is XLA because we are using PyTorch-XLA's autocast backend. The PyTorch-XLA `autocast mode source code <https://github.com/pytorch/xla/blob/master/torch_xla/csrc/autocast_mode.cpp>`_ lists which operations are casted to lower precision BF16 ("lower precision fp cast policy" section), which are maintained in FP32 ("fp32 cast policy"), and which are promoted to the widest input types ("promote" section).

.. note::

   If an operation is not part of any policy in `autocast mode source code <https://github.com/pytorch/xla/blob/master/torch_xla/csrc/autocast_mode.cpp>`_, the data type of the inputs will be used for the computation of the operation.


Example showing the original training code snippet:

.. code:: python

   def train_loop_fn(train_loader):
       for i, data in enumerate(train_loader):
           inputs = data[0]
           labels = data[3]
           outputs = model(inputs, labels=labels)
           loss = outputs.loss/ flags.grad_acc_steps
           loss.backward()
           optimizer.step()
           xm.mark_step()

The following shows the training loop modified to use BF16 autocast:

.. code:: python

   os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"

   def train_loop_fn(train_loader):
       for i, data in enumerate(train_loader):
           torch.cuda.is_bf16_supported = lambda: True
           with torch.autocast(dtype=torch.bfloat16, device_type='xla'):
               inputs = data[0]
               labels = data[3]
               outputs = model(inputs, labels=labels)
           loss = outputs.loss/ flags.grad_acc_steps
           loss.backward()
           optimizer.step()
           xm.mark_step()

For a full example of BF16 mixed-precision, see :ref:`PyTorch Neuron BERT Pretraining Tutorial (Data-Parallel) <hf-bert-pretraining-tutorial>`.

See official PyTorch documentation for more details about
`torch.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`__
.
