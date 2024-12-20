.. _migration_from_xla_downcast_bf16:

Migration From ``XLA_USE_BF16``/``XLA_DOWNCAST_BF16``
=====================================================

Introduction
------------

The environmental variables ``XLA_USE_BF16`` and ``XLA_DOWNCAST_BF16`` were created to provide an easy cast-to-bf16 option before automatic mixed-precision or ``model.to(torch.bfloat16)`` as available in Torch-XLA. Now that both automatic mixed precision and ``model.to(torch.bfloat16)`` are available in Torch-XLA,  ``XLA_USE_BF16`` and ``XLA_DOWNCAST_BF16`` are redundant and can be replaced with these options as a more familiar experience as on other platforms such as CPUs and GPUs. (They are deprecated in Torch-XLA 2.5 as warnings, and will be removed in a future release).

This change can best be made to scripts running with Torch-XLA 2.1 and 2.5.

Full BF16 with stochastic rounding enabled
------------------------------------------

On Neuron, when the environmental variable ``XLA_USE_BF16`` or ``XLA_DOWNCAST_BF16`` is set, stochastic rounding is enabled by default. If they are not used, then stochastic rounding is off unless ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``.

To replace ``XLA_USE_BF16`` or ``XLA_DOWNCAST_BF16`` with stochastic rounding on Neuron, set ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1`` and use the “to” function to move the model to data-type bfloat16 as follows:


.. code:: python

    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"

    # model is created
    model.to(torch.bfloat16)

If the loss is to be kept in FP32, initialize it with ``dtype=torch.float`` as follows:

.. code:: python

    running_loss = torch.zeros(1, dtype=torch.float).to(device)

Similarly, if the optimizer states are to be kept in FP32, convert the gradients to FP32 before optimizer computations:

.. code:: python

    grad = p.grad.data.float()

For a full example, please see the updated BERT pretraining tutorial.

BF16 in GPU-compatible mode without stochastic rounding enabled
---------------------------------------------------------------

To enable BF16 in GPU-compatible mode without stochastic rounding enabled, use the “to” function to move the model to data-type bfloat16 as follows without setting ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``:

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


Automatic Mixed Precision
-------------------------

See the existing `Automatic Mixed Precision example <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/bert.html#phase-1-bert-large-pretraining-with-pytorch-autocast-amp-and-stochastic-rounding>`_.
