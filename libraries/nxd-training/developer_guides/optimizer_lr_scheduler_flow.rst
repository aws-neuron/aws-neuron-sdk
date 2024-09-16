.. _nxdt_developer_flow_register_optimizer_lr_scheduler:

Registering an optimizer and LR scheduler
=========================================

A new optimizer or LR scheduler can be registered with the framework and enabled from the config.

.. contents:: Table of contents
   :local:
   :depth: 2


Setting up the optimizer
------------------------

One can write their own optimizer class. One such example is the
`AdamW_FP32OptimParams <https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/utils/adamw_fp32_optim_params.py>`_.

The inputs to the optimizer can be exposed in the config YAML file. To do this, we need to create a ``Params`` class
as shown below:

.. code-block:: python

    from dataclasses import dataclass
    from typing import Any, Dict, Optional, Tuple

    from omegaconf import MISSING

    @dataclass
    class OptimizerParams:
        """
        All the params listed below can be configured from the YAML file
        """

        lr: Optional[float] = MISSING
        betas: Tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-08
        weight_decay: float = 0
        amsgrad: bool = False


Once we create the optimizer and the optimizer params class, we can now register the optimizer with the
framework using the following code:

.. code-block:: python

    from nemo.core.optim import register_optimizer

    # `adamw_fp32OptState` would be the name in the optim config of the YAML file.
    register_optimizer("adamw_fp32OptState", AdamW_FP32OptimParams, OptimizerParams)

This registration can be done inside the ``training.py`` file which resides in ``examples`` folder.

Once the registration is done, we can now expose the ``OptimizerParams`` under ``optim`` config of the
YAML file.


Setting up the LR scheduler
---------------------------

One can write their own LR scheduler and register with the framework. One such example of LR scheduler is
shown below:

.. code-block:: python

    from functools import partial

    from torch.optim.lr_scheduler import LambdaLR
    from transformers.optimization import _get_linear_schedule_with_warmup_lr_lambda


    class LinearAnnealingWithWarmUp(LambdaLR):
        def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
            lr_lambda = partial(
                _get_linear_schedule_with_warmup_lr_lambda,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
            super().__init__(optimizer, lr_lambda, last_epoch)


Once we build this LR scheduler, we can expose the arguments to the config YAML file. Before that,
we need to write up a ``LRSchedulerParams`` class. Here is an example for the same:

.. code-block:: python

    from nemo.core.config.schedulers import SchedulerParams

    class LinearAnnealingWithWarmupParams(SchedulerParams):
        warmup_steps: int = 0
        max_steps: int = 0


Once the LR scheduler and the ``SchedulerParams`` class are set, we can now register the scheduler
with the framework as below:

.. code-block:: python

    from nemo.core.optim.lr_scheduler import register_scheduler


    # Here, `LinearAnnealingWithWarmUp` is the name of the scheduler we would use in the config YAML file
    register_scheduler("LinearAnnealingWithWarmUp", LinearAnnealingWithWarmUp, LinearAnnealingWithWarmupParams)


This registration can be done inside the ``training.py`` file which resides under ``examples`` folder.

Once the registration is done, we can now expose the ``LinearAnnealingWithWarmupParams`` under ``sched`` config
of the YAML file.
