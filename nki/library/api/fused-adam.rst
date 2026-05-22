.. meta::
    :description: Adam optimizer kernel with L2 regularization.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.optimizer

Fused Adam Kernel API Reference
===============================

Adam optimizer kernel with L2 regularization.

This kernel implements the Adam optimizer with L2 weight regularization. For decoupled weight decay (AdamW), use ``adamw_kernel`` instead.

Background
-----------

The ``adam_kernel`` and ``adamw_kernel`` implement fused Adam and AdamW optimizers on NeuronCore, processing parameter updates entirely on-device to avoid host-device round trips.

API Reference
--------------

**Source code for this kernel API can be found at**: `fused_adam.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/optimizer/fused_adam.py>`_

adam_kernel
^^^^^^^^^^^

.. py:function:: adam_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, max_exp_avg_sq_ptr, step_size_ptr, inv_bc2_sqrt_ptr, wd_factor_ptr, numel, beta1, beta2, eps, amsgrad = False)

   Adam optimizer kernel with L2 regularization.

   :param param_ptr: [N], Parameter tensor on HBM
   :param grad_ptr: [N], Gradient tensor on HBM
   :param exp_avg_ptr: [N], First moment estimate tensor on HBM
   :param exp_avg_sq_ptr: [N], Second moment estimate tensor on HBM
   :param max_exp_avg_sq_ptr: [N], Max second moment tensor on HBM (AMSGrad only)
   :param step_size_ptr: [P_MAX, 1], Bias-corrected step size on HBM
   :param inv_bc2_sqrt_ptr: [P_MAX, 1], Inverse bias correction sqrt on HBM
   :param wd_factor_ptr: [P_MAX, 1], Weight decay coefficient (lambda) on HBM
   :param numel: Number of elements in the parameter tensor
   :param beta1: First moment decay factor (typically 0.9)
   :param beta2: Second moment decay factor (typically 0.999)
   :param eps: Epsilon for numerical stability (typically 1e-8)
   :param amsgrad: If True, use AMSGrad variant (default: False)
   :return: [N], Updated parameter tensor on HBM
   :rtype: ``nl.ndarray``
   :return: [N], Updated first moment tensor on HBM
   :rtype: ``nl.ndarray``
   :return: [N], Updated second moment tensor on HBM
   :rtype: ``nl.ndarray``
   :return: [N], Updated max second moment (if amsgrad=True)
   :rtype: ``nl.ndarray``

   **Notes**:

   * Uses L2 regularization: grad = grad + lambda * param
   * For decoupled weight decay (AdamW), use adamw_kernel instead

   **Dimensions**:

   * N: Number of elements in the parameter tensor (numel)

adamw_kernel
^^^^^^^^^^^^

.. py:function:: adamw_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, max_exp_avg_sq_ptr, step_size_ptr, inv_bc2_sqrt_ptr, wd_factor_ptr, numel, beta1, beta2, eps, amsgrad = False)

   AdamW optimizer kernel with decoupled weight decay.

   :param param_ptr: [N], Parameter tensor on HBM
   :param grad_ptr: [N], Gradient tensor on HBM
   :param exp_avg_ptr: [N], First moment estimate tensor on HBM
   :param exp_avg_sq_ptr: [N], Second moment estimate tensor on HBM
   :param max_exp_avg_sq_ptr: [N], Max second moment tensor on HBM (AMSGrad only)
   :param step_size_ptr: [P_MAX, 1], Bias-corrected step size on HBM
   :param inv_bc2_sqrt_ptr: [P_MAX, 1], Inverse bias correction sqrt on HBM
   :param wd_factor_ptr: [P_MAX, 1], Weight decay factor (1 - lr*lambda) on HBM
   :param numel: Number of elements in the parameter tensor
   :param beta1: First moment decay factor (typically 0.9)
   :param beta2: Second moment decay factor (typically 0.999)
   :param eps: Epsilon for numerical stability (typically 1e-8)
   :param amsgrad: If True, use AMSGrad variant (default: False)
   :return: [N], Updated parameter tensor on HBM
   :rtype: ``nl.ndarray``
   :return: [N], Updated first moment tensor on HBM
   :rtype: ``nl.ndarray``
   :return: [N], Updated second moment tensor on HBM
   :rtype: ``nl.ndarray``
   :return: [N], Updated max second moment (if amsgrad=True)
   :rtype: ``nl.ndarray``

   **Notes**:

   * Uses decoupled weight decay: param = param * wd_factor - update
   * For L2 regularization (Adam), use adam_kernel instead

   **Dimensions**:

   * N: Number of elements in the parameter tensor (numel)

