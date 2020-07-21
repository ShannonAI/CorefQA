#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



import tensorflow as tf
from tensorflow.python import (
        ops, math_ops, state_ops, control_flow_ops, resource_variable_ops)
from tensorflow.python.training.optimizer import Optimizer

__all__ = ['RAdam']


class RAdam(Optimizer):
    """Rectified Adam (RAdam) optimizer.
    According to the paper
    [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 amsgrad=False,
                 use_locking=False,
                 name='RAdam'):
        r"""Construct a new Rectified Adam optimizer.
        Args:
            learning_rate: A Tensor or a floating point value.    The learning rate.
            beta1: A float value or a constant float tensor. The exponential decay
                rate for the 1st moment estimates.
            beta2: A float value or a constant float tensor. The exponential decay
                rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just before
                Section 2.1), not the epsilon in Algorithm 1 of the paper.
            amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
                the paper "On the Convergence of Adam and beyond".
            use_locking: If `True` use locks for update operations.
            name: Optional name for the operations created when applying gradients.
                Defaults to "Adam".    @compatibility(eager) When eager execution is
                enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be
                a callable that takes no arguments and returns the actual value to use.
                This can be useful for changing these values across different
                invocations of optimizer functions. @end_compatibility
        """

        super(RAdam, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._amsgrad = amsgrad

    def _get_beta_accumulators(self):
        with ops.init_scope():
            graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph),
                    )

    def _get_niter(self):
        with ops.init_scope():
            graph = ops.get_default_graph()
            return self._get_non_slot_variable("niter", graph=graph)

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=1, name="niter", colocate_with=first_var)
        for var in var_list:
            self._zeros_slot(var, 'm', self._name)
            self._zeros_slot(var, 'v', self._name)
        if self._amsgrad:
            for var in var_list:
                self._zeros_slot(var, 'vhat', self._name)

    def _prepare(self):
        learning_rate = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)

        self._lr_t = ops.convert_to_tensor(learning_rate, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _apply_dense_shared(self, grad, var):
        var_dtype = var.dtype.base_dtype
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var_dtype)
        beta2_power = math_ops.cast(beta2_power, var_dtype)
        niter = self._get_niter()
        niter = math_ops.cast(niter, var_dtype)
        lr_t = math_ops.cast(self._lr_t, var_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var_dtype)

        sma_inf = 2.0 / (1.0 - beta2_t) - 1.0
        sma_t = sma_inf - 2.0 * niter * beta2_power / (1.0 - beta2_power)

        m = self.get_slot(var, 'm')
        m_t = state_ops.assign(m,
                               beta1_t * m + (1.0 - beta1_t) * grad,
                               use_locking=self._use_locking)
        m_corr_t = m_t / (1.0 - beta1_power)

        v = self.get_slot(var, 'v')
        v_t = state_ops.assign(v,
                               beta2_t * v + (1.0 - beta2_t) * math_ops.square(grad),
                               use_locking=self._use_locking)

        if self._amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            v_corr_t = math_ops.sqrt(vhat_t / (1.0 - beta2_power) + epsilon_t)
        else:
            v_corr_t = math_ops.sqrt(v_t / (1.0 - beta2_power) + epsilon_t)

        r_t = math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                            (sma_t - 2.0) / (sma_inf - 2.0) *
                            sma_inf / sma_t)

        var_t = tf.where(sma_t > 5.0, r_t * m_corr_t / v_corr_t, m_corr_t)

        var_update = state_ops.assign_sub(var,
                                          lr_t * var_t,
                                          use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self._amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _apply_dense(self, grad, var):
        return self._apply_dense_shared(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self._apply_dense_shared(grad, var.handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        var_dtype = var.dtype.base_dtype
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var_dtype)
        beta2_power = math_ops.cast(beta2_power, var_dtype)
        niter = self._get_niter()
        niter = math_ops.cast(niter, var_dtype)
        lr_t = math_ops.cast(self._lr_t, var_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var_dtype)

        sma_inf = 2.0 / (1.0 - beta2_t) - 1.0
        sma_t = sma_inf - 2.0 * niter * beta2_power / (1.0 - beta2_power)

        m = self.get_slot(var, 'm')
        m_t = state_ops.assign(m, beta1_t * m, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, grad * (1 - beta1_t))
        m_corr_t = m_t / (1.0 - beta1_power)

        v = self.get_slot(var, 'v')
        v_t = state_ops.assign(v, beta2_t * v, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, (1.0 - beta2_t) * math_ops.square(grad))

        if self._amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            v_corr_t = math_ops.sqrt(vhat_t / (1.0 - beta2_power) + epsilon_t)
        else:
            v_corr_t = math_ops.sqrt(v_t / (1.0 - beta2_power) + epsilon_t)

        r_t = math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                            (sma_t - 2.0) / (sma_inf - 2.0) *
                            sma_inf / sma_t)

        var_t = tf.where(sma_t > 5.0, r_t * m_corr_t / v_corr_t, m_corr_t)

        var_update = state_ops.assign_sub(var,
                                          lr_t * var_t,
                                          use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self._amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x,
                i,
                v,
                use_locking=self._use_locking))

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add)

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            niter = self._get_niter()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
                update_niter = niter.assign(
                    niter + 1, use_locking=self._use_locking)
        return control_flow_ops.group(
            *update_ops + [update_beta1, update_beta2, update_niter], name=name_scope)
