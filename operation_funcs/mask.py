#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: operations for running experiments on TPU device 
# https://github.com/tensorflow/tpu/blob/5a5d62339b08eb4366515c05e405104ab79297a4/models/official/detection/utils/object_detection/box_list_ops.py


import tensorflow as tf 
from operation_funcs import ops 

VERY_LARGE_NEGATIVE_VALUE = -1e12


def exp_mask(logits, mask, mask_is_length=True):
    """Exponential mask for logits.
    Logits cannot be masked with 0 (i.e. multiplying boolean mask)
    because expnentiating 0 becomes 1. `exp_mask` adds very large negative value
    to `False` portion of `mask` so that the portion is effectively ignored
    when exponentiated, e.g. softmaxed.
    Args:
        logits: Arbitrary-rank logits tensor to be masked.
        mask: `boolean` type mask tensor.
            Could be same shape as logits (`mask_is_length=False`)
            or could be length tensor of the logits (`mask_is_length=True`).
    mask_is_length: `bool` value. whether `mask` is boolean mask.
    Returns:
    Masked logits with the same shape of `logits`.
    """
    if mask_is_length:
        mask = tf.sequence_mask(mask, maxlen=tf.shape(logits)[-1])

    return logits + (1.0 - tf.cast(mask, 'float')) * VERY_LARGE_NEGATIVE_VALUE


def boolean_mask(itemlist, indicator, fields=None, scope=None,
                 use_static_shapes=False, indicator_sum=None, use_tpu=True):
  """Select boxes from BoxList according to indicator and return new BoxList.
  `boolean_mask` returns the subset of boxes that are marked as "True" by the
  indicator tensor. By default, `boolean_mask` returns boxes corresponding to
  the input index list, as well as all additional fields stored in the boxlist
  (indexing into the first dimension).  However one can optionally only draw
  from a subset of fields.
  Args:
    boxlist: BoxList holding N boxes
    indicator: a rank-1 boolean tensor
    fields: (optional) list of fields to also gather from.  If None (default),
      all fields are gathered from.  Pass an empty fields list to only gather
      the box coordinates.
    scope: name scope.
    use_static_shapes: Whether to use an implementation with static shape
      gurantees.
    indicator_sum: An integer containing the sum of `indicator` vector. Only
      required if `use_static_shape` is True.
  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
      specified by indicator
  Raises:
    ValueError: if `indicator` is not a rank-1 boolean tensor.
  """
  ###### if not use_tpu:
  ######   return tf.boolean_mask(itemlist, indicator)
  with tf.name_scope(scope, 'BooleanMask'):
      selected_positions = tf.cast(indicator, dtype=tf.float32)
      indicator_sum = tf.cast(tf.math.reduce_sum(tf.cast(indicator, dtype=tf.int32)), tf.int32)
      indexed_positions = tf.cast(
          tf.multiply(
              tf.cumsum(selected_positions), selected_positions),
          dtype=tf.int32)
      one_hot_selector = tf.one_hot(
          indexed_positions - 1, indicator_sum, dtype=tf.float32)
      sampled_indices = tf.cast(
          tf.tensordot(
              tf.cast(tf.range(tf.shape(indicator)[0]), dtype=tf.float32),
              one_hot_selector,
              axes=[0, 0]),
          dtype=tf.int32)
      return tf.gather(itemlist, sampled_indices)
      # return gather(boxlist, sampled_indices, use_static_shapes=True)
#     else:
#      return tf.boolean_mask(itemlist, indicator)


def gather(boxlist, indices, fields=None, scope=None, use_static_shapes=False, use_tpu=True):
  """Gather boxes from BoxList according to indices and return new BoxList.
  By default, `gather` returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.
  Args:
    boxlist: BoxList holding N boxes
    indices: a rank-1 tensor of type int32 / int64
    fields: (optional) list of fields to also gather from.  If None (default),
      all fields are gathered from.  Pass an empty fields list to only gather
      the box coordinates.
    scope: name scope.
    use_static_shapes: Whether to use an implementation with static shape
      gurantees.
  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
    specified by indices
  Raises:
    ValueError: if specified field is not contained in boxlist or if the
      indices are not of type int32
  """
  with tf.name_scope(scope, 'Gather'):
    if len(indices.shape.as_list()) != 1:
      raise ValueError('indices should have rank 1')
    if indices.dtype != tf.int32 and indices.dtype != tf.int64:
      raise ValueError('indices should be an int32 / int64 tensor')
    gather_op = tf.gather
    if use_static_shapes or use_tpu:
      gather_op = ops.matmul_gather_on_zeroth_axis
    subboxlist = gather_op(boxlist.get(), indices)
    return subboxlist


if __name__ == "__main__":
  pass 





