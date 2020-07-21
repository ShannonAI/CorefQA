#!/usr/bin/env python3
# -*- coding: utf-8 -*- 



# desc:
# construct labels 


import tensorflow as tf 



if __name__ == "__main__":
    sess = tf.compat.v1.InteractiveSession()
    gold_starts = tf.constant([1, 2, 3, 4])
    gold_ends = tf.constant([2, 3, 4, 5])
    num_word = 10 
    gold_mention_sparse_label = tf.stack([gold_starts, gold_ends], axis=1)
    gold_mention_sparse_label.eval()
    gold_span_value = tf.reshape(tf.ones_like(gold_starts, tf.int32), [-1])
    gold_span_shape = tf.constant([num_word, num_word])
    gold_span_label = tf.cast(tf.scatter_nd(gold_mention_sparse_label, gold_span_value, gold_span_shape), tf.int32)
    gold_span_label.eval()

    candidate_start = tf.constant([1, 4, 5])
    candidate_end = tf.constant([2, 5, 5])
    candidate_span = tf.stack([candidate_start, candidate_end], axis=1)

    gold_span_label = tf.gather_nd(gold_span_label, tf.expand_dims(candidate_span, 1))
    gold_span_label.eval()
