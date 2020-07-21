#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


import tensorflow as tf


def file_based_input_fn_builder(input_file, num_window=None, window_size=None, max_num_mention=None, is_training=False, drop_remainder=True):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
            'sentence_map': tf.FixedLenFeature([num_window * window_size], tf.int64),
            'text_len': tf.FixedLenFeature([num_window], tf.int64),
            'subtoken_map': tf.FixedLenFeature([num_window *  window_size], tf.int64),
            'speaker_ids': tf.FixedLenFeature([num_window * window_size], tf.int64),
            'flattened_input_ids': tf.FixedLenFeature([num_window * window_size], tf.int64),
            'flattened_input_mask': tf.FixedLenFeature([num_window * window_size], tf.int64),
            'span_starts': tf.FixedLenFeature([max_num_mention], tf.int64),
            'span_ends': tf.FixedLenFeature([max_num_mention], tf.int64), 
            'cluster_ids': tf.FixedLenFeature([max_num_mention], tf.int64),
            }


    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t 
        return example


    def input_fn_from_tfrecord(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat() 
            d = d.shuffle(buffer_size=100)
    
        d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

        return d
        
    return input_fn_from_tfrecord 