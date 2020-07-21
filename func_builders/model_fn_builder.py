#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 



import tensorflow as tf
from utils import util
from utils.radam import RAdam


def model_fn_builder(config, model_sign="mention_proposal"):

    def mention_proposal_model_fn(features, labels, mode, params): 
        """The `model_fn` for TPUEstimator."""
        input_ids = features["flattened_input_ids"]
        input_mask = features["flattened_input_mask"]
        text_len = features["text_len"]
        speaker_ids = features["speaker_ids"]
        gold_starts = features["span_starts"]
        gold_ends = features["span_ends"]
        cluster_ids = features["cluster_ids"]
        sentence_map = features["sentence_map"] 
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = util.get_model(config, model_sign="mention_proposal")

        if config.use_tpu:
            def tpu_scaffold():
                return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold
        else:
            scaffold_fn = None 

        if mode == tf.estimator.ModeKeys.TRAIN: 
            tf.logging.info("****************************** tf.estimator.ModeKeys.TRAIN ******************************")
            tf.logging.info("********* Features *********")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            instance = (input_ids, input_mask, sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids)
            total_loss, start_scores, end_scores, span_scores = model.get_mention_proposal_and_loss(instance, is_training)
            gold_start_sequence_labels, gold_end_sequence_labels, gold_span_sequence_labels = model.get_gold_mention_sequence_labels_from_pad_index(gold_starts, gold_ends, text_len)

            if config.use_tpu:
                optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step()) 
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                optimizer = RAdam(learning_rate=config.learning_rate, epsilon=1e-8, beta1=0.9, beta2=0.999)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step())
        
                train_logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=1)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn,
                    training_hooks=[train_logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL: 
            tf.logging.info("****************************** tf.estimator.ModeKeys.EVAL ******************************")
            
            instance = (input_ids, input_mask, sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids)
            total_loss, start_scores, end_scores, span_scores = model.get_mention_proposal_and_loss(instance, is_training)
            total_loss, start_scores, end_scores, span_scores = model.get_mention_proposal_and_loss(instance, is_training)
            gold_start_sequence_labels, gold_end_sequence_labels, gold_span_sequence_labels = model.get_gold_mention_sequence_labels_from_pad_index(gold_starts, gold_ends, text_len)

            def metric_fn(start_scores, end_scores, span_scores, gold_span_label):
                start_scores = tf.reshape(start_scores, [-1, config.window_size])
                end_scores = tf.reshape(end_scores, [-1, config.window_size])
                start_scores = tf.tile(tf.expand_dims(start_scores, 2), [1, 1, config.window_size])
                end_scores = tf.tile(tf.expand_dims(end_scores, 2), [1, 1, config.window_size])
                sce_span_scores = (start_scores + end_scores + span_scores)/ 3
                pred_span_label = tf.cast(tf.reshape(tf.math.greater_equal(sce_span_scores, config.mention_threshold), [-1]), tf.bool)

                gold_span_label = tf.cast(tf.reshape(gold_span_sequence_labels, [-1]), tf.bool)

                return {"precision": tf.compat.v1.metrics.precision(gold_span_label, pred_span_label), 
                        "recall": tf.compat.v1.metrics.recall(gold_span_label, pred_span_label)}

            eval_metrics = (metric_fn, [start_scores, end_scores, span_scores])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            tf.logging.info("****************************** tf.estimator.ModeKeys.PREDICT ******************************")
            
            instance = (input_ids, input_mask, sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids)
            total_loss, start_scores, end_scores, span_scores = model.get_mention_proposal_and_loss(instance, is_training)
            gold_start_sequence_labels, gold_end_sequence_labels, gold_span_sequence_labels = model.get_gold_mention_sequence_labels_from_pad_index(gold_starts, gold_ends, text_len)
            predictions = {
                    "total_loss": total_loss,
                    "start_scores": start_scores,
                    "start_gold": gold_starts,
                    "end_gold": gold_ends,
                    "end_scores": end_scores, 
                    "span_scores": span_scores
            }            
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Please check the the mode ! ")
        
        return output_spec


    def corefqa_model_fn(features, labels, mode, params):

        """The `model_fn` for TPUEstimator."""
        input_ids = features["flattened_input_ids"]
        input_mask = features["flattened_input_mask"]
        text_len = features["text_len"]
        speaker_ids = features["speaker_ids"]
        gold_starts = features["span_starts"]
        gold_ends = features["span_ends"]
        cluster_ids = features["cluster_ids"]
        sentence_map = features["sentence_map"] 
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = util.get_model(config, model_sign="corefqa")
    
        if config.use_tpu:
            tf.logging.info("****************************** Training on TPU ******************************")
            def tpu_scaffold():
                return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold
        else:
            scaffold_fn = None 


        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("****************************** tf.estimator.ModeKeys.TRAIN ******************************")
            tf.logging.info("********* Features *********")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            instance = (input_ids, input_mask, sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids)
            total_loss, (topk_mention_start_indices, topk_mention_end_indices), (forward_topc_mention_start_indices, forward_topc_mention_end_indices), top_mention_span_linking_scores  = model.get_coreference_resolution_and_loss(instance, is_training, use_tpu=config.use_tpu)

            if config.use_tpu:
                optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step()) 
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                optimizer = RAdam(learning_rate=config.learning_rate, epsilon=1e-8, beta1=0.9, beta2=0.999)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step())

                training_logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=1)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn, 
                    training_hooks=[training_logging_hook])


        elif mode == tf.estimator.ModeKeys.EVAL: 
            tf.logging.info("****************************** tf.estimator.ModeKeys.EVAL ******************************")
            tf.logging.info("@@@@@ MERELY support tf.estimator.ModeKeys.PREDICT ! @@@@@")
            tf.logging.info("@@@@@ YOU can EVAL your checkpoints after the training process. @@@@@")  
            tf.logging.info("****************************** tf.estimator.ModeKeys.EVAL ******************************")
        
        elif mode == tf.estimator.ModeKeys.PREDICT :
            tf.logging.info("****************************** tf.estimator.ModeKeys.PREDICT ******************************")

            instance = (input_ids, input_mask, sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids)
            total_loss, (topk_mention_start_indices, topk_mention_end_indices), (forward_topc_mention_start_indices, forward_topc_mention_end_indices), top_mention_span_linking_scores  = model.get_coreference_resolution_and_loss(instance, True, use_tpu=config.use_tpu)

            top_antecedent = tf.math.argmax(top_mention_span_linking_scores, axis=-1)
            predictions = {
                        "total_loss": total_loss, 
                        "topk_span_starts": topk_mention_start_indices,
                        "topk_span_ends": topk_mention_end_indices, 
                        "top_antecedent_scores": top_mention_span_linking_scores,
                        "top_antecedent": top_antecedent,
                        "cluster_ids" : cluster_ids, 
                        "gold_starts": gold_starts, 
                        "gold_ends": gold_ends}   

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, 
                predictions=predictions, 
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Please check the the mode ! ")
        return output_spec


    if model_sign == "mention_proposal":
        return mention_proposal_model_fn
    elif model_sign == "corefqa":
        return corefqa_model_fn
    else:
        raise ValueError("Please check the model sign! Only support [mention_proposal] and [corefqa] .")







