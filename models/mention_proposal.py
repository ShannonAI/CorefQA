#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# the mention proposal model for pre-training the Span-BERT model. 


import os
import sys 

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import tensorflow as tf
from bert import modeling



class MentionProposalModel(object):
    def __init__(self, config):
        self.config = config 
        self.bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
        self.bert_config.hidden_dropout_prob = config.dropout_rate

    def get_mention_proposal_and_loss(self, instance, is_training, use_tpu=False):
        """
        Desc:
            forward function for training mention proposal module. 
        Args:
            instance: a tuple of train/dev/test data instance. 
                e.g., (flat_input_ids, flat_doc_overlap_input_mask, flat_sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids)
            is_training: True/False is in the training process. 
        """
        self.use_tpu = use_tpu 
        self.dropout = self.get_dropout(self.config.dropout_rate, is_training)

        flat_input_ids, flat_doc_overlap_input_mask, flat_sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids = instance
        # flat_input_ids: (num_window, window_size)
        # flat_doc_overlap_input_mask: (num_window, window_size)
        # flat_sentence_map: (num_window, window_size)
        # text_len: dynamic length and is padded to fix length
        # gold_start: (max_num_mention), mention start index in the original (NON-OVERLAP) document. Pad with -1 to the fix length max_num_mention.
        # gold_end: (max_num_mention), mention end index in the original (NON-OVERLAP) document. Pad with -1 to the fix length max_num_mention.
        # cluster_ids/speaker_ids is not used in the mention proposal model.

        flat_input_ids = tf.math.maximum(flat_input_ids, tf.zeros_like(flat_input_ids, tf.int32)) # (num_window * window_size)
        
        flat_doc_overlap_input_mask = tf.where(tf.math.greater_equal(flat_doc_overlap_input_mask, 0), 
            x=tf.ones_like(flat_doc_overlap_input_mask, tf.int32), y=tf.zeros_like(flat_doc_overlap_input_mask, tf.int32)) # (num_window * window_size)
        # flat_doc_overlap_input_mask = tf.math.maximum(flat_doc_overlap_input_mask, tf.zeros_like(flat_doc_overlap_input_mask, tf.int32))
        flat_sentence_map = tf.math.maximum(flat_sentence_map, tf.zeros_like(flat_sentence_map, tf.int32)) # (num_window * window_size)
        
        gold_start_end_mask = tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts, tf.int32)), tf.bool) # (max_num_mention)
        gold_start_index_labels = self.boolean_mask_1d(gold_starts, gold_start_end_mask, name_scope="gold_starts", use_tpu=self.use_tpu) # (num_of_mention)
        gold_end_index_labels = self.boolean_mask_1d(gold_ends, gold_start_end_mask, name_scope="gold_ends", use_tpu=self.use_tpu) # (num_of_mention)

        text_len = tf.math.maximum(text_len, tf.zeros_like(text_len, tf.int32)) # (num_of_non_empty_window)
        num_subtoken_in_doc = tf.math.reduce_sum(text_len) # the value should be num_subtoken_in_doc 

        input_ids = tf.reshape(flat_input_ids, [-1, self.config.window_size]) # (num_window, window_size)
        input_mask = tf.ones_like(input_ids, tf.int32) # (num_window, window_size)

        model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=input_ids, input_mask=input_mask, 
            use_one_hot_embeddings=False, scope='bert')

        doc_overlap_window_embs = model.get_sequence_output() # (num_window, window_size, hidden_size)
        doc_overlap_input_mask = tf.reshape(flat_doc_overlap_input_mask, [self.config.num_window, self.config.window_size]) # (num_window, window_size)

        doc_flat_embs = self.transform_overlap_windows_to_original_doc(doc_overlap_window_embs, doc_overlap_input_mask) 
        doc_flat_embs = tf.reshape(doc_flat_embs, [-1, self.config.hidden_size]) # (num_subtoken_in_doc, hidden_size)

        expand_start_embs = tf.tile(tf.expand_dims(doc_flat_embs, 1), [1, num_subtoken_in_doc, 1]) # (num_subtoken_in_doc, num_subtoken_in_doc, hidden_size)
        expand_end_embs = tf.tile(tf.expand_dims(doc_flat_embs, 0), [num_subtoken_in_doc, 1, 1]) # (num_subtoken_in_doc, num_subtoken_in_doc, hidden_size)
        expand_mention_span_embs = tf.concat([expand_start_embs, expand_end_embs], axis=-1) # (num_subtoken_in_doc, num_subtoken_in_doc, 2*hidden_size)
        expand_mention_span_embs = tf.reshape(expand_mention_span_embs, [-1, self.config.hidden_size*2])
        span_sequence_logits = self.ffnn(expand_mention_span_embs, self.config.hidden_size*2, 1, dropout=self.dropout, name_scope="mention_span") # (num_subtoken_in_doc * num_subtoken_in_doc)

        if self.config.start_end_share:
            start_end_sequence_logits = self.ffnn(doc_flat_embs, self.config.hidden_size, 2, dropout=self.dropout, name_scope="mention_start_end") # (num_subtoken_in_doc, 2)
            start_sequence_logits, end_sequence_logits = tf.split(start_end_sequence_logits, axis=1)
            # start_sequence_logits -> (num_subtoken_in_doc, 1)
            # end_sequence_logits -> (num_subtoken_in_doc, 1)
        else:
            start_sequence_logits = self.ffnn(doc_flat_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="mention_start") # (num_subtoken_in_doc)
            end_sequence_logits = self.ffnn(doc_flat_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="mention_end") # (num_subtoken_in_doc)

        gold_start_sequence_labels = self.scatter_gold_index_to_label_sequence(gold_start_index_labels, num_subtoken_in_doc) # (num_subtoken_in_doc)
        gold_end_sequence_labels = self.scatter_gold_index_to_label_sequence(gold_end_index_labels, num_subtoken_in_doc) # (num_subtoken_in_doc)

        start_loss, start_sequence_probabilities = self.compute_score_and_loss(start_sequence_logits, gold_start_sequence_labels)
        end_loss, end_sequence_probabilities = self.compute_score_and_loss(end_sequence_logits, gold_end_sequence_labels)
        # *_loss -> a scalar 
        # *_sequence_scores -> (num_subtoken_in_doc)

        gold_span_sequence_labels = self.scatter_span_sequence_labels(gold_start_index_labels, gold_end_index_labels, num_subtoken_in_doc) # (num_subtoken_in_doc * num_subtoken_in_doc)
        span_loss, span_sequence_probabilities = self.compute_score_and_loss(span_sequence_logits, gold_span_sequence_labels)
        # span_loss -> a scalar 
        # span_sequence_probabilities -> (num_subtoken_in_doc * num_subtoken_in_doc)
        
        total_loss = self.config.loss_start_ratio * start_loss + self.config.loss_end_ratio * end_loss + self.config.loss_span_ratio * span_loss 
        return total_loss, start_sequence_probabilities, end_sequence_probabilities, span_sequence_probabilities


    def get_gold_mention_sequence_labels_from_pad_index(self, pad_gold_start_index_labels, pad_gold_end_index_labels, pad_text_len):
        """
        Desc:
            the original gold labels is padded to the fixed length and only contains the position index of gold mentions. 
            return the gold sequence of labels for evaluation. 
        Args:
            pad_gold_start_index_labels: a tf.int32 tensor with a fixed length (self.config.max_num_mention). 
                every element in the tensor is the start position index for the mentions. 
            pad_gold_end_index_labels: a tf.int32 tensor with a fixed length (self.config.max_num_mention). 
                every element in the tensor is the end position index of the mentions. 
            pad_text_len: a tf.int32 tensor with a fixed length (self.config.num_window). 
                every positive element in the tensor indicates that the number of subtokens in the window. 
        Returns:
            gold_start_sequence_labels: a tf.int32 tensor with the shape of (num_subtoken_in_doc). 
                if the element in the tensor equals to 0, this subtoken is not a start for a mention. 
                if the elemtn in the tensor equals to 1, this subtoken is a start for a mention.  
            gold_end_sequence_labels: a tf.int32 tensor with the shape of (num_subtoken_in_doc). 
                if the element in the tensor equals to 0, this subtoken is not a end for a mention. 
                if the elemtn in the tensor equals to 1, this subtoken is a end for a mention.  
            gold_span_sequence_labels: a tf.int32 tensor with the shape of (num_subtoken_in_doc, num_subtoken_in_doc)/ 
                if the element[i][j] equals to 0, this subtokens from $i$ to $j$ is not a mention. 
                if the element[i][j] equals to 1, this subtokens from $i$ to $j$ is a mention. 
        """
        text_len = tf.math.maximum(pad_text_len, tf.zeros_like(pad_text_len, tf.int32)) # (num_of_non_empty_window)
        num_subtoken_in_doc = tf.math.reduce_sum(text_len) # the value should be num_subtoken_in_doc 
        
        gold_start_end_mask = tf.cast(tf.math.greater_equal(pad_gold_start_index_labels, tf.zeros_like(pad_gold_start_index_labels, tf.int32)), tf.bool) # (max_num_mention)
        gold_start_index_labels = self.boolean_mask_1d(pad_gold_start_index_labels, gold_start_end_mask, name_scope="gold_starts", use_tpu=self.use_tpu) # (num_of_mention)
        gold_end_index_labels = self.boolean_mask_1d(pad_gold_end_index_labels, gold_start_end_mask, name_scope="gold_ends", use_tpu=self.use_tpu) # (num_of_mention)

        gold_start_sequence_labels = self.scatter_gold_index_to_label_sequence(gold_start_index_labels, num_subtoken_in_doc) # (num_subtoken_in_doc)
        gold_end_sequence_labels = self.scatter_gold_index_to_label_sequence(gold_end_index_labels, num_subtoken_in_doc) # (num_subtoken_in_doc)
        gold_span_sequence_labels = self.scatter_span_sequence_labels(gold_start_index_labels, gold_end_index_labels, num_subtoken_in_doc) # (num_subtoken_in_doc, num_subtoken_in_doc)

        return gold_start_sequence_labels, gold_end_sequence_labels, gold_span_sequence_labels


    def scatter_gold_index_to_label_sequence(self, gold_index_labels, expect_length_of_labels):
        """
        Desc:
            transform the mention start/end position index tf.int32 Tensor to a tf.int32 Tensor with 1/0 labels for the input subtoken sequences.
            1 denotes this subtoken is the start/end for a mention. 
        Args:
            gold_index_labels: a tf.int32 Tensor with mention start/end position index in the original document. 
            expect_length_of_labels: the number of subtokens in the original document. 
        """
        gold_labels_pos = tf.reshape(gold_index_labels, [-1, 1]) # (num_of_mention, 1)
        gold_value = tf.reshape(tf.ones_like(gold_index_labels), [-1]) # (num_of_mention)
        label_shape = tf.Variable(expect_length_of_labels) 
        label_shape = tf.reshape(label_shape, [1]) # [1]
        gold_sequence_labels = tf.cast(tf.scatter_nd(gold_labels_pos, gold_value, label_shape), tf.int32) # (num_subtoken_in_doc)
        return gold_sequence_labels


    def scatter_span_sequence_labels(self, gold_start_index_labels, gold_end_index_labels, expect_length_of_labels):
        """
        Desc:
            transform the mention (start, end) position pairs to a span matrix gold_span_sequence_labels. 
                matrix[i][j]: whether the subtokens between the position $i$ to $j$ can be a mention.  
                if matrix[i][j] == 0: from $i$ to $j$ is not a mention. 
                if matrix[i][j] == 1: from $i$ to $j$ is a mention.
        Args:
            gold_start_index_labels: a tf.int32 Tensor with mention start position index in the original document. 
            gold_end_index_labels: a tf.int32 Tensor with mention end position index in the original document. 
            expect_length_of_labels: a scalar, should be the same with num_subtoken_in_doc
        """ 
        gold_span_index_labels = tf.stack([gold_start_index_labels, gold_end_index_labels], axis=1) # (num_of_mention, 2)
        gold_span_value = tf.reshape(tf.ones_like(gold_start_index_labels, tf.int32), [-1]) # (num_of_mention)
        gold_span_label_shape = tf.Variable([expect_length_of_labels, expect_length_of_labels]) 
        gold_span_label_shape = tf.reshape(gold_span_label_shape, [-1])

        gold_span_sequence_labels = tf.cast(tf.scatter_nd(gold_span_index_labels, gold_span_value, gold_span_label_shape), tf.int32) # (num_subtoken_in_doc, num_subtoken_in_doc)
        return gold_span_sequence_labels


    def compute_score_and_loss(self, pred_sequence_logits, gold_sequence_labels, loss_mask=None):
        """
        Desc:
            compute the unifrom start/end loss and probabilities. 
        Args:
            pred_sequence_logits: (input_shape, 1) 
            gold_sequence_labels: (input_shape, 1)
            loss_mask: [optional] if is not None, it should be (input_shape). should be tf.int32 0/1 tensor. 
            FOR start/end score and loss, input_shape should be num_subtoken_in_doc.
            FOR span score and loss, input_shape should be num_subtoken_in_doc * num_subtoken_in_doc. 
        """
        pred_sequence_probabilities = tf.cast(tf.reshape(tf.sigmoid(pred_sequence_logits), [-1]),tf.float32) # (input_shape)
        expand_pred_sequence_scores = tf.stack([(1 - pred_sequence_probabilities), pred_sequence_probabilities], axis=-1) # (input_shape, 2)
        expand_gold_sequence_labels = tf.cast(tf.one_hot(tf.reshape(gold_sequence_labels, [-1]), 2, axis=-1), tf.float32) # (input_shape, 2)

        loss = tf.keras.losses.binary_crossentropy(expand_gold_sequence_labels, expand_pred_sequence_scores)
        # loss -> shape is (input_shape)

        if loss_mask is not None:
            loss = tf.multiply(loss, tf.cast(loss_mask, tf.float32))

        total_loss = tf.reduce_mean(loss)
        # total_loss -> a scalar 

        return total_loss, pred_sequence_probabilities


    def transform_overlap_windows_to_original_doc(self, doc_overlap_window_embs, doc_overlap_input_mask):
        """
        Desc:
            hidden_size should be equal to embeddding_size. 
        Args:
            doc_overlap_window_embs: (num_window, window_size, hidden_size). 
                the output of (num_window, window_size) input_ids forward into BERT model. 
            doc_overlap_input_mask: (num_window, window_size). A tf.int32 Tensor contains 0/1. 
                0 represents token in this position should be neglected. 1 represents token in this position should be reserved. 
        """
        ones_input_mask = tf.ones_like(doc_overlap_input_mask, tf.int32) # (num_window, window_size)
        cumsum_input_mask = tf.math.cumsum(ones_input_mask, axis=1) # (num_window, window_size)
        offset_input_mask = tf.tile(tf.expand_dims(tf.range(self.config.num_window) * self.config.window_size, 1), [1, self.config.window_size]) # (num_window, window_size)
        offset_cumsum_input_mask = offset_input_mask + cumsum_input_mask # (num_window, window_size)
        global_input_mask = tf.math.multiply(ones_input_mask, offset_cumsum_input_mask) # (num_window, window_size)
        global_input_mask = tf.reshape(global_input_mask, [-1]) # (num_window * window_size)
        global_input_mask_index = self.boolean_mask_1d(global_input_mask, tf.math.greater(global_input_mask, tf.zeros_like(global_input_mask, tf.int32))) # (num_subtoken_in_doc)

        doc_overlap_window_embs = tf.reshape(doc_overlap_window_embs, [-1, self.config.hidden_size]) # (num_window * window_size, hidden_size)
        original_doc_embs = tf.gather(doc_overlap_window_embs, global_input_mask_index) # (num_subtoken_in_doc, hidden_size)

        return original_doc_embs 


    def ffnn(self, inputs, hidden_size, output_size, dropout=None, name_scope="fully-conntected-neural-network",
        hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
        """
        Desc:
            fully-connected neural network. 
            transform non-linearly the [input] tensor with [hidden_size] to a fix [output_size] size.  
        Args: 
            hidden_size: should be the size of last dimension of [inputs]. 
        """
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            hidden_weights = tf.get_variable("hidden_weights", [hidden_size, output_size],
                initializer=hidden_initializer)
            hidden_bias = tf.get_variable("hidden_bias", [output_size], initializer=tf.zeros_initializer())
            outputs = tf.nn.relu(tf.nn.xw_plus_b(inputs, hidden_weights, hidden_bias))

            if dropout is not None:
                outputs = tf.nn.dropout(outputs, dropout)

        return outputs 


    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)


    def get_shape(self, x, dim):
        """
        Desc:
            return the size of input x in DIM. 
        """ 
        return x.get_shape()[dim].value or tf.shape(x)[dim]


    def boolean_mask_1d(self, itemtensor, boolmask_indicator, name_scope="boolean_mask1d", use_tpu=False):
        """
        Desc:
            the same functionality of tf.boolean_mask. 
            The tf.boolean_mask operation is not available on the cloud TPU. 
        Args:
            itemtensor : a Tensor contains [tf.int32, tf.float32] numbers. Should be 1-Rank.
            boolmask_indicator : a tf.bool Tensor. Should be 1-Rank. 
            scope : name scope for the operation. 
            use_tpu : if False, return tf.boolean_mask.  
        """
        with tf.name_scope(name_scope):
            if not use_tpu:
                return tf.boolean_mask(itemtensor, boolmask_indicator)

            boolmask_sum = tf.reduce_sum(tf.cast(boolmask_indicator, tf.int32))
            selected_positions = tf.cast(boolmask_indicator, dtype=tf.float32)
            indexed_positions = tf.cast(tf.multiply(tf.cumsum(selected_positions), selected_positions),dtype=tf.int32)
            one_hot_selector = tf.one_hot(indexed_positions - 1, boolmask_sum, dtype=tf.float32)
            sampled_indices = tf.cast(tf.tensordot(tf.cast(tf.range(tf.shape(boolmask_indicator)[0]), dtype=tf.float32),
                one_hot_selector,axes=[0, 0]),dtype=tf.int32)
            sampled_indices = tf.reshape(sampled_indices, [-1])
            mask_itemtensor = tf.gather(itemtensor, sampled_indices)

            return mask_itemtensor







