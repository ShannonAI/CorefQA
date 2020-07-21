#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import os
import sys 

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import tensorflow as tf
from bert import modeling



class CorefQAModel(object):
    def __init__(self, config):
        self.config = config 
        self.dropout = None
        self.pad_idx = 0 
        self.mention_start_idx = 37
        self.mention_end_idx = 42
        self.bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
        self.bert_config.hidden_dropout_prob = config.dropout_rate
        self.cls_in_vocab = 101
        self.sep_in_vocab = 102


    def get_coreference_resolution_and_loss(self, instance, is_training, use_tpu=False):


        self.use_tpu = use_tpu 
        self.dropout = self.get_dropout(self.config.dropout_rate, is_training)

        flat_window_input_ids, flat_window_input_mask, flat_doc_sentence_map, window_text_len, speaker_ids, gold_starts, gold_ends, gold_cluster_ids = instance
        # flat_input_ids: (num_window, window_size)
        # flat_doc_overlap_input_mask: (num_window, window_size)
        # flat_sentence_map: (num_window, window_size)
        # text_len: dynamic length and is padded to fix length
        # gold_start: (max_num_mention), mention start index in the original (NON-OVERLAP) document. Pad with -1 to the fix length max_num_mention.
        # gold_end: (max_num_mention), mention end index in the original (NON-OVERLAP) document. Pad with -1 to the fix length max_num_mention.
        # cluster_ids/speaker_ids is not used in the mention proposal model.

        flat_window_input_ids = tf.math.maximum(flat_window_input_ids, tf.zeros_like(flat_window_input_ids, tf.int32)) # (num_window * window_size)
        
        flat_doc_overlap_input_mask = tf.where(tf.math.greater_equal(flat_window_input_mask, 0), 
            x=tf.ones_like(flat_window_input_mask, tf.int32), y=tf.zeros_like(flat_window_input_mask, tf.int32)) # (num_window * window_size)
        # flat_doc_overlap_input_mask = tf.math.maximum(flat_doc_overlap_input_mask, tf.zeros_like(flat_doc_overlap_input_mask, tf.int32))
        flat_doc_sentence_map = tf.math.maximum(flat_doc_sentence_map, tf.zeros_like(flat_doc_sentence_map, tf.int32)) # (num_window * window_size)
        
        gold_start_end_mask = tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts, tf.int32)), tf.bool) # (max_num_mention)
        gold_start_index_labels = self.boolean_mask_1d(gold_starts, gold_start_end_mask, name_scope="gold_starts", use_tpu=self.use_tpu) # (num_of_mention)
        gold_end_index_labels = self.boolean_mask_1d(gold_ends, gold_start_end_mask, name_scope="gold_ends", use_tpu=self.use_tpu) # (num_of_mention)

        gold_cluster_mask = tf.cast(tf.math.greater_equal(gold_cluster_ids, tf.zeros_like(gold_cluster_ids, tf.int32)), tf.bool) # (max_num_cluster)
        gold_cluster_ids = self.boolean_mask_1d(gold_cluster_ids, gold_cluster_mask, name_scope="gold_cluster", use_tpu=self.use_tpu)

        window_text_len = tf.math.maximum(window_text_len, tf.zeros_like(window_text_len, tf.int32)) # (num_of_non_empty_window)
        num_subtoken_in_doc = tf.math.reduce_sum(window_text_len) # the value should be num_subtoken_in_doc 
        ####################
        ####################
        ## mention proposal stage starts 
        mention_input_ids = tf.reshape(flat_window_input_ids, [-1, self.config.window_size]) # (num_window, window_size)
        # each row of mention_input_ids is a subdocument 
        mention_input_mask = tf.ones_like(mention_input_ids, tf.int32) # (num_window, window_size)
        mention_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=mention_input_ids, input_mask=mention_input_mask, use_one_hot_embeddings=False, scope='bert')

        mention_doc_overlap_window_embs = mention_model.get_sequence_output() # (num_window, window_size, hidden_size)
        # get BERT embeddings for mention_input_ids 
        doc_overlap_input_mask = tf.reshape(flat_doc_overlap_input_mask, [self.config.num_window, self.config.window_size]) # (num_window, window_size)

        mention_doc_flat_embs = self.transform_overlap_sliding_windows_to_original_document(mention_doc_overlap_window_embs, doc_overlap_input_mask) 
        mention_doc_flat_embs = tf.reshape(mention_doc_flat_embs, [-1, self.config.hidden_size]) # (num_subtoken_in_doc, hidden_size) 

        candidate_mention_starts = tf.tile(tf.expand_dims(tf.range(num_subtoken_in_doc), 1), [1, self.config.max_span_width]) # (num_subtoken_in_doc, max_span_width)
        # getting all eligible mentions in each subdocument
        # the number if eligible mentions of each subdocument is  config.max_span_width * num_subtoken_in_doc
        candidate_mention_ends = tf.math.add(candidate_mention_starts, tf.expand_dims(tf.range(self.config.max_span_width), 0)) # (num_subtoken_in_doc, max_span_width)
        
        candidate_mention_sentence_start_idx = tf.gather(flat_doc_sentence_map, candidate_mention_starts) # (num_subtoken_in_doc, max_span_width)
        candidate_mention_sentence_end_idx = tf.gather(flat_doc_sentence_map, candidate_mention_ends) # (num_subtoken_in_doc, max_span_width)
        
        candidate_mention_mask = tf.logical_and(candidate_mention_ends < num_subtoken_in_doc, tf.equal(candidate_mention_sentence_start_idx, candidate_mention_sentence_end_idx))
        candidate_mention_mask = tf.reshape(candidate_mention_mask, [-1]) 

        candidate_mention_starts = self.boolean_mask_1d(tf.reshape(candidate_mention_starts, [-1]), candidate_mention_mask, name_scope="candidate_mention_starts", use_tpu=self.use_tpu)
        candidate_mention_ends = self.boolean_mask_1d(tf.reshape(candidate_mention_ends, [-1]), candidate_mention_mask, name_scope="candidate_mention_ends", use_tpu=self.use_tpu)
        # num_candidate_mention_in_doc is smaller than num_subtoken_in_doc

        candidate_cluster_idx_labels = self.get_candidate_cluster_labels(candidate_mention_starts, candidate_mention_ends, gold_start_index_labels, gold_end_index_labels, gold_cluster_ids)
        # 

        candidate_mention_span_embs, candidate_mention_start_embs, candidate_mention_end_embs = self.get_candidate_span_embedding(
            mention_doc_flat_embs, candidate_mention_starts, candidate_mention_ends) 
        # candidate_mention_span_embs -> (num_candidate_mention_in_doc, 2 * hidden_size)
        # candidate_mention_start_embs -> (num_candidate_mention_in_doc, hidden_size)
        # candidate_mention_end_embs -> (num_candidate_mention_in_doc, hidden_size)

        gold_label_candidate_mention_spans, gold_label_candidate_mention_starts, gold_label_candidate_mention_ends = self.get_candidate_mention_gold_sequence_label(
            candidate_mention_starts, candidate_mention_ends, gold_start_index_labels, gold_end_index_labels, num_subtoken_in_doc)
        # gold_label_candidate_mention_spans -> (num_candidate_mention_in_doc)
        # gold_label_candidate_mention_starts -> (num_candidate_mention_in_doc)
        # gold_label_candidate_mention_ends -> (num_candidate_mention_in_doc)

        mention_proposal_loss, candidate_mention_start_prob, candidate_mention_end_prob, candidate_mention_span_prob, candidate_mention_span_scores = self.get_mention_score_and_loss(
            candidate_mention_span_embs, candidate_mention_start_embs, candidate_mention_end_embs, gold_label_candidate_mention_spans=gold_label_candidate_mention_spans, 
            gold_label_candidate_mention_starts=gold_label_candidate_mention_starts, gold_label_candidate_mention_ends=gold_label_candidate_mention_ends, expect_length_of_labels=num_subtoken_in_doc)
        # mention_proposal_loss -> a scalar 
        # candidate_mention_start_prob, candidate_mention_end_prob, candidate_mention_span_prob, -> (num_candidate_mention_in_doc)

        self.k = tf.minimum(self.config.max_candidate_mentions, tf.to_int32(tf.floor(tf.to_float(num_subtoken_in_doc) * self.config.top_span_ratio)))
        # self.k is a hyper-parameter. We want to select the top self.k mentions from the config.max_span_width * num_subtoken_in_doc mentions.

        candidate_mention_span_scores = tf.reshape(candidate_mention_span_scores, [-1])
        topk_mention_span_scores, topk_mention_span_indices = tf.nn.top_k(candidate_mention_span_scores, self.k, sorted=False) 
        topk_mention_span_indices = tf.reshape(topk_mention_span_indices, [-1])
        # topk_mention_span_scores -> (k,)
        # topk_mention_span_indices -> (k,)

        topk_mention_start_indices = tf.gather(candidate_mention_starts, topk_mention_span_indices) # (k,)
        topk_mention_end_indices = tf.gather(candidate_mention_ends, topk_mention_span_indices) # (k,)
        topk_mention_span_cluster_ids = tf.gather(candidate_cluster_idx_labels, topk_mention_span_indices) # (k,)
        topk_mention_span_scores = tf.gather(candidate_mention_span_scores, topk_mention_span_indices) # (k,)
        ## mention proposal stage ends
        ###########
        ###########


        ###### mention linking stage starts
        ## foward QA score computation starts
        ## for a given proposed mention i, we first compute the score of a span j being the correferent answer to i, denoted by s(j|i) 
        i0 = tf.constant(0)
        forward_qa_input_ids = tf.zeros((1, self.config.num_window, self.config.window_size + self.config.max_query_len + 2), dtype=tf.int32) # (1, num_window, max_query_len + window_size + 2)
        forward_qa_input_mask = tf.zeros((1, self.config.num_window, self.config.window_size + self.config.max_query_len + 2), dtype=tf.int32) # (1, num_window, max_query_len + window_size + 2)
        forward_qa_input_token_type_mask = tf.zeros((1, self.config.num_window, self.config.window_size + self.config.max_query_len + 2), dtype=tf.int32) # (1, num_window, max_query_len + window_size + 2)

        # prepare for non-overlap input token ids 
        nonoverlap_doc_input_ids = self.transform_overlap_sliding_windows_to_original_document(flat_window_input_ids, flat_doc_overlap_input_mask) # (num_subtoken_in_doc)
        overlap_window_input_ids = tf.reshape(flat_window_input_ids, [self.config.num_window, self.config.window_size]) # (num_window, window_size)

        @tf.function
        def forward_qa_mention_linking(i, batch_qa_input_ids, batch_qa_input_mask, batch_qa_input_token_type_mask):
            tmp_mention_start_idx = tf.gather(topk_mention_start_indices, i)
            tmp_mention_end_idx = tf.gather(topk_mention_end_indices, i)

            query_input_token_ids, mention_start_idx_in_sent, mention_end_idx_in_sent = self.get_query_token_ids(
                nonoverlap_doc_input_ids, flat_doc_sentence_map, tmp_mention_start_idx, tmp_mention_end_idx)

            query_pad_token_ids = tf.zeros([self.config.max_query_len - self.get_shape(query_input_token_ids, 0)], dtype=tf.int32)

            pad_query_input_token_ids = tf.concat([query_input_token_ids, query_pad_token_ids], axis=0) # (max_query_len,)
            pad_query_input_token_mask = tf.ones_like(pad_query_input_token_ids, tf.int32) # (max_query_len)
            pad_query_input_token_type_mask = tf.zeros_like(pad_query_input_token_ids, tf.int32) # (max_query_len)


            expand_pad_query_input_token_ids = tf.tile(tf.expand_dims(pad_query_input_token_ids, 0), [self.config.num_window, 1])  # (num_window, max_query_len)
            expand_pad_query_input_token_mask = tf.tile(tf.expand_dims(pad_query_input_token_mask, 0), [self.config.num_window, 1]) # (num_window, max_query_len)
            expand_pad_query_input_token_type_mask = tf.tile(tf.expand_dims(pad_query_input_token_type_mask, 0), [self.config.num_window, 1]) # (num_window, max_query_len)

            sep_tokens = tf.cast(tf.fill([self.config.num_window, 1], self.sep_in_vocab), tf.int32) # (num_window, 1)
            cls_tokens = tf.cast(tf.fill([self.config.num_window, 1], self.cls_in_vocab), tf.int32) # (num_window, 1)

            query_context_input_token_ids = tf.concat([cls_tokens, expand_pad_query_input_token_ids, sep_tokens, overlap_window_input_ids], axis=1) # (1, num_window, max_query_len + window_size + 2)
            query_context_input_token_mask = tf.concat([tf.ones_like(cls_tokens, tf.int32), expand_pad_query_input_token_mask, tf.ones_like(sep_tokens, tf.int32), tf.ones_like(overlap_window_input_ids, tf.int32)], axis=1) # (1, num_window, max_query_len + window_size + 2)
            query_context_input_token_type_mask = tf.concat([tf.zeros_like(cls_tokens, tf.int32), expand_pad_query_input_token_type_mask, tf.zeros_like(sep_tokens, tf.int32), tf.ones_like(overlap_window_input_ids, tf.int32)], axis=1) # (1, num_window, max_query_len + window_size + 2)


            return [tf.math.add(i, 1), tf.concat([batch_qa_input_ids, query_context_input_token_ids], 0), 
                    tf.concat([batch_qa_input_mask, query_context_input_token_mask], 0), 
                    tf.concat([batch_qa_input_token_type_mask, query_context_input_token_type_mask], 0)]



        _, stack_forward_qa_input_ids, stack_forward_qa_input_mask, stack_forward_qa_input_type_mask = tf.while_loop(
            cond=lambda i, o1, o2, o3 : i < self.k,
            body=forward_qa_mention_linking, 
            loop_vars=[i0, forward_qa_input_ids, forward_qa_input_mask, forward_qa_input_token_type_mask], 
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, None, None]), 
                tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None])])

        # stack_forward_qa_input_ids, stack_forward_qa_input_mask, stack_forward_qa_input_type_mask -> (k, num_window, max_query_len + window_size + 2)

        batch_forward_qa_input_ids = tf.reshape(stack_forward_qa_input_ids, [-1, self.config.max_query_len+self.config.window_size+2]) # (k * num_window, max_query_len + window_size + 2)
        batch_forward_qa_input_mask = tf.reshape(stack_forward_qa_input_mask, [-1, self.config.max_query_len+self.config.window_size+2]) # (k * num_window, max_query_len + window_size + 2)
        batch_forward_qa_input_type_mask = tf.reshape(stack_forward_qa_input_type_mask, [-1, self.config.max_query_len+self.config.window_size+2]) # (k * num_window, max_query_len + window_size + 2)

        forward_qa_linking_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=batch_forward_qa_input_ids, input_mask=batch_forward_qa_input_mask, 
            token_type_ids=batch_forward_qa_input_type_mask, use_one_hot_embeddings=False, 
            scope="bert")

        forward_qa_overlap_window_embs = forward_qa_linking_model.get_sequence_output() # (k * num_window, max_query_len + window_size + 2, hidden_size)
        forward_context_overlap_window_embs = self.transform_overlap_sliding_windows_to_original_document(forward_qa_overlap_window_embs, batch_forward_qa_input_type_mask)
        forward_context_overlap_window_embs = tf.reshape(forward_context_overlap_window_embs, [self.k*self.config.num_widnow, self.config.window_size])
        # forward_context_overlap_window_embs -> (k*num_window, window_size, hidden_size)

        expand_doc_overlap_input_mask = tf.tile(tf.expand_dims(doc_overlap_input_mask, 0), [self.k, 1, 1]) # (k, num_window, window_size)
        expand_doc_overlap_input_mask = tf.reshape(expand_doc_overlap_input_mask, [-1, self.config.window_size]) # (k * num_window, window_size)

        forward_context_flat_doc_embs = self.transform_overlap_sliding_windows_to_original_document(forward_context_overlap_window_embs, expand_doc_overlap_input_mask) # (k * num_subtoken_in_doc, hidden_size)
        forward_context_flat_doc_embs = self.reshape(forward_context_flat_doc_embs, [self.k, -1, self.config.hidden_size]) # (k, num_subtoken_in_doc, hidden_size)
        num_candidate_mention = self.get_shape(candidate_mention_span_embs, 0) # (num_candidate_mention_in_doc)
        forward_qa_mention_pos_offset = tf.cast(tf.tile(tf.reshape(tf.range(0, num_candidate_mention) * num_subtoken_in_doc, [1, -1]), [self.k, 1]), tf.int32) # (k, num_candidate_mention_in_doc)

        forward_qa_mention_starts = tf.tile(tf.expand_dims(candidate_mention_starts, 0), [self.k, 1]) + forward_qa_mention_pos_offset # (k, num_candidate_mention_in_doc)
        forward_qa_mention_ends = tf.tile(tf.expand_dims(candidate_mention_ends, 0), [self.k, 1]) + forward_qa_mention_pos_offset # (k, num_candidate_mention_in_doc)

        forward_qa_mention_span_embs, forward_qa_mention_start_embs, forward_qa_mention_end_embs = self.get_candidate_span_embedding(tf.reshape(forward_context_flat_doc_embs, 
                [-1, self.config.hidden_size]), tf.reshape(forward_qa_mention_starts, [-1]), tf.reshape(forward_qa_mention_ends, [-1]))
        # forward_qa_mention_span_embs -> (k * num_candidate_mention_in_doc, hidden_size*2)
        # forward_qa_mention_start_embs -> (k * num_candidate_mention_in_doc, hidden_size)

        self.c = tf.to_int32(tf.minimum(self.config.max_top_antecedents, self.k))

        forward_qa_mention_span_scores, forward_qa_mention_start_scores, forward_qa_mention_end_scores = self.get_mention_score_and_loss(forward_qa_mention_span_embs, 
                forward_qa_mention_start_embs, forward_qa_mention_end_embs, name_scope="forward_qa") 
        # forward_qa_mention_span_prob, forward_qa_mention_start_prob, forward_qa_mention_end_prob -> (k * num_candidate_mention_in_doc)

        # computes the s(j|i) for all eligible span j in the document 
        if self.config.sec_qa_mention_score:
            forward_qa_mention_span_scores = (forward_qa_mention_span_scores + forward_qa_mention_start_scores + forward_qa_mention_end_scores)/3.0
        else:
            forward_qa_mention_span_scores = forward_qa_mention_span_scores

        forward_candidate_mention_span_scores = tf.reshape(forward_qa_mention_span_scores, [self.k, -1]) # (k, num_candidate_mention_in_doc)
        forward_topc_mention_span_scores, local_forward_topc_mention_span_indices = tf.nn.top_k(forward_candidate_mention_span_scores, self.c, sorted=False) # (k, c)
        # for each i, we only maintain the top self.c spans based on s(j|i)
        local_flat_forward_topc_mention_span_indices = tf.reshape(local_forward_topc_mention_span_indices, [-1]) # (k * c)

        # topk_mention_start_indices
        forward_topc_mention_start_indices = tf.gather(candidate_mention_starts, local_flat_forward_topc_mention_span_indices) # (k, c)
        forward_topc_mention_end_indices = tf.gather(candidate_mention_ends, local_flat_forward_topc_mention_span_indices) # (k, c)
        forward_topc_mention_span_scores_in_mention_proposal = tf.gather(candidate_mention_span_scores, local_flat_forward_topc_mention_span_indices) # (k, c)
        forward_topc_span_cluster_ids = tf.gather(candidate_cluster_idx_labels, local_flat_forward_topc_mention_span_indices)
        ## foward QA score computation ends


        ## backward QA score computation begins
        ## we need to compute the score of backward score, i.e., the span i is the correferent answer for j, denoted by s(i|j)
        i0 = tf.constant(0)
        backward_qa_input_ids = tf.zeros((1, self.config.max_query_len + self.config.max_context_len + 2), dtype=tf.int32) # (1, max_query_len + max_context_len + 2)
        backward_qa_input_mask = tf.zeros((1, self.config.max_query_len + self.config.max_context_len + 2), dtype=tf.int32) # (1, max_query_len + max_context_len + 2)
        backward_qa_input_token_type_mask = tf.zeros((1, self.config.max_query_len + self.config.max_context_len + 2), dtype=tf.int32) # (1, max_query_len + max_context_len + 2)
        backward_qa_mention_start_in_context = tf.convert_to_tensor(tf.constant([0]), dtype=tf.int32)
        backward_qa_mention_end_in_context = tf.convert_to_tensor(tf.constant([0]), dtype=tf.int32)

        @tf.function
        def backward_qa_mention_linking(i, batch_qa_input_ids, batch_qa_input_mask, batch_qa_input_token_type_mask, 
            batch_qa_mention_start_in_context, batch_qa_mention_end_in_context):

            tmp_query_mention_start_idx = tf.gather(forward_topc_mention_start_indices, i)
            tmp_query_mention_end_idx = tf.gather(forward_topc_mention_end_indices, i)

            tmp_index_for_topk_mention = tf.floor_div(i, self.k)
            tmp_context_mention_start_idx = tf.gather(topk_mention_start_indices, tmp_index_for_topk_mention)
            tmp_context_mention_end_idx = tf.gather(topk_mention_end_indices, tmp_index_for_topk_mention)

            query_input_token_ids, mention_start_idx_in_query, mention_end_idx_in_query = self.get_query_token_ids(
                nonoverlap_doc_input_ids, flat_doc_sentence_map, tmp_query_mention_start_idx, tmp_query_mention_end_idx)

            context_input_token_ids, mention_start_idx_in_context, mention_end_idx_in_context = self.get_query_token_ids(
                nonoverlap_doc_input_ids, flat_doc_sentence_map, tmp_context_mention_start_idx, tmp_context_mention_end_idx)

            query_pad_token_ids = tf.zeros([self.config.max_query_len - self.get_shape(query_input_token_ids, 0)], dtype=tf.int32)
            context_pad_token_ids = tf.zeros([self.config.max_context_len - self.get_shape(context_input_token_ids, 0)], dtype=tf.int32)

            pad_query_input_token_ids = tf.concat([query_input_token_ids, query_pad_token_ids], axis=0) # (max_query_len)
            pad_query_input_token_mask = tf.ones_like(pad_query_input_token_ids, tf.int32)
            pad_query_input_token_type_mask = tf.zeros_like(pad_query_input_token_ids, tf.int32)

            pad_context_input_token_ids = tf.concat([context_input_token_ids, context_pad_token_ids], axis=0) # (max_context_len)
            pad_context_input_token_mask = tf.ones_like(pad_context_input_token_ids, tf.int32)
            pad_context_input_token_type_mask = tf.ones_like(pad_context_input_token_ids, tf.int32)

            sep_tokens = tf.cast(tf.fill([1], self.sep_in_vocab), tf.int32) # (num_window, 1)
            cls_tokens = tf.cast(tf.fill([1], self.cls_in_vocab), tf.int32) # (num_window, 1)

            query_context_input_token_ids = tf.concat([cls_tokens, pad_query_input_token_ids, sep_tokens, pad_context_input_token_ids], axis=1)
            query_context_input_token_mask = tf.concat([tf.ones_like(cls_tokens, tf.int32), pad_query_input_token_mask, tf.zeros_like(sep_tokens, tf.int32), pad_context_input_token_mask], axis=1)
            query_context_input_token_type_mask = tf.concat([tf.zeros_like(cls_tokens, tf.int32), pad_query_input_token_type_mask, tf.zeros_like(sep_tokens, tf.int32), pad_context_input_token_type_mask], axis=1)

            return [tf.math.add(i, 1), tf.concat([batch_qa_input_ids, query_context_input_token_ids], 0), 
                    tf.concat([batch_qa_input_mask, query_context_input_token_mask], 0), 
                    tf.concat([batch_qa_input_token_type_mask, query_context_input_token_type_mask], 0), 
                    tf.concat([backward_qa_mention_start_in_context, mention_start_idx_in_context], 0), 
                    tf.concat([ backward_qa_mention_end_in_context, mention_end_idx_in_context], 0)]


        _, stack_backward_qa_input_ids, stack_backward_qa_input_mask, stack_backward_qa_input_type_mask, stack_backward_mention_start_in_context, stack_backward_mention_end_in_context = tf.while_loop(
            cond = lambda i, o1, o2, o3, o4, o5: i < self.k * self.c,
            body=backward_qa_mention_linking, 
            loop_vars=[i0, backward_qa_input_ids, backward_qa_input_mask, backward_qa_input_token_type_mask, backward_qa_mention_start_in_context, backward_qa_mention_end_in_context], 
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
            tf.TensorShape([None]), tf.TensorShape([None])]) 

        # stack_backward_qa_input_ids, stack_backward_qa_input_mask, stack_backward_qa_input_type_mask -> (k*c, max_query_len + max_context_len + 2)
        # stack_backward_mention_start_in_context, stack_backward_mention_end_in_context -> (k*c,)

        batch_backward_qa_input_ids = tf.reshape(stack_backward_qa_input_ids, [-1, self.config.max_query_len+self.config.max_context_len+2])
        batch_backward_qa_input_mask = tf.reshape(stack_backward_qa_input_mask, [-1, self.config.max_query_len+self.config.max_context_len+2])
        batch_backward_qa_input_type_mask = tf.reshape(stack_backward_qa_input_type_mask, [-1, self.config.max_query_len+self.config.max_context_len+2])

        backward_qa_linking_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=batch_backward_qa_input_ids, input_mask=batch_backward_qa_input_mask, 
            token_type_ids=batch_backward_qa_input_type_mask, use_one_hot_embeddings=False, 
            scope="bert")

        backward_query_context_embs = backward_qa_linking_model.get_sequence_output() # (k*c, max_query_len + max_context_len + 2, hidden_size)
        backward_query_context_embs = tf.reshape(backward_query_context_embs, [self.k*self.c, -1, self.config.hidden_size])
        flat_batch_backward_qa_input_type_mask = tf.reshape(batch_backward_qa_input_type_mask, [self.k*self.c, -1])

        backward_context_flat_embs = self.transform_overlap_sliding_windows_to_original_document(backward_query_context_embs, flat_batch_backward_qa_input_type_mask) # (k*c, max_context_len, hidden_size)
        batch_backward_mention_start_in_context = tf.reshape(stack_backward_mention_start_in_context, [-1]) + tf.range(0, self.c*self.k) * (self.config.max_query_len+self.config.max_context_len) 
        batch_backward_mention_end_in_context = tf.reshape(stack_backward_mention_end_in_context, [-1]) + tf.range(0, self.c*self.k) * (self.config.max_query_len+self.config.max_context_len) 

        backward_qa_mention_span_embs, backward_qa_mention_start_embs, backward_qa_mention_end_embs = self.get_candidate_span_embedding(tf.reshape(backward_context_flat_embs, 
                [-1, self.config.hidden_size]), tf.reshape(batch_backward_mention_start_in_context, [-1]), tf.reshape(batch_backward_mention_end_in_context, [-1]))
        # backward_qa_mention_span_embs -> (k*c, 2*hidden_size)
        # backward_qa_mention_start_embs, backward_qa_mention_end_embs -> (k*c, hidden_size)

        backward_qa_mention_span_scores, backward_qa_mention_start_scores, backward_qa_mention_end_scores = self.get_mention_score_and_loss(backward_qa_mention_span_embs, 
                backward_qa_mention_start_embs, backward_qa_mention_end_embs, name_scope="backward_qa")
        # backward_qa_mention_span_prob -> (k*c)
        # backward_qa_mention_start_prob, backward_qa_mention_end_prob -> (k*c)

        if self.config.sec_qa_mention_score:
            backward_qa_mention_span_scores = (backward_qa_mention_span_scores + backward_qa_mention_start_scores + backward_qa_mention_end_scores)/3.0
        else:
            backward_qa_mention_span_scores = backward_qa_mention_span_scores 
        #############
        ############# backward QA computation ends
        
        expand_forward_topc_mention_span_scores = tf.tile(tf.expand_dims(forward_topc_mention_span_scores, 0), [self.k, 1]) # forward_topc_mention_span_scores -> (c); expand_forward_topc_mention_span_scores -> (c, k)
        expand_forward_topc_mention_span_scores_in_mention_proposal = tf.tile(tf.expand_dims(forward_topc_mention_span_scores_in_mention_proposal, 0), [self.k, 1])
        expand_topk_mention_span_scores = tf.tile(tf.expand_dims(topk_mention_span_scores, 1), [1, self.c]) # (k, c)

        backward_qa_mention_span_scores = tf.reshape(backward_qa_mention_span_scores, [self.k, self.c]) # (k, c)

        mention_span_linking_scores = (expand_forward_topc_mention_span_scores + backward_qa_mention_span_scores ) / 2.0 
        mention_span_linking_scores = mention_span_linking_scores+ expand_forward_topc_mention_span_scores_in_mention_proposal + expand_topk_mention_span_scores
        mention_span_linking_scores = tf.reshape(mention_span_linking_scores, [self.k, self.c]) # (k, c)
        dummy_scores = tf.zeros([self.k, 1]) # (k, 1)

        top_mention_span_linking_scores = tf.concat([dummy_scores, mention_span_linking_scores], axis=1) # (k, c)

        forward_topc_span_cluster_ids = tf.reshape(forward_topc_span_cluster_ids, [self.k, self.c]) # (k, c)
        same_cluster_indicator = tf.equal(forward_topc_span_cluster_ids, tf.expand_dims(topk_mention_span_cluster_ids, 1))  
        non_dummy_indicator = tf.expand_dims(topk_mention_span_cluster_ids > 0, 1)
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) 
        top_mention_span_linking_labels = tf.concat([dummy_labels, pairwise_labels], 1)

        linking_loss = self.marginal_likelihood_loss(top_mention_span_linking_scores, top_mention_span_linking_labels)

        total_loss = mention_proposal_loss + linking_loss 

        return total_loss, (topk_mention_start_indices, topk_mention_end_indices), (forward_topc_mention_start_indices, forward_topc_mention_end_indices), top_mention_span_linking_scores 


    def marginal_likelihood_loss(self, antecedent_scores, antecedent_labels):
        """
        Desc:
            marginal likelihood of gold antecedent spans form coreference cluster 
        Args:
            antecedent_scores: [k, c+1] the predicted scores by the model
            antecedent_labels: [k, c+1] the gold-truth cluster labels
        Returns:
            a scalar of loss 
        """
        gold_scores = tf.math.add(antecedent_scores, tf.log(tf.to_float(antecedent_labels)))
        marginalized_gold_scores = tf.math.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.math.reduce_logsumexp(antecedent_scores, [1])  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return tf.math.reduce_sum(loss)


    def get_query_token_ids(self, nonoverlap_doc_input_ids, sentence_map, mention_start_idx, mention_end_idx, paddding=True):
        """
        Desc:
            construct question based on the selected mention. 
        """
        nonoverlap_doc_input_ids = tf.reshape(nonoverlap_doc_input_ids, [-1])

        sentence_idx_for_mention = tf.gather(sentence_map, mention_start_idx)
        sentence_mask_for_mention = tf.math.equal(sentence_map, sentence_idx_for_mention)
        query_token_input_ids = self.boolean_mask_1d(nonoverlap_doc_input_ids, sentence_mask_for_mention, name_scope="query_mention", use_tpu=self.use_tpu)

        sentence_start = tf.where(tf.equal(nonoverlap_doc_input_ids, tf.gather(query_token_input_ids, tf.constant(0))))

        mention_start_in_sent = mention_start_idx - tf.cast(sentence_start, tf.int32) 
        mention_end_in_sent = mention_end_idx - tf.cast(sentence_start, tf.int32) 

        return query_token_input_ids, mention_start_in_sent, mention_end_in_sent 



    def get_mention_score_and_loss(self, candidate_mention_span_embs, candidate_mention_start_embs, candidate_mention_end_embs, 
        gold_label_candidate_mention_spans=None, gold_label_candidate_mention_starts=None, gold_label_candidate_mention_ends=None, expect_length_of_labels=None, 
        name_scope="mention"):

        candidate_mention_span_logits = self.ffnn(candidate_mention_span_embs, self.config.hidden_size*2, 1, dropout=self.dropout, name_scope="{}_span".format(name_scope))
        candidate_mention_start_logits = self.ffnn(candidate_mention_start_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="{}_start".format(name_scope))
        candidate_mention_end_logits = self.ffnn(candidate_mention_end_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="{}_end".format(name_scope))

        if gold_label_candidate_mention_spans is None or gold_label_candidate_mention_starts is None or gold_label_candidate_mention_ends is None: 
            candidate_mention_span_scores = tf.math.log(tf.sigmoid(candidate_mention_span_logits))
            candidate_mention_start_scores = tf.math.log(tf.sigmoid(candidate_mention_start_logits))
            candidate_mention_end_scores = tf.math.log(tf.sigmoid(candidate_mention_end_logits))

            return candidate_mention_span_scores, candidate_mention_start_scores, candidate_mention_end_scores


        start_loss, candidate_mention_start_probability = self.compute_mention_score_and_loss(candidate_mention_start_logits, gold_label_candidate_mention_starts)
        end_loss, candidate_mention_end_probability = self.compute_mention_score_and_loss(candidate_mention_end_logits, gold_label_candidate_mention_ends)
        span_loss, candidate_mention_span_probability = self.compute_mention_score_and_loss(candidate_mention_span_logits, gold_label_candidate_mention_spans)

        
        total_loss = start_loss + end_loss + span_loss
        candidate_mention_span_scores = (tf.math.log(candidate_mention_start_probability) + tf.math.log(candidate_mention_end_probability) + tf.math.log(candidate_mention_span_probability)) / 3.0 

        return total_loss, candidate_mention_start_probability, candidate_mention_end_probability, candidate_mention_span_probability, candidate_mention_span_scores


    def compute_mention_score_and_loss(self, pred_sequence_logits, gold_sequence_labels, loss_mask=None):
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


    def get_candidate_span_embedding(self, doc_sequence_embeddings, candidate_span_starts, candidate_span_ends):
        doc_sequence_embeddings = tf.reshape(doc_sequence_embeddings, [-1, self.config.hidden_size])

        span_start_embedding = tf.gather(doc_sequence_embeddings, candidate_span_starts)
        span_end_embedding = tf.gather(doc_sequence_embeddings, candidate_span_ends)
        span_embedding = tf.concat([span_start_embedding, span_end_embedding], 1) 

        return span_embedding, span_start_embedding, span_end_embedding 

    def get_candidate_mention_gold_sequence_label(self, candidate_mention_starts, candidate_mention_ends, 
        gold_start_index_labels, gold_end_index_labels, expect_length_of_labels):

        gold_start_sequence_label = self.scatter_gold_index_to_label_sequence(gold_start_index_labels, expect_length_of_labels)
        gold_end_sequence_label = self.scatter_gold_index_to_label_sequence(gold_end_index_labels, expect_length_of_labels)

        gold_label_candidate_mention_starts = tf.gather(gold_start_sequence_label, candidate_mention_starts)
        gold_label_candidate_mention_ends = tf.gather(gold_end_sequence_label, candidate_mention_ends)

        gold_mention_sparse_label = tf.stack([gold_start_index_labels, gold_end_index_labels], axis=1)
        gold_span_value = tf.reshape(tf.ones_like(gold_start_index_labels, tf.int32), [-1])
        gold_span_shape = tf.constant([expect_length_of_labels, expect_length_of_labels])
        gold_span_label = tf.cast(tf.scatter_nd(gold_mention_sparse_label, gold_span_value, gold_span_shape), tf.int32)

        candidate_mention_spans = tf.stack([candidate_mention_starts, candidate_mention_ends], axis=1)
        gold_label_candidate_mention_spans = tf.gather_nd(gold_span_label, tf.expand_dims(candidate_mention_spans, 1))
 
        return gold_label_candidate_mention_spans, gold_label_candidate_mention_starts, gold_label_candidate_mention_ends


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
        gold_label_sequence = tf.cast(tf.scatter_nd(gold_labels_pos, gold_value, label_shape), tf.int32) # (num_subtoken_in_doc)
        return gold_label_sequence


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


    def get_candidate_cluster_labels(self, candidate_mention_starts, candidate_mention_ends, 
            gold_mention_starts, gold_mention_ends, gold_cluster_ids):
        
        same_mention_start = tf.equal(tf.expand_dims(gold_mention_starts, 1), tf.expand_dims(candidate_mention_starts, 0))
        same_mention_end = tf.equal(tf.expand_dims(gold_mention_ends, 1), tf.expand_dims(candidate_mention_ends, 0)) 
        same_mention_span = tf.logical_and(same_mention_start, same_mention_end)
        
        candidate_cluster_idx_labels = tf.matmul(tf.expand_dims(gold_cluster_ids, 0), tf.to_int32(same_mention_span))  # [1, num_candidates]
        candidate_cluster_idx_labels = tf.squeeze(candidate_cluster_idx_labels, 0)  # [num_candidates]

        return candidate_cluster_idx_labels 


    def transform_overlap_sliding_windows_to_original_document(self, overlap_window_inputs, overlap_window_mask):
        """
        Desc:
            hidden_size should be equal to embeddding_size. 
        Args:
            doc_overlap_window_embs: (num_window, window_size, hidden_size). 
                the output of (num_window, window_size) input_ids forward into BERT model. 
            doc_overlap_input_mask: (num_window, window_size). A tf.int32 Tensor contains 0/1. 
                0 represents token in this position should be neglected. 1 represents token in this position should be reserved. 
        """
        ones_input_mask = tf.ones_like(overlap_window_mask, tf.int32) # (num_window, window_size)
        cumsum_input_mask = tf.math.cumsum(ones_input_mask, axis=1) # (num_window, window_size)
        offset_input_mask = tf.tile(tf.expand_dims(tf.range(self.config.num_window) * self.config.window_size, 1), [1, self.config.window_size]) # (num_window, window_size)
        offset_cumsum_input_mask = offset_input_mask + cumsum_input_mask # (num_window, window_size)
        global_input_mask = tf.math.multiply(ones_input_mask, offset_cumsum_input_mask) # (num_window, window_size)
        global_input_mask = tf.reshape(global_input_mask, [-1]) # (num_window * window_size)
        global_input_mask_index = self.boolean_mask_1d(global_input_mask, tf.math.greater(global_input_mask, tf.zeros_like(global_input_mask, tf.int32))) # (num_subtoken_in_doc)

        overlap_window_inputs = tf.reshape(overlap_window_inputs, [self.config.num_window * self.config.window_size, -1]) # (num_window * window_size, hidden_size)
        original_doc_inputs = tf.gather(overlap_window_inputs, global_input_mask_index)  # (num_subtoken_in_doc, hidden_size)

        return original_doc_inputs


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


    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)


    def get_shape(self, x, dim):
        """
        Desc:
            return the size of input x in DIM. 
        """ 
        return x.get_shape()[dim].value or tf.shape(x)[dim]


    def evaluate(self, top_span_starts, top_span_ends, predicted_antecedents, 
            gold_clusters, gold_starts, gold_ends):
        """
        Desc:
            expected cluster ids is : [[[21, 25], [18, 18]], [[63, 65], [46, 48], [27, 29]], [[88, 88], [89, 89]]]
        Args:
            top_span_starts: 
            top_span_ends:
            predicted_antecedents: 
        Returns:
            predicted_clusters: 
            gold_clusters:
            mention_to_predicted:
            mention_to_gold: 
        """ 
        # predicted_antecedents = np.argmax(predicted_antecedents, axis=-1)
        top_span_starts, top_span_ends, predicted_antecedents = top_span_starts.tolist(), top_span_ends.tolist(), predicted_antecedents.tolist()
        gold_clusters, gold_starts, gold_ends =  gold_clusters.tolist()[0], gold_starts.tolist()[0], gold_ends.tolist()[0]

        def transform_gold_labels(gold_clusters, gold_starts, gold_ends):
            gold_clusters_idx = [tmp for tmp in gold_clusters if tmp >= 0]
            gold_starts = [tmp for tmp in gold_starts if tmp >= 0]
            gold_ends = [tmp for tmp in gold_ends if tmp >= 0]

            gold_clusters_dict = {}
            gold_cluster_lst = []

            for idx, (tmp_start, tmp_end) in enumerate(zip(gold_starts, gold_ends)):
                tmp_cluster_idx = gold_clusters_idx[idx]
                if tmp_cluster_idx not in gold_clusters_dict.keys():
                    gold_cluster_lst.append(tmp_cluster_idx)
                    gold_clusters_dict[tmp_cluster_idx] = [[tmp_start, tmp_end]]
                else:
                    gold_clusters_dict[tmp_cluster_idx].append([tmp_start, tmp_end])

            gold_cluster = [gold_clusters_dict[tmp_idx] for tmp_idx in gold_cluster_lst]

            return gold_cluster, gold_starts, gold_ends

        gold_clusters, gold_starts, gold_ends = transform_gold_labels(gold_clusters, gold_starts, gold_ends)

        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    
        return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold


