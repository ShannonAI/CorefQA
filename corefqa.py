from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys 
import random
import threading

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)



import conll
import metrics
import util
import numpy as np
import tensorflow as tf
from bert import modeling
from bert import tokenization
import operation_funcs.mask as mask 




class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.max_segment_len = config['max_segment_len']
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None  # Load eval data lazily.
        self.dropout = None
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config['vocab_file'], do_lower_case=False)

        self.pad_idx = 0 
        self.mention_start_idx = 37
        self.mention_end_idx = 42

        self.coref_evaluator = metrics.CorefEvaluator()

    def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, 
            genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, span_mention):
        ##### _, total_loss = model.get_predictions_and_loss(input_ids, input_mask, text_len, speaker_ids, 
        ##### genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, span_mention)

        """
        Desc:
            input_mask: (max_sent_len, max_seg_len), 如果input_mask[i] > 0, 说明了当前位置的token是组成最终doc里面的一部分。
                如果input_mask[i] < 0, 说明了当前位置的token是overlap里面的词语，或者是speaker的补充词。
            e.g.: [[-3, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
            [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3]]
        """

        input_ids = tf.where(tf.cast(tf.math.greater_equal(input_ids, tf.zeros_like(input_ids)),tf.bool), x=input_ids, y=tf.zeros_like(input_ids)) 
        input_mask = tf.where(tf.cast(tf.math.greater_equal(input_mask, tf.zeros_like(input_mask)), tf.bool), x=input_mask, y=tf.zeros_like(input_mask)) 
        text_len = tf.where(tf.cast(tf.math.greater_equal(text_len, tf.zeros_like(text_len)), tf.bool), x= text_len, y=tf.zeros_like(text_len)) 
        speaker_ids = tf.where(tf.cast(tf.math.greater_equal(speaker_ids, tf.zeros_like(speaker_ids)),tf.bool), x=speaker_ids, y=tf.zeros_like(speaker_ids)) 
        gold_starts = tf.where(tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts)),tf.bool), x=gold_starts, y=tf.zeros_like(gold_starts)) 
        gold_ends = tf.where(tf.cast(tf.math.greater_equal(gold_ends, tf.zeros_like(gold_ends)),tf.bool), x=gold_ends, y=tf.zeros_like(gold_ends) ) 
        cluster_ids = tf.where(tf.cast(tf.math.greater_equal(cluster_ids, tf.zeros_like(cluster_ids)),tf.bool), x=cluster_ids, y=tf.zeros_like(cluster_ids)) 
        # sentence_map = tf.where(tf.cast(tf.math.greater_equal(sentence_map, tf.zeros_like(sentence_map)),tf.bool), x=sentence_map, y=tf.zeros_like(sentence_map)) 


        input_ids = tf.reshape(input_ids, [-1, self.config["max_segment_len"]])
        input_mask  = tf.reshape(input_mask, [-1, self.config["max_segment_len"]])
        text_len = tf.reshape(text_len, [-1])
        speaker_ids = tf.reshape(speaker_ids, [-1, self.config["max_segment_len"]])
        sentence_map = tf.reshape(sentence_map, [-1])
        cluster_ids = tf.reshape(cluster_ids, [-1]) 
        gold_starts = tf.reshape(gold_starts, [-1]) 
        gold_ends = tf.reshape(gold_ends, [-1]) 
        span_mention = tf.reshape(span_mention, [self.config["max_training_sentences"], self.config["max_segment_len"] * self.config["max_segment_len"]])

        self.input_ids = input_ids # (max_sent_len, max_seg_len)
        self.input_mask = input_mask  # (max_sent_len, max_seg_len)
        self.sentence_map = sentence_map 

        model = modeling.BertModel(
            config=self.bert_config, 
            is_training=is_training, 
            input_ids = input_ids, 
            input_mask = input_mask, 
            use_one_hot_embeddings=False, 
            scope="bert")
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

        doc_seq_emb = model.get_sequence_output() # (max_sentence_len, max_seg_len)
        doc_seq_emb, doc_overlap_mask = self.flatten_emb_by_sentence(doc_seq_emb, input_mask) # (max_sent_len * )
        doc_seq_emb = tf.reshape(doc_seq_emb, [-1, self.config["hidden_size"]])


        num_words = util.shape(doc_seq_emb, 0) # true words in one document  # senten_map 
        # num_words is smaller than the max_sentence_len * max_segment_len
        # candidate_span: 
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width])
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0)
        ######### sentence_map = mask.boolean_mask(tf.reshape(sentence_map, [-1]), doc_overlap_mask, use_tpu=self.config["tpu"])
        sentence_map = tf.reshape(sentence_map, [-1])

        candidate_start_sentence_indices = tf.gather(sentence_map, candidate_starts)
        candidate_end_sentence_indices = tf.gather(sentence_map, tf.minimum(candidate_ends, num_words - 2))

        # [num_words, max_span_width], 合法的span需要满足start/end不能越界；start/end必须在同一个句子里
        candidate_mask = tf.logical_and(candidate_ends < num_words,
                                        tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices))

        flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
        candidate_starts = mask.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask, use_tpu=self.config["tpu"] )
        candidate_ends = mask.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask, use_tpu=self.config["tpu"] )
        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids)

        candidate_binary_labels = candidate_cluster_ids > 0 
        # [num_candidates, emb] -> 候选答案的向量表示
        # [num_candidates, ] -> 候选答案的得分

        candidate_span_emb = self.get_span_emb(doc_seq_emb, candidate_starts, candidate_ends) # (candidate_mention, embedding)
        candidate_mention_scores = self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
        
        pred_probs = tf.sigmoid(candidate_mention_scores)
        ############################
        mention_proposal_loss = self.get_mention_proposal_loss(pred_probs, span_mention, candidate_starts, candidate_ends)

        # beam size 所有span的数量小于num_words * top_span_ratio
        k = tf.minimum(2, tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))
        c = tf.to_int32(tf.minimum(self.config["max_top_antecedents"], k))  # 初筛挑出0.4*500=200个候选，细筛再挑出50个候选

        top_span_scores, top_span_indices = tf.nn.top_k(candidate_mention_scores, k)

        top_span_indices = tf.reshape(top_span_indices, [-1]) # k
        top_span_starts = tf.gather(candidate_starts, top_span_indices) # k 
        top_span_ends = tf.gather(candidate_ends, top_span_indices)
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k] 

        self.topk_span_starts = top_span_starts 
        self.topk_span_ends = top_span_ends 

        i0 = tf.constant(0)
        num_forward_question = k 
        batch_qa_input_ids = tf.zeros((1, self.config["max_training_sentences"], self.config["max_segment_len"] + self.config["max_query_len"]), dtype=tf.int32)
        batch_qa_input_mask = tf.zeros((1, self.config["max_training_sentences"], self.config["max_segment_len"] + self.config["max_query_len"]), dtype=tf.int32)
        batch_qa_input_token_type_mask = tf.zeros((1, self.config["max_training_sentences"],  self.config["max_segment_len"] + self.config["max_query_len"]), dtype=tf.int32)
        # i0, batch_qa_input_ids, batch_qa_input_mask, batch_qa_input_token_type_mask
        batch_query_ids = tf.zeros((1, self.config["max_query_len"]), dtype=tf.int32)

        @tf.function
        def forward_qa_loop(i, link_qa_input_ids, link_qa_input_mask, link_qa_input_type_mask, link_qa_query_ids):
            tmp_context_input_ids = tf.reshape(self.input_ids,[-1, self.config["max_segment_len"]])  
            # (max_train_sent, max_segment_len)
            tmp_context_input_mask = tf.reshape(self.input_mask, [-1, self.config["max_segment_len"]]) 
            # tf.ones_like(tmp_context_input_ids) 
            actual_mask = tf.cast(tf.not_equal(self.input_mask, self.pad_idx), tf.int32)  
            # (max_train_sent, max_segment_len) 
            # def get_question_token_ids(self, input_ids, input_mask, sentence_map, top_start, top_end, special=True)
            question_tokens, start_in_sentence, end_in_sentence = self.get_question_token_ids(
                self.input_ids, self.input_mask, self.sentence_map, tf.gather(top_span_starts, i), tf.gather(top_span_ends, i))

            # quesiton_tokens: dynamic query lens 
            # pad or clip to max_query_len 
            # query_input_mask = tf.ones_like(question_tokens)
            ##### question_len = util.shape(question_tokens, 0)
            ##########if question_len < self.config["max_query_len"]:
            # pad_tokens = tf.zeros([self.config["max_query_len"] - question_len], dtype=tf.int32)
            pad_tokens = tf.zeros([self.config["max_query_len"] - util.shape(question_tokens, 0)], dtype=tf.int32)
            pad_query_tokens = tf.concat([question_tokens, pad_tokens], axis=0)
            #####else:
            ##########    pad_query_tokens = tf.gather(question_tokens, tf.range(0, self.config["max_query_len"]))
            ##############
            pad_query_token_mask = tf.ones_like(pad_query_tokens, dtype=tf.int32)
            ###### - pad_query_tokens = tf.keras.preprocessing.sequence.pad_sequences(question_tokens, maxlen=self.config["max_query_len"],
            ###### -     padding="post", truncating="post")
            # pad_quesiton_tokens: max_query_len 
            ###### - pad_query_token_mask = tf.keras.preprocessing.sequence.pad_sequences(query_input_mask, maxlen=self.config["max_query_len"],
            ###### -     padding="post", truncating="post")
            batch_query_tokens = tf.tile(tf.expand_dims(pad_query_tokens, 0), tf.constant([self.config["max_training_sentences"], 1])) 
            # batch_pad_question_tokens: (max_training_sentences, max_query_len)
            batch_query_token_type_mask = tf.zeros_like(batch_query_tokens)
            batch_query_token_mask = tf.tile(tf.expand_dims(pad_query_token_mask, 0), tf.constant([self.config["max_training_sentences"], 1])) 

            batch_context_tokens = tf.reshape(self.input_ids, [self.config["max_training_sentences"], self.config["max_segment_len"]])
            batch_context_token_type_mask = tf.ones_like(batch_context_tokens) # max_train_sent, max_segment_len 
            batch_context_token_mask = tf.ones_like(batch_context_tokens)


            batch_qa_input_ids = tf.concat([batch_query_tokens, batch_context_tokens], -1)
            batch_qa_input_token_type_mask = tf.concat([batch_query_token_type_mask, batch_context_token_type_mask], -1)
            batch_qa_input_mask = tf.concat([batch_query_token_mask, batch_context_token_mask], -1)

            batch_qa_input_ids = tf.cast(tf.reshape(batch_qa_input_ids, [1, self.config["max_training_sentences"], self.config["max_segment_len"] + self.config["max_query_len"]]), tf.int32)
            batch_qa_input_token_type_mask = tf.cast(tf.reshape(batch_qa_input_token_type_mask, [1, self.config["max_training_sentences"], self.config["max_segment_len"] + self.config["max_query_len"]]), tf.int32)
            batch_qa_input_mask = tf.cast(tf.reshape(batch_qa_input_mask, [1, self.config["max_training_sentences"], self.config["max_segment_len"] + self.config["max_query_len"]]), tf.int32)
            pad_query_tokens = tf.cast(tf.reshape(pad_query_tokens, [1, self.config["max_query_len"]]), tf.int32)

            # link_qa_input_ids, link_qa_input_mask, link_qa_input_type_mask
            return (i+1, tf.concat([link_qa_input_ids, batch_qa_input_ids], 0), 
                tf.concat([link_qa_input_mask, batch_qa_input_mask], 0), 
                tf.concat([link_qa_input_type_mask, batch_qa_input_token_type_mask], 0), 
                tf.concat([link_qa_query_ids, pad_query_tokens], 0))


        _, forward_qa_input_ids, forward_qa_input_mask, forward_qa_input_token_type_mask, qa_topk_query_tokens = tf.while_loop(
            cond=lambda i, o1, o2, o3, o4 : i < k, 
            body=forward_qa_loop, 
            loop_vars=[i0, batch_qa_input_ids, batch_qa_input_mask, batch_qa_input_token_type_mask, batch_query_ids], 
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, None, None]), 
                tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None]), 
                tf.TensorShape([None, None])])
        # 

        # forward_qa_input_ids -> (k, max_train_sent, max_query_len + max_segment_len) -> (k * max_train_sent, max_query_len + max_segment_len)
        forward_qa_input_ids = tf.reshape(forward_qa_input_ids, [-1, self.config["max_query_len"] + self.config["max_segment_len"]]) 
        forward_qa_input_mask = tf.reshape(forward_qa_input_mask, [-1, self.config["max_query_len"] + self.config["max_segment_len"]]) 
        forward_qa_input_token_type_mask = tf.reshape(forward_qa_input_token_type_mask, [-1, self.config["max_query_len"] + self.config["max_segment_len"]]) # self.config["max_query_len"] + self.config["max_segment_len"]]) 

        forward_bert_qa_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=forward_qa_input_ids, input_mask=forward_qa_input_mask, 
            token_type_ids=forward_qa_input_token_type_mask, use_one_hot_embeddings=False, 
            scope="forward_qa_linking")

        forward_qa_emb = forward_bert_qa_model.get_sequence_output() # (k * max_train_sent, max_query_len + max_segment_len, hidden_size)
        forward_qa_input_token_type_mask_bool = tf.cast(tf.reshape(forward_qa_input_token_type_mask, [-1, self.config["max_query_len"] + self.config["max_segment_len"]]), tf.bool)
        ###### forward_qa_input_token_type_mask_bool: [k * max_train_sent, max_query_len + max_segment_len]
        ######## forward_qa_input_token_type_mask_bool = tf.tile(tf.expand_dims(forward_qa_input_token_type_mask_bool, 2), [1, 1, self.config["hidden_size"]])
        # forward_qa_emb = tf.gather_nd(forward_qa_emb, tf.range(self.config["max_query_len"], self.config["max_query_len"] + self.config["max_segment_len"]), batch_dims=1)

        # forward_qa_emb = tf.slice(forward_qa_emb, [1, self.config["max_query_len"], ], [self.config["max_segment_len"]])
        forward_qa_emb = tf.reshape(forward_qa_emb, [k*self.config["max_training_sentences"], self.config["max_query_len"]+self.config["max_segment_len"], self.config["hidden_size"]])
        # forward_doc_emb = tf.slice(forward_qa_emb, [1, self.config["max_query_len"], 0], [k*self.config["max_training_sentences"], self.config["max_query_len"] + self.config["max_segment_len"], self.config["hidden_size"]])
        # forward_doc_emb = tf.slice(forward_qa_emb, [k*self.config["max_training_sentences"], self.config["max_query_len"], self.config["hidden_size"]], [k*self.config["max_training_sentences"], self.config["max_query_len"] + self.config["max_segment_len"], self.config["hidden_size"]])
        ##########################################forward_doc_emb = tf.slice(forward_qa_emb, [0, self.config["max_query_len"], 0], [k*self.config["max_training_sentences"], self.config["max_segment_len"], 
        ##########################################    self.config["hidden_size"]])
        print("$"*30)
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 1))
        print(util.shape(forward_qa_emb, 1))
        print(util.shape(forward_qa_emb, 1))
        print(util.shape(forward_qa_emb, 1))
        print(util.shape(forward_qa_emb, 1))
        print(util.shape(forward_qa_emb, 1))


        forward_qa_emb = mask.boolean_mask(tf.reshape(forward_qa_emb, [-1, self.config["hidden_size"]]), tf.reshape(forward_qa_input_token_type_mask, [-1]), use_tpu=self.config["tpu"])

        print("%"*30)
        print("checkout forward_qa_embedding")
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        print(util.shape(forward_qa_emb, 0))
        # (k * max_train_sent, max_segment_len, hidden_size)
        ############################################ forward_doc_emb = tf.reshape(forward_doc_emb, [-1, self.config["hidden_size"]]) 
        # (k * max_train_sent * max_segment_len, hidden_size)
        flat_sentence_map = tf.tile(tf.expand_dims(sentence_map, 0), [k, 1]) # (k, max_sent * max_segment) 
        flat_sentence_map = tf.reshape(flat_sentence_map, [-1])
        flat_sentence_map = tf.where(tf.cast(tf.math.greater_equal(flat_sentence_map, tf.zeros_like(flat_sentence_map)),tf.bool), x=flat_sentence_map, y=tf.zeros_like(flat_sentence_map)) 

        ################################################################################################################################################################
        ################
        flat_forward_doc_emb = mask.boolean_mask(forward_qa_emb, flat_sentence_map, use_tpu=self.config["tpu"])
        # flat_forward_doc_emb -> (k * non_overlap_doc_len * hidden_size)
        # flat_forward_doc_emb = forward_doc_emb 
        flat_forward_doc_emb = tf.reshape(flat_forward_doc_emb, [k, -1, self.config["hidden_size"]])
        # flat_forward_doc_emb -> (k, non_overlap_doc_len, hidden_size)
        non_overlap_doc_len = util.shape(flat_forward_doc_emb, 1)
        top_span_starts = tf.reshape(top_span_starts, [-1])
        top_span_ends = tf.reshape(top_span_ends, [-1])
        top_span_starts = tf.reshape(tf.tile(tf.expand_dims(top_span_starts, 0), [k, 1]), [k, k])
        top_span_ends = tf.reshape(tf.tile(tf.expand_dims(top_span_ends, 0), [k, 1]), [k, k])

        forward_pos_offset = tf.cast(tf.tile(tf.reshape(tf.range(0, k) * non_overlap_doc_len, [-1, 1]), [1, k]), tf.int32)
        top_span_starts = tf.reshape(tf.cast(top_span_starts + forward_pos_offset, tf.int32), [-1])
        top_span_ends = tf.reshape(tf.cast(top_span_ends + forward_pos_offset, tf.int32), [-1])

        forward_mention_start_emb = tf.gather(tf.reshape(flat_forward_doc_emb, [-1, self.config["hidden_size"]]), top_span_starts,) # (k, k, emb)
        forward_mention_end_emb = tf.gather(tf.reshape(flat_forward_doc_emb, [-1, self.config["hidden_size"]]), top_span_ends)

        ##### forward_mention_end_emb = tf.gather_nd(flat_forward_doc_emb, top_span_ends, batch_dims=0) # (k, k, emb)
        forward_mention_start_emb = tf.reshape(forward_mention_start_emb, [k*k, self.config["hidden_size"]])
        forward_mention_end_emb = tf.reshape(forward_mention_end_emb, [k*k, self.config["hidden_size"]])
        forward_mention_span_emb = tf.concat([forward_mention_start_emb, forward_mention_end_emb], 1) # (k, k emb * 2) 

        # with tf.variable_scope("forward_qa",):
        forward_mention_span_emb = tf.reshape(forward_mention_span_emb, [k*k, self.config["hidden_size"]*2])
        forward_mention_ij_score = util.ffnn(forward_mention_span_emb, 1, self.config["hidden_size"]*2, 1, self.dropout)
        ################

        forward_mention_ij_score = tf.reshape(forward_mention_ij_score, [k, k])
        topc_forward_scores, topc_forward_indices = tf.nn.top_k(forward_mention_ij_score, c, sorted=False)
        # topc_forward_scores, topc_forward_indices : [k, c]

        flat_topc_forward_indices = tf.reshape(topc_forward_indices, [-1])

        topc_start_index_doc = tf.gather(top_span_starts, flat_topc_forward_indices) # (k * c)
        topc_end_index_doc = tf.gather(top_span_ends, flat_topc_forward_indices) # (k * c)
        topc_span_scores = tf.gather(top_span_mention_scores, flat_topc_forward_indices) # (k*c)
        topc_span_cluster_ids = tf.gather(top_span_cluster_ids, flat_topc_forward_indices)

        # link_qa_input_ids, link_qa_input_mask, link_qa_input_type_mask, link_qa_query_ids
        backward_qa_input_ids = tf.zeros((1, self.config["max_query_len"] + self.config["max_context_len"]), dtype=tf.int32)
        backward_qa_input_mask = tf.zeros((1, self.config["max_query_len"] + self.config["max_context_len"]), dtype=tf.int32)
        backward_qa_input_token_type = tf.zeros((1, self.config["max_query_len"] + self.config["max_context_len"]), dtype=tf.int32)
        backward_start_in_sent = tf.zeros((1), dtype=tf.int32)
        backward_end_in_sent = tf.zeros((1), dtype=tf.int32)

        tile_top_span_starts = tf.reshape(tf.tile(tf.reshape(self.topk_span_starts, [1, -1]), [c, 1]), [-1])
        tile_top_span_ends = tf.reshape(tf.tile(tf.reshape(self.topk_span_ends, [1, -1]), [c, 1]), [-1])


        
        # backward_qa_input_ids, backward_qa_input_mask, backward_qa_input_token_type, backward_start_in_sent, backward_end_in_sent
        backward_qa_input_ids = tf.zeros([1, self.config["max_query_len"]+self.config["max_context_len"]], dtype=tf.int32)
        backward_qa_input_mask = tf.zeros([1, self.config["max_query_len"]+self.config["max_context_len"]], dtype=tf.int32)
        backward_qa_input_token_type = tf.zeros([1, self.config["max_query_len"]+self.config["max_context_len"]], dtype=tf.int32)
        tmp_start_in_sent = tf.convert_to_tensor(tf.constant([0]), dtype=tf.int32)
        tmp_end_in_sent = tf.convert_to_tensor(tf.constant([0]), dtype=tf.int32)
        i0 = tf.constant(0)
        

        @tf.function
        def backward_qa_loop(i, rank_qa_input_ids, rank_qa_input_mask, rank_qa_input_type_mask, start_in_sent, end_in_sent,):
            
            query_tokens, t_start_in_sent, t_end_in_sent = self.get_question_token_ids(
                self.input_ids, self.input_mask, self.sentence_map, tf.gather(topc_start_index_doc, i), tf.gather(topc_end_index_doc, i))
        
            ## question_len = util.shape(query_tokens, 0)
            ## if question_len < self.config["max_query_len"]:
            pad_tokens = tf.zeros([self.config["max_query_len"] - util.shape(query_tokens, 0)], dtype=tf.int32)
            pad_query_tokens = tf.concat([query_tokens, pad_tokens], axis=0)
            ### else:
            ###    pad_query_tokens = tf.gather(query_tokens, tf.range(0, self.config["max_query_len"]))

            pad_query_tokens = tf.cast(pad_query_tokens, tf.int32) 
            # pad_query_tokens = tf.keras.preprocessing.sequence.pad_sequences(query_tokens, maxlen=self.config["max_query_len"],
            #     padding="post", truncating="post")
            query_input_token_type_mask = tf.zeros_like(pad_query_tokens, dtype=tf.int32)
            query_input_mask = tf.ones_like(pad_query_tokens, dtype=tf.int32)

            context_tokens, k_start_in_sent, k_end_in_sent = self.get_question_token_ids(
                self.input_ids, self.input_mask, self.sentence_map, tf.gather(tile_top_span_starts, i), tf.gather(tile_top_span_ends, i), special=False)

            pad_tokens = tf.zeros([self.config["max_query_len"] - util.shape(context_tokens, 0)], dtype=tf.int32)
            pad_context_tokens = tf.concat([context_tokens, pad_tokens], axis=0)
            ### else:
            ###    pad_context_tokens = tf.gather(context_tokens, tf.range(0, self.config["max_query_len"]))

            pad_context_tokens = tf.cast(pad_context_tokens, tf.int32)
            # pad_context_tokens = tf.keras.preprocessing.sequence.pad_sequences(context_tokens, maxlen=self.config["max_context_len"],
            #     padding="post", truncating="post")
            # k_start_in_sent, k_end_in_sent clip???? 因为后面要截断句子的长度
            context_input_mask = tf.ones_like(pad_context_tokens, dtype=tf.int32)
            context_input_token_type_mask = tf.ones_like(pad_context_tokens, dtype=tf.int32)

            qa_input_tokens = tf.concat([pad_query_tokens, pad_context_tokens], axis=-1)
            qa_input_mask = tf.concat([query_input_mask, context_input_mask], axis=-1)
            qa_input_token_type_mask = tf.concat([query_input_token_type_mask, context_input_token_type_mask], -1)

            qa_input_tokens = tf.cast(tf.reshape(qa_input_tokens, [1, self.config["max_query_len"]+self.config["max_context_len"]]), tf.int32)
            qa_input_mask = tf.cast(tf.reshape(qa_input_mask, [1, self.config["max_query_len"]+self.config["max_context_len"]]), tf.int32)
            qa_input_token_type_mask = tf.cast(tf.reshape(qa_input_token_type_mask, [1, self.config["max_query_len"]+self.config["max_context_len"]]), tf.int32)
            k_start_in_sent = tf.convert_to_tensor(tf.cast(k_start_in_sent, tf.int32), dtype=tf.int32) 
            k_end_in_sent = tf.convert_to_tensor(tf.cast(k_end_in_sent, tf.int32), dtype=tf.int32) 

            rank_qa_input_ids = tf.cast(rank_qa_input_ids, tf.int32)
            rank_qa_input_mask = tf.cast(rank_qa_input_mask, tf.int32)
            rank_qa_input_type_mask = tf.cast(rank_qa_input_type_mask, tf.int32)
            start_in_sent = tf.cast(start_in_sent, tf.int32)
            end_in_sent = tf.cast(end_in_sent, tf.int32)

            return (i+1, tf.concat([rank_qa_input_ids, qa_input_tokens],axis=0),
                tf.concat([rank_qa_input_mask, qa_input_mask], axis=0), 
                tf.concat([rank_qa_input_type_mask, qa_input_token_type_mask], axis=0), 
                tf.concat([start_in_sent, k_start_in_sent], axis=0), 
                tf.concat([end_in_sent, k_end_in_sent], axis=0))

        _, batch_backward_input_ids, batch_backward_input_mask, batch_backward_token_type_mask, batch_backward_start_sent, batch_backward_end_sent = tf.while_loop(
            cond = lambda i, o1, o2, o3, o4, o5: i < k * c,
            body=backward_qa_loop, 
            loop_vars=[i0, backward_qa_input_ids, backward_qa_input_mask, backward_qa_input_token_type, tmp_start_in_sent, tmp_end_in_sent], 
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
            tf.TensorShape([None]), tf.TensorShape([None])])

        self.batch_backward_start_sent = tf.gather(batch_backward_start_sent, tf.range(1, k * c+1))
        self.batch_backward_end_sent = tf.gather(batch_backward_end_sent, tf.range(1, k*c+1)) 

        backward_bert_qa_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=batch_backward_input_ids, input_mask=batch_backward_input_mask, 
            token_type_ids=batch_backward_token_type_mask, use_one_hot_embeddings=False, 
            scope="backward_qa_linking")

        backward_qa_emb = backward_bert_qa_model.get_sequence_output() # (c*k, num_ques_token+ max_context_len, embedding)
        # 1. (c*k
        backward_qa_input_token_type_mask_bool = tf.cast(batch_backward_token_type_mask ,tf.bool)
        # backward_qa_input_token_type_mask_bool = tf.tile(tf.expand_dims(backward_qa_input_token_type_mask_bool, 2), [1, 1, self.config["hidden_size"]])
        backward_k_sent_emb = mask.boolean_mask(backward_qa_emb, backward_qa_input_token_type_mask_bool, use_tpu=self.config["tpu"])
        # backward_k_sent_emb -> (c*k, max_context_len, embedding)

        backward_k_sent_emb =  tf.reshape(backward_k_sent_emb, [-1, self.config["hidden_size"]]) 

        backward_pos_offset = tf.cast(tf.reshape(tf.range(0, k*c) * self.config["max_context_len"], [-1, 1]), tf.int32) 
        # #forward_pos_offset = tf.cast(tf.tile(tf.reshape(tf.range(0, k) * non_overlap_doc_len, [-1, 1]), [1, k]), tf.int32)


        ##### batch_backward_start_sent = tf.reshape(self.batch_backward_start_sent, [c*k]) + tf.reshape(backward_pos_offset, [-1])
        ##### batch_backward_end_sent = tf.reshape(self.batch_backward_end_sent, [c*k]) + tf.reshape(backward_pos_offset, [-1])
        batch_backward_start_sent = tf.reshape(self.batch_backward_start_sent, [-1]) + tf.reshape(backward_pos_offset, [-1])
        batch_backward_end_sent = tf.reshape(self.batch_backward_end_sent, [-1]) + tf.reshape(backward_pos_offset, [-1])


        backward_qa_start_emb = tf.gather(backward_k_sent_emb, tf.reshape(batch_backward_start_sent, [-1])) # (c*k, emb)
        backward_qa_end_emb = tf.gather(backward_k_sent_emb, tf.reshape(batch_backward_end_sent, [-1]))  # (c*k, emb)
        backward_qa_span_emb = tf.concat([backward_qa_start_emb,backward_qa_end_emb], axis=1) # (c*k, 2*emb)

        #### 这里也需要向forward的一样加上当前位置的test
        #### 需要加上test


        with tf.variable_scope("backward_qa",):
            backard_mention_ji_score = util.ffnn(tf.reshape(backward_qa_span_emb, [-1, self.config["hidden_size"]*2]), 1, self.config["hidden_size"]*2, 1, self.dropout)
        # inputs, num_hidden_layers, hidden_size, output_size, dropout,
        # s(j) topc_span_scores # (k*c)
        # s(i) top_span_mention_scores # k
        # topc_forward_scores # (k*c)
        # backard_mention_ji_score # (k*c) 

        tile_top_span_mention_scores = tf.tile(tf.expand_dims(tf.reshape(top_span_mention_scores, [-1]), 1), [1, c])
        
        # [2, 2] [16, 1]
        top_antecedent_scores = (tf.reshape(topc_forward_scores, [-1]) + tf.reshape(backard_mention_ji_score, [-1]) ) / 2.0 * self.config["score_ratio"] \
            + (1 - self.config["score_ratio"]) * (tf.reshape(topc_span_scores, [-1]) + tf.reshape(tile_top_span_mention_scores, [-1]))

        top_antecedent_scores = tf.reshape(top_antecedent_scores, [k, c])

        dummy_scores = tf.zeros([k, 1])  # [k, 1]

        ###############################################################################################################################################
        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]
        # top_antecedent_cluster_ids [k, c] 每个mention每个antecedent的cluster_id
        # same_cluster_indicator [k, c] 每个mention跟每个预测的antecedent是否同一个cluster
        # pairwise_labels [k, c] 用pairwise的方法得到的label，非mention、非antecedent都是0，mention跟antecedent共指是1
        # top_antecedent_labels [k, c+1] 最终的标签，如果某个mention没有antecedent就是dummy_label为1
        # top_antecedent_scores = 
        

        topc_span_cluster_ids = tf.reshape(topc_span_cluster_ids, [k, c])
        same_cluster_indicator = tf.equal(topc_span_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # (k, c)
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]

        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        

        loss = self.marginal_likelihood_loss(top_antecedent_scores, top_antecedent_labels)  # [k]

        ######## loss += mention_proposal_loss * self.config["mention_proposal_loss_ratio"]

        # return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
        #         topc_forward_antecedent, top_antecedent_scores], loss
        ####################################################################################################################################################
        return loss

        ############################################################################################################################################

        # forward_mention_ij_score = util.ffnn(forward_mention_span_emb, self.config["ffnn_depth"], self.config["ffnn_size"]*2, 1, self.dropout)


        # forward_qa_input_token_type_mask_bool = tf.cast(tf.reshape(forward_qa_input_token_type_mask, [-1, self.config["max_query_len"] + self.config["max_segment_len"]]), tf.bool)
        # forward_qa_input_token_type_mask_bool = tf.tile(tf.expand_dims(forward_qa_input_token_type_mask_bool, 2), [1, 1, self.config["hidden_size"]])
        
        # 找到topC个在原来文本中的index 

        # 先把query 和 context mask掉，然后再按照sentence map把之前的东西拿出来
        # 最后出来的是: k, max_training_sentences, max_query_len + max_segment_len 
        ############################################################################################################


    def flatten_emb_by_sentence(self, emb, segment_overlap_mask):
        """
        Desc:
            flatten_embeddings_by_sentence_segment_mask
        Args:
            emb: [max_sentence_len, max_segment_len] 
            segment_overlap_mask:  [max_sentence_len, max_segment_len]
        """
        flattened_emb = tf.reshape(emb, [-1, self.config["hidden_size"]])
        flattened_overlap_mask = tf.reshape(segment_overlap_mask, [-1])
        segment_overlap_mask = tf.maximum(segment_overlap_mask, tf.zeros_like(segment_overlap_mask))
        segment_overlap_mask = tf.reshape(segment_overlap_mask, [-1])
        
        # flattened_emb = mask.boolean_mask(flattened_emb, segment_overlap_mask, use_tpu=self.config["tpu"])

        return flattened_emb, flattened_overlap_mask 


    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        """
        Desc:
            pass 
        Args:
            candidate_starts/candidate_ends: 
            labeled_starts/labeled_ends: 
            labels: 
        """
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) 
        same_span = tf.logical_and(same_start, same_end)
        # candidate_labels: [num_candidates] 预测对的candidate标上正确的cluster_id，预测错的标0
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels # 每个候选答案得到真实标注的cluster_id


    def get_span_emb(self, context_outputs, span_starts, span_ends):
        """
        一个span的表示由下面的组成
        span_start_embedding, span_end_embedding, span_with_embedding, head_attention_representation 
        """

        span_emb_list = []
        context_outputs = tf.reshape(context_outputs, [-1, self.config["hidden_size"]])

        span_end_emb = tf.gather(context_outputs, tf.reshape(span_ends, [-1]))
        span_start_emb = tf.gather(context_outputs, tf.reshape(span_starts, [-1])) # [k, emb]
        span_emb_list.append(span_start_emb)
        # span_end_emb = tf.gather(context_outputs, tf.reshape(span_ends, [-1]))
        span_emb_list.append(span_end_emb)
        
        span_width = 1 + span_ends - span_starts # [k]

        if self.config["use_features"]:
            span_width_index = span_width -1 # [k]
            with tf.variable_scope("span_features",):
                span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index)  # [k, emb]
                span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
                span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_reps = tf.matmul(mention_word_scores, context_outputs) # [k, t]
            span_emb_list.append(head_attn_reps)

        span_emb = tf.concat([span_start_emb, span_end_emb], 1) # [k, emb] origin span_emb_list 
        span_emb = tf.reshape(span_emb, [-1, self.config["hidden_size"]*2])
        return span_emb # [k, emb]

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = util.shape(encoded_doc, 0) # T 
        num_c = util.shape(span_starts, 0)
        doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1]) # [num_candidate, num_words]
        mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1), 
            doc_range <= tf.expand_dims(span_ends, 1)) # [num_candidates, num_word]


        with tf.variable_scope("mention_word_attn",):
            word_attn = tf.squeeze(
                util.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)

        mention_word_attn = tf.nn.softmax(tf.log(tf.to_float(mention_mask)) + tf.expand_dims(word_attn, 0))
        return mention_word_attn  # [num_candidates, num_words] 


    def get_mention_scores(self, span_emb, span_starts, span_ends):
        with tf.variable_scope("mention_scores", ):
            span_scores = util.ffnn(span_emb, 1, self.config["hidden_size"]*2, 1, self.dropout)
        return   tf.squeeze(span_scores, 1)
        # def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout,
        #  output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        #  hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):


    def get_question_token_ids(self, input_ids, flat_input_mask, sentence_map, top_start, top_end, special=True):
        """
        Desc:
            construct question based on the selected mention 
        Args:
            sentence_map: original sentence_map 
            top_start: start index in non-overlap document 
            top_end: end index in non-overlap document 
        """

        nonoverlap_sentence = tf.where(tf.cast(tf.math.greater_equal(sentence_map, tf.zeros_like(sentence_map)),tf.bool), x=sentence_map, y=tf.zeros_like(sentence_map)) 

        nonoverlap_sentence = mask.boolean_mask(tf.reshape(input_ids, [-1]), tf.reshape(nonoverlap_sentence, [-1]), use_tpu=self.config["tpu"])

        flat_sentence_map = tf.reshape(sentence_map, [-1])

        sentence_idx = tf.gather(nonoverlap_sentence, top_start)

        query_sentence_mask = tf.math.equal(flat_sentence_map, sentence_idx)
        input_ids = tf.reshape(input_ids, [-1])
        query_sentence_tokens = mask.boolean_mask(input_ids, query_sentence_mask, use_tpu=self.config["tpu"])
        len_query_tokens = util.shape(query_sentence_tokens, 0)

        sentence_start = tf.where(tf.equal(nonoverlap_sentence, tf.gather(query_sentence_tokens, tf.constant(0))))
        # sentence_end = tf.where(tf.equal(nonoverlap_sentence, tf.gather(query_sentence_tokens, len_query_tokens -1 ))) 
        ############### mention_start = tf.where(tf.equal(nonoverlap_sentence, tf.gather(nonoverlap_sentence, top_start)))
        ############### mention_end = tf.where(tf.equal(nonoverlap_sentence, tf.gather(nonoverlap_sentence, top_end)))
        ##### should be flat_input_mask 
        mention_start = tf.cast(top_start, tf.int32) 
        mention_end = tf.cast(top_end, tf.int32) 

        original_tokens = query_sentence_tokens

        start_in_sent = mention_start - tf.cast(sentence_start, tf.int32) 
        end_in_sent = mention_end - tf.cast(sentence_start, tf.int32) 

        start_in_sent = tf.gather(start_in_sent, tf.constant(0))
        end_in_sent = tf.gather(end_in_sent, tf.constant(0))

        return original_tokens, tf.reshape(start_in_sent, [-1]), tf.reshape(end_in_sent, [-1]) 
        #####num_token = util.shape(original_tokens, 0)
        #######left_sentence = tf.gather(original_tokens, tf.range(0, start_in_sent))
        # left_sentence = tf.cast(original_tokens[: start_in_sent], tf.int32)
        #####mid_sentence = tf.gather(original_tokens, tf.range(start_in_sent, end_in_sent+1))
        ###### right_sentence = tf.gather(original_tokens, tf.range(end_in_sent+1, num_token))
        ######original_tokens = tf.concat([tf.cast(left_sentence, tf.int32), 
        ######    tf.cast(mid_sentence, tf.int32), 
        ######    tf.cast(right_sentence, tf.int32)
        ######    ], 0)
        ######return original_tokens, start_in_sent, end_in_sent



        ######mention_start_in_sentence = mention_start - sentence_start
        ######mention_end_in_sentence = mention_end - sentence_end

        ######mention_start_in_sentence = tf.reshape(tf.cast(mention_start_in_sentence, tf.int32), [-1])
        ######mention_end_in_sentence = tf.reshape(tf.cast(mention_end_in_sentence, tf.int32), [-1])

        ######if special:
        ######    # 补充上special token， 注意start end应该按照这个向后移动一步
        ######    len_sent = util.shape(original_tokens, 0)
        ######    before_sent = tf.gather(original_tokens, tf.range(0, mention_start_in_sentence[0]))
        ######    mid_sent = tf.gather(original_tokens, tf.range(mention_start_in_sentence[0], mention_end_in_sentence[0] + 1))
        ######    end_sent = tf.gather(original_tokens, tf.range(mention_end_in_sentence[0] + 1, len_sent))



        ######    # question_token_ids = tf.concat([tf.cast(original_tokens[: mention_start_in_sentence], tf.int32),
        ######   #                             [tf.cast(self.mention_start_idx, tf.int32)],
        ######    #                             tf.cast(original_tokens[mention_start_in_sentence: mention_end_in_sentence + 1], tf.int32),
        ######    #                             [tf.cast(self.mention_end_idx, tf.int32)],
        ######    #                             tf.cast(original_tokens[mention_end_in_sentence + 1:], tf.int32),
        ######    #                             ], 0)

        ######    question_token_ids = tf.concat([before_sent,mid_sent,end_sent], 0)
        ######    return question_token_ids, mention_start_in_sentence , mention_end_in_sentence # + 1
        ######else:
        ######    question_token_ids = original_tokens 
        ######    return question_token_ids, mention_start_in_sentence, mention_end_in_sentence  


    def get_mention_proposal_loss(self, span_scores, span_mention, start_pos, end_pos, span_mention_loss_mask=None):

        span_mention = tf.reshape(span_mention, [-1])

        if span_mention_loss_mask is None:
            span_mention_loss_mask = tf.reshape(tf.ones_like(span_mention), [-1])

        span_scores = tf.cast(tf.reshape(span_scores, [-1]), tf.float32)
        span_scores = tf.stack([(1 - span_scores), span_scores], axis=-1)
        ##### span_mention = tf.cast(tf.one_hot(tf.reshape(span_mention, [-1]), 2, axis=-1),tf.float32)
        #  [1472,2] vs. [81920,2]

        ####### if start_pos is not None:
        #######    # start_pos : [1, ]
        #######    # end_pos : [1, ]
        #######    start_end_mask = tf.concat([tf.reshape(start_pos, [-1, 1]), tf.reshape(end_pos, [-1, 1])], -1)
        #######    span_mention = tf.reshape(span_mention, [-1,self.config["max_segment_len"],self.config["max_segment_len"]])
        #######    span_mention = tf.gather_nd(span_mention, start_end_mask)

        start_end_mask = tf.concat([tf.reshape(start_pos, [-1, 1]), tf.reshape(end_pos, [-1, 1])], 1)
        start_end_mask = tf.reshape(start_end_mask, [self.config["max_training_sentences"], -1, 2])
        span_mention = tf.reshape(span_mention, [-1,self.config["max_segment_len"],self.config["max_segment_len"]])
        span_mention = tf.gather_nd(span_mention, start_end_mask)


        span_mention = tf.cast(tf.one_hot(tf.reshape(span_mention, [-1]), 2, axis=-1),tf.float32)


        span_loss = tf.keras.losses.binary_crossentropy(span_mention, span_scores,)
        if span_mention_loss_mask is not None:
            span_loss = tf.reduce_mean(tf.multiply(span_loss, tf.cast(span_mention_loss_mask, tf.float32)))
        else:
            span_loss = tf.reduce_mean(span_loss)
        
        
        return span_loss 

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
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return tf.reduce_sum(loss)

    def get_dropout(self, dropout_rate, is_training):  # is_training为True时keep=1-drop, 为False时keep=1
        
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def get_top_span_cluster_ids(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels, top_span_indices):

        """
        method to get top_span_cluster_ids
        :param candidate_starts: [num_candidates, ]
        :param candidate_ends: [num_candidates, ]
        :param labeled_starts: [num_mentions, ]
        :param labeled_ends: [num_mentions, ]
        :param labels: [num_mentions, ] gold truth cluster ids
        :param top_span_indices: [k, ]
        :return: [k, ] ground truth cluster ids for each proposed candidate span
        """
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0))
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates] predict_i == label_j

        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        top_span_cluster_ids = tf.gather(candidate_labels, top_span_indices)
        return top_span_cluster_ids





















