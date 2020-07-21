#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# train and evaluate the middle checkpoints on dev and test sets. 



REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

output_dir=/xiaoya/mention_proposal_output
bert_dir=/xiaoya/pretrain_ckpt/uncased_L-2_H-128_A-2
data_dir=/xiaoya/corefqa_data/final_overlap_64_2


rm -rf ${output_dir}
mkdir -p ${output_dir}



CUDA_VISIBLE_DEVICES=3 python3 ${REPO_PATH}/run/run_corefqa.py \
--output_dir=${output_dir} \
--bert_config_file=${bert_dir}/bert_config_nodropout.json \
--init_checkpoint=${bert_dir}/bert_model.ckpt \
--vocab_file=${bert_dir}/vocab.txt \
--logfile_path=${output_dir}/train.log \
--num_epochs=20 \
--keep_checkpoint_max=50 \
--save_checkpoints_steps=500 \
--train_file=${data_dir}/train.64.english.tfrecord \
--dev_file=${data_dir}/dev.64.english.tfrecord \
--test_file=${data_dir}/test.64.english.tfrecord \
--do_train=True \
--do_eval=False \
--do_predict=False \
--learning_rate=1e-5 \
--dropout_rate=0.0 \
--mention_threshold=0.5 \
--hidden_size=128 \
--num_docs=5604 \
--window_size=64 \
--num_window=2 \
--max_num_mention=20 \
--start_end_share=False \
--max_span_width=20 \
--max_candidate_mentions=50 \
--top_span_ratio=0.2 \
--max_top_antecedents=30 \
--max_query_len=150 \
--max_context_len=150 \
--sec_qa_mention_score=False \
--use_tpu=False \
--seed=2333



