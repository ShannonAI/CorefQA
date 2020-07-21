#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# finetune the spanbert model on squad 2.0 for data augment.  



REPO_PATH=/home/shannon/coref-tf
export TPU_NAME=tf-tpu
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
QUOREF_DIR=gs://qa_tasks/quoref
BERT_DIR=gs://corefqa_output_squad/panbert_large_squad2_2e-5
OUTPUT_DIR=gs://corefqa_output_quoref/spanbert_large_squad2_best_quoref_3e-5 


python3 ${REPO_PATH}/run_quoref.py \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$BERT_DIR/best_bert_model.ckpt \
--do_train=True \
--train_file=$QUOREF_DIR/quoref-train-v0.1.json \
--do_predict=True \
--predict_file=$QUOREF_DIR/quoref-dev-v0.1.json \
--train_batch_size=8 \
--learning_rate=3e-5 \
--num_train_epochs=5 \
--max_seq_length=384 \
--do_lower_case=False \
--doc_stride=128 \
--output_dir=${OUTPUT_DIR} \
--use_tpu=True \
--tpu_name=$TPU_NAME