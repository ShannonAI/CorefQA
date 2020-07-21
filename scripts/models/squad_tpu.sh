#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# finetune the spanbert model on squad 2.0 for data augment.  



REPO_PATH=/home/shannon/coref-tf
export TPU_NAME=tf-tpu
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
SQUAD_DIR=gs://qa_tasks/squad2
BERT_DIR=gs://pretrained_mlm_checkpoint/spanbert_large_tf
OUTPUT_DIR=gs://corefqa_output_squad/spanbert_large_squad2_2e-5  


python3 ${REPO_PATH}/run/run_squad.py \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$BERT_DIR/bert_model.ckpt \
--do_train=True \
--train_file=$SQUAD_DIR/train-v2.0.json \
--do_predict=True \
--predict_file=$SQUAD_DIR/dev-v2.0.json \
--train_batch_size=8 \
--learning_rate=2e-5 \
--num_train_epochs=4.0 \
--max_seq_length=384 \
--do_lower_case=False \
--doc_stride=128 \
--output_dir=${OUTPUT_DIR} \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--version_2_with_negative=True