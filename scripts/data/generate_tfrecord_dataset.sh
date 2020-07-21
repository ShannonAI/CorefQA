#!/usr/bin/env bash 
# -*- coding: utf-8 -*-



# author: xiaoy li
# description:
# generate train/dev/test tfrecord files for training the model. 
# example:
# bash generate_tfrecord_dataset.sh /path-to-conll-coreference-resolution-dataset /path-to-save-tfrecord-for-training /cased_L-12_H-768_A-12/vocab.txt



REPO_PATH=/home/lixiaoya/coref-tf
export PYTHONPATH=$REPO_PATH

source_dir=$1
target_dir=$2
vocab_file=$3

mkdir -p ${target_dir}


python3 ${REPO_PATH}/run/build_dataset_to_tfrecord.py \
--source_files_dir $source_dir \
--target_output_dir $target_dir \
--num_window 2 \
--window_size 64 \
--max_num_mention 50 \
--max_num_cluster 40 \
--vocab_file $vocab_file \
--language english \
--max_doc_length 600 