#!/usr/bin/env bash 
# -*- coding: utf-8 -*-



# author: xiaoy li
# description:
# transform trained spanbert language model from pytorch(.bin) to tensorflow(.ckpt). 
# PLEASE NOTICE: the same scale(Base/Large) BERT(TF) Models are also necessary. 



REPO_PATH=/home/lixiaoya/coref-tf
export PYTHONPATH=${REPO_PATH}


MODEL_NAME=$1
PATH_TO_SPANBERT_PYTORCH_DIR=$2
PATH_TO_SAME_SCALE_BERT_TF_DIR=$3
PATH_TO_SAVE_SPANBERT_TF_DIR=$4


if [[ $MODEL_NAME == "spanbert_base" ]]; then
    # spanbert large 
    echo "Transform SpanBERT Cased Base from Pytorch To TF"
    python3 ${REPO_PATH}/run/transform_spanbert_pytorch_to_tf.py \
        --spanbert_config_path $PATH_TO_SPANBERT_PYTORCH_DIR/config.json \
        --bert_tf_ckpt_path $PATH_TO_SAME_SCALE_BERT_TF_DIR/bert_model.ckpt \
        --spanbert_pytorch_bin_path $PATH_TO_SPANBERT_PYTORCH_DIR/pytorch_model.bin \
        --output_spanbert_tf_dir $PATH_TO_SAVE_SPANBERT_TF_DIR
elif [[ $MODEL_NAME == "spanbert_large" ]]; then
    # spanbert base 
    echo "Transform SpanBERT Cased Large from Pytorch To TF"
    python3 ${REPO_PATH}/run/transform_spanbert_pytorch_to_tf.py \
    --spanbert_config_path $PATH_TO_SPANBERT_PYTORCH_DIR/config.json \
    --bert_tf_ckpt_path $PATH_TO_SAME_SCALE_BERT_TF_DIR/bert_model.ckpt \
    --spanbert_pytorch_bin_path $PATH_TO_SPANBERT_PYTORCH_DIR/pytorch_model.bin \
    --output_spanbert_tf_dir $PATH_TO_SAVE_SPANBERT_TF_DIR
else
    echo 'Unknown argment 1 (Model Sign)'
fi 