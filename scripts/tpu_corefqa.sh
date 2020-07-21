#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



REPO_PATH=/home/xiaoyli1110/xiaoya/Coref-tf
export PYTHONPATH="$PYTHONPATH:/home/xiaoyli1110/xiaoya/Coref-tf"
export TPU_NAME=tensorflow-tpu
# export TPU_NAME=tf-tpu
GCP_PROJECT=xiaoyli-20-04-274510
OUTPUT_DIR=gs://corefqa/span_all_128_5_output_bertlarge



python3 ${REPO_PATH}/run/train_corefqa.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--use_tpu=True \
--iterations_per_loop=500 \
--tpu_name=${TPU_NAME} \
--tpu_zone=us-central1-f \
--gcp_project=${GCP_PROJECT} \
--num_tpu_cores=1 # > ${OUTPUT_DIR}/log.txt