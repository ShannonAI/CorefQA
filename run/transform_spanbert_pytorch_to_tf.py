#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# transform pytorch .bin models to tensorflow ckpt 


import os
import sys  
import shutil 
import torch
import argparse 
import random 
import numpy as np 
import tensorflow as tf  

REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


from bert import modeling 
from utils.load_pytorch_to_tf import load_from_pytorch_checkpoint 


def load_models(bert_config_path, ):
    bert_config = modeling.BertConfig.from_json_file(bert_config_path)
    input_ids = tf.ones((8, 128), tf.int32)

    model = modeling.BertModel(
        config=bert_config,
        is_training=False, 
        input_ids=input_ids,
        use_one_hot_embeddings=False, 
        scope="bert")

    return model, bert_config 


def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)


def main(bert_config_path, bert_ckpt_path, pytorch_init_checkpoint, output_tf_dir):

    with tf.Session() as session:
        model, bert_config = load_models(bert_config_path)
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, bert_ckpt_path)
        session.run(tf.global_variables_initializer())
        init_from_checkpoint = load_from_pytorch_checkpoint
        init_from_checkpoint(pytorch_init_checkpoint, assignment_map)

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                print("name = %s, shape = %s%s" % (var.name, var.shape, init_string))
        
        saver = tf.train.Saver()
        saver.save(session, os.path.join(output_tf_dir, "model"), global_step=100)
        copy_checkpoint(os.path.join(output_tf_dir, "model-{}".format(str(100))), os.path.join(output_tf_dir, "bert_model.ckpt"))
        print("=*="*30)
        print("save models : {}".format(output_tf_dir))
        print("=*="*30)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spanbert_config_path", default="/home/lixiaoya/spanbert_base_cased/config.json", type=str)
    parser.add_argument("--bert_tf_ckpt_path", default="/home/lixiaoya/cased_L-12_H-768_A-12/bert_model.ckpt", type=str)
    parser.add_argument("--spanbert_pytorch_bin_path", default="/home/lixiaoya/spanbert_base_cased/pytorch_model.bin", type=str)
    parser.add_argument("--output_spanbert_tf_dir", default="/home/lixiaoya/tf_spanbert_base_case", type=str)
    parser.add_argument("--seed", default=2333, type=int)


    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.set_random_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_spanbert_tf_dir, exist_ok=True)

    try:
        shutil(args.spanbert_config_path, args.output_spanbert_tf_dir)
    except:
        print("#=#"*30)
        print("copy spanbert_config from {} to {}".format(args.spanbert_config_path, args.output_spanbert_tf_dir))

    return args


if __name__ == "__main__":

    args_config = parse_args()

    main(args_config.spanbert_config_path, args_config.bert_tf_ckpt_path, args_config.spanbert_pytorch_bin_path, args_config.output_spanbert_tf_dir)

    # 
    # Please refer to scripts/data/transform_ckpt_pytorch_to_tf.sh 
    # 

    # for spanbert large 
    # 
    # python3 transform_spanbert_pytorch_to_tf.py \
    # --spanbert_config_path /xiaoya/pretrain_ckpt/spanbert_large_cased/config.json \
    # --bert_tf_ckpt_path /xiaoya/pretrain_ckpt/cased_L-24_H-1024_A-16/bert_model.ckpt \
    # --spanbert_pytorch_bin_path /xiaoya/pretrain_ckpt/spanbert_large_cased/pytorch_model.bin \
    # --output_spanbert_tf_dir /xiaoya/pretrain_ckpt/tf_spanbert_large_cased


    # for spanbert base 
    # 
    # python3 transform_spanbert_pytorch_to_tf.py \
    # --spanbert_config_path /xiaoya/pretrain_ckpt/spanbert_base_cased/config.json \
    # --bert_tf_ckpt_path /xiaoya/pretrain_ckpt/cased_L-12_H-768_A-12/bert_model.ckpt \
    # --spanbert_pytorch_bin_path /xiaoya/pretrain_ckpt/spanbert_base_cased/pytorch_model.bin \
    # --output_spanbert_tf_dir /xiaoya/pretrain_ckpt/tf_spanbert_base_cased

