#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



import codecs
import collections
import errno
import os
import shutil
import pyhocon
import tensorflow as tf
from models import corefqa
from models import mention_proposal 


repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])


def get_model(config, model_sign="corefqa"): 
    if model_sign == "corefqa":
        return corefqa.CorefQAModel(config)
    else:
        return mention_proposal.MentionProposalModel(config)


def initialize_from_env(eval_test=False, config_params="train_spanbert_base", config_file="experiments_tinybert.conf", use_tpu=False, print_info=False):
    if not use_tpu:
        print("loading experiments.conf ... ")
        config = pyhocon.ConfigFactory.parse_file(os.path.join(repo_path, config_file)) 
    else: 
        print("loading experiments_tpu.conf ... ")
        config = pyhocon.ConfigFactory.parse_file(os.path.join(repo_path, config_file))

    config = config[config_params] 

    if print_info:
        tf.logging.info("%*%"*20)
        tf.logging.info("%*%"*20)
        tf.logging.info("%%%%%%%% Configs are showed as follows : %%%%%%%%")
        for tmp_key, tmp_value in config.items():
            tf.logging.info(str(tmp_key) + " : " + str(tmp_value)) 
    
        tf.logging.info("%*%"*20)
        tf.logging.info("%*%"*20)

    config["log_dir"] = mkdirs(os.path.join(config["log_root"], config_params))

    if print_info:
        tf.logging.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def flatten(l):
    return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
    # pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)



def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
         hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
    current_inputs = inputs
    hidden_weights = tf.get_variable("hidden_weights", [hidden_size, output_size],
                                         initializer=hidden_initializer)
    hidden_bias = tf.get_variable("hidden_bias", [output_size], initializer=tf.zeros_initializer())
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    return current_outputs


def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
    offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
    gathered = tf.gather(flattened_emb, indices + offset)  # [batch_size, num_indices, emb]
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered

