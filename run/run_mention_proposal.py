#!/usr/bin/env python3
# -*- coding: utf-8 -*- 


"""
this file contains pre-training and testing the mention proposal model
"""

import os 
import math 
import random 
import logging
import numpy as np 
import tensorflow as tf
from data_utils.config_utils import ModelConfig
from func_builders.model_fn_builder import model_fn_builder 
from func_builders.input_fn_builder import file_based_input_fn_builder
from utils.metrics import mention_proposal_prediction

tf.app.flags.DEFINE_string('f', '', 'kernel')
flags = tf.app.flags

flags.DEFINE_string("output_dir", "data", "The output directory of the model training.")
flags.DEFINE_string("bert_config_file", "/home/uncased_L-2_H-128_A-2/config.json", "The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string("init_checkpoint", "/home/uncased_L-2_H-128_A-2/bert_model.ckpt", "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("vocab_file", "/home/uncased_L-2_H-128_A-2/vocab.txt", "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("logfile_path", "/home/lixiaoya/spanbert_large_mention_proposal.log", "the path to the exported log file.")
flags.DEFINE_integer("num_epochs", 20, "Total number of training epochs to perform.")
flags.DEFINE_integer("keep_checkpoint_max", 30, "How many checkpoint models keep at most.")
flags.DEFINE_integer("save_checkpoints_steps", 500, "Save checkpoint every X updates steps.")


flags.DEFINE_string("train_file", "/home/lixiaoya/train.english.tfrecord", "TFRecord file for training. E.g., train.english.tfrecord")
flags.DEFINE_string("dev_file", "/home/lixiaoya/dev.english.tfrecord", "TFRecord file for validating. E.g., dev.english.tfrecord")
flags.DEFINE_string("test_file", "/home/lixiaoya/test.english.tfrecord", "TFRecord file for testing. E.g., test.english.tfrecord")


flags.DEFINE_bool("do_train", True, "Whether to train a model.")
flags.DEFINE_bool("do_eval", False, "whether to do evaluation: evaluation is done on a set of trained checkpoints, the checkpoint with the best score on the dev set will be selected.")
flags.DEFINE_bool("do_predict", False, "Whether to test (only) one trained model.")
flags.DEFINE_string("eval_checkpoint", "/home/lixiaoya/mention_proposal_output_dir/bert_model.ckpt", "[Optional] The saved checkpoint for evaluation (usually after the training process).")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")


flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate for the training process.")
flags.DEFINE_float("mention_threshold", 0.5, "The threshold for determining whether the span is a mention.")
flags.DEFINE_integer("hidden_size", 128, "The size of hidden layers for the pre-trained model.")
flags.DEFINE_integer("num_docs", 5604, "[Optional] The number of documents in the training files. Only need to change when conduct experiments on the small test sets.")
flags.DEFINE_integer("window_size", 384, "The number of sliding window size. Each document is split into a set of subdocuments with length set to window_size.")
flags.DEFINE_integer("num_window", 5, "The max number of windows for one document. This is used for fitting a document into fix shape for TF computation. \
    If a document is longer than num_window*window_size, the exceeding part will be abandoned. This only affects training and does not affect test, since the all \
    docs in the test set is shorter than num_window*window_size")
flags.DEFINE_integer("max_num_mention", 30, "The max number of mentions in one document.")
flags.DEFINE_bool("start_end_share", False, "Whether only to use [start, end] embedding to calculate the start/end scores.") 
flags.DEFINE_float("loss_start_ratio", 0.3, "As described in the paper, the loss for a span being a mention is -loss_start_ratio* log p(the start of the given span is a start).")
flags.DEFINE_float("loss_end_ratio", 0.3, "As described in the paper, the loss for a span being a mention is -loss_end_ratio* log p(the end of the given span is a end).")
flags.DEFINE_float("loss_span_ratio", 0.4, "As described in the paper, the loss for a span being a mention is -loss_span_ratio* log p(the start and the end forms a span).")


flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("num_tpu_cores", 1, "[Optional] Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_integer("seed", 2333, "[Optional] Random seed for initialization.")
FLAGS = tf.flags.FLAGS



format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format, filename=FLAGS.logfile_path, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    num_train_steps = FLAGS.num_docs * FLAGS.num_epochs
    # num_train_steps = 100 
    keep_chceckpoint_max = max(math.ceil(num_train_steps / FLAGS.save_checkpoints_steps), FLAGS.keep_checkpoint_max)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        # evaluation_master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max = keep_chceckpoint_max,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        # session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))


    model_config = ModelConfig(FLAGS, FLAGS.output_dir)
    model_config.logging_configs()

    model_fn = model_fn_builder(model_config, model_sign="mention_proposal")
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        # eval_on_tpu=FLAGS.use_tpu,
        warm_start_from=tf.estimator.WarmStartSettings(FLAGS.init_checkpoint,
            vars_to_warm_start="bert*"),
        model_fn=model_fn,
        config=run_config,
        train_batch_size=1,
        predict_batch_size=1)


    if FLAGS.do_train:
        estimator.train(input_fn=file_based_input_fn_builder(model_config.train_file, num_window=model_config.num_window,
            window_size=model_config.window_size, max_num_mention=model_config.max_num_mention, is_training=True, drop_remainder=True), max_steps=num_train_steps)


    if FLAGS.do_eval:
        # doing evaluation  on a set of trained checkpoints, the checkpoint with the best score on the dev set will be selected.
        best_dev_f1, best_dev_prec, best_dev_rec, test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = 0, 0, 0, 0, 0, 0
        best_ckpt_path = ""
        checkpoints_iterator = [os.path.join(FLAGS.eval_dir, "model.ckpt-{}".format(str(int(ckpt_idx)))) for ckpt_idx in range(0, num_train_steps, FLAGS.save_checkpoints_steps)]
        for checkpoint_path in checkpoints_iterator[1:]:
            eval_dev_result = estimator.evaluate(input_fn=file_based_input_fn_builder(FLAGS.dev_file, num_window=FLAGS.num_window, 
                window_size=FLAGS.window_size, max_num_mention=FLAGS.max_num_mention, is_training=False, drop_remainder=False),
                steps=698, checkpoint_path=checkpoint_path)
            dev_f1 = 2*eval_dev_result["precision"] * eval_dev_result["recall"] / (eval_dev_result["precision"] + eval_dev_result["recall"]+1e-10)
            tf.logging.info("***** Current ckpt path is ***** : {}".format(checkpoint_path))
            tf.logging.info("***** EVAL ON DEV SET *****")
            tf.logging.info("***** [DEV EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(eval_dev_result["precision"], eval_dev_result["recall"], dev_f1))
            if dev_f1 > best_dev_f1:
                best_dev_f1, best_dev_prec, best_dev_rec = dev_f1, eval_dev_result["precision"], eval_dev_result["recall"]
                best_ckpt_path = checkpoint_path
                eval_test_result = estimator.evaluate(input_fn=file_based_input_fn_builder(FLAGS.test_file, 
                    num_window=FLAGS.num_window, window_size=FLAGS.window_size, max_num_mention=FLAGS.max_num_mention, 
                    is_training=False, drop_remainder=False),steps=698, checkpoint_path=checkpoint_path)
                test_f1 = 2*eval_test_result["precision"] * eval_test_result["recall"] / (eval_test_result["precision"] + eval_test_result["recall"]+1e-10)
                test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = test_f1, eval_test_result["precision"], eval_test_result["recall"]
                tf.logging.info("***** EVAL ON TEST SET *****")
                tf.logging.info("***** [TEST EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(eval_test_result["precision"], eval_test_result["recall"], test_f1))
        tf.logging.info("*"*20)
        tf.logging.info("- @@@@@ the path to the BEST DEV result is : {}".format(best_ckpt_path))
        tf.logging.info("- @@@@@ BEST DEV F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(best_dev_f1, best_dev_prec, best_dev_rec))
        tf.logging.info("- @@@@@ TEST when DEV best F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best))
        tf.logging.info("- @@@@@ mention_proposal_only_concate {}".format(FLAGS.mention_proposal_only_concate))


    if FLAGS.do_predict:
        tp, fp, fn = 0, 0, 0
        epsilon = 1e-10
        for doc_output in estimator.predict(file_based_input_fn_builder(FLAGS.test_file,
            num_window=FLAGS.num_window, window_size=FLAGS.window_size, max_num_mention=FLAGS.max_num_mention,
            is_training=False, drop_remainder=False), checkpoint_path=FLAGS.eval_checkpoint, yield_single_examples=False): 
            # iterate over each doc for evaluation
            pred_span_label, gold_span_label = mention_proposal_prediction(FLAGS, doc_output)

            tem_tp = np.logical_and(pred_span_label, gold_span_label).sum()
            tem_fp = np.logical_and(pred_span_label, np.logical_not(gold_span_label)).sum()
            tem_fn = np.logical_and(np.logical_not(pred_span_label), gold_span_label).sum()

            tp += tem_tp
            fp += tem_fp
            fn += tem_fn

        p = tp / (tp+fp+epsilon)
        r = tp / (tp+fn+epsilon)
        f = 2*p*r/(p+r+epsilon)
        tf.logging.info("Average precision: {:.4f}, Average recall: {:.4f}, Average F1 {:.4f}".format(p, r, f))



if __name__ == '__main__':
    # set the random seed. 
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    # start train/evaluate the model.
    tf.app.run()






