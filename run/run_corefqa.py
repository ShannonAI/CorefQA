#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
this file contains training and testing the CorefQA model. 
"""

import os 
import math 
import logging
import random 
import numpy as np 
import tensorflow as tf
from utils import util
from utils import metrics
from data_utils.config_utils import ModelConfig
from func_builders.model_fn_builder import model_fn_builder 
from func_builders.input_fn_builder import file_based_input_fn_builder


tf.app.flags.DEFINE_string('f', '', 'kernel')
flags = tf.app.flags

flags.DEFINE_string("output_dir", "data", "The output directory.")
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
flags.DEFINE_bool("do_eval", False, "Whether to do evaluation: evaluation is done on a set of trained checkpoints, the model will select the best one on the dev set, and report result on the test set")
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
flags.DEFINE_integer("max_span_width", 5, "The max length of a mention.")
flags.DEFINE_integer("max_candidate_mentions", 30, "The number of candidate mentions.")
flags.DEFINE_float("top_span_ratio", 0.2, "The ratio of.")
flags.DEFINE_integer("max_top_antecedents", 30, "The number of top_antecedents candidate mentions.")
flags.DEFINE_integer("max_query_len", 150, ".")
flags.DEFINE_integer("max_context_len", 150, ".")
flags.DEFINE_bool("sec_qa_mention_score", False, "Whether to use TPU or GPU/CPU.")


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
        model_dir=FLAGS.output_dir,
        evaluation_master=FLAGS.master,
        keep_checkpoint_max = keep_chceckpoint_max,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))


    model_config = ModelConfig(FLAGS, FLAGS.output_dir)
    model_config.logging_configs()


    model_fn = model_fn_builder(model_config, model_sign="corefqa")
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        eval_on_tpu=FLAGS.use_tpu,
        warm_start_from=tf.estimator.WarmStartSettings(FLAGS.init_checkpoint,
            vars_to_warm_start="bert*"),
        model_fn=model_fn,
        config=run_config,
        train_batch_size=1,
        eval_batch_size=1,
        predict_batch_size=1)


    if FLAGS.do_train:
        estimator.train(input_fn=file_based_input_fn_builder(FLAGS.train_file, num_window=FLAGS.num_window,
            window_size=FLAGS.window_size, max_num_mention=FLAGS.max_num_mention, is_training=True, drop_remainder=True), 
            max_steps=num_train_steps)


    if FLAGS.do_eval:
        best_dev_f1, best_dev_prec, best_dev_rec, test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = 0, 0, 0, 0, 0, 0
        best_ckpt_path = ""
        checkpoints_iterator = [os.path.join(FLAGS.eval_dir, "model.ckpt-{}".format(str(int(ckpt_idx)))) for ckpt_idx in range(0, num_train_steps+1, FLAGS.save_checkpoints_steps)]
        model = util.get_model(model_config, model_sign="corefqa")
        for checkpoint_path in checkpoints_iterator[1:]:
            dev_coref_evaluator = metrics.CorefEvaluator()
            for result in estimator.predict(file_based_input_fn_builder(FLAGS.dev_file, num_window=FLAGS.num_window, 
                window_size=FLAGS.window_size, max_num_mention=FLAGS.max_num_mention, is_training=False, drop_remainder=False), 
                steps=698, checkpoint_path=checkpoint_path, yield_single_examples=False):
                
                predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(result["topk_span_starts"], result["topk_span_ends"], result["top_antecedent"],
                    result["cluster_ids"], result["gold_starts"], result["gold_ends"])
                dev_coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)            
            dev_prec, dev_rec, dev_f1 = dev_coref_evaluator.get_prf()
            tf.logging.info("***** Current ckpt path is ***** : {}".format(checkpoint_path))
            tf.logging.info("***** EVAL ON DEV SET *****")
            tf.logging.info("***** [DEV EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(dev_prec, dev_rec, dev_f1))
            if dev_f1 > best_dev_f1:
                best_ckpt_path = checkpoint_path
                best_dev_f1 = dev_f1
                best_dev_prec = dev_prec
                best_dev_rec = dev_rec 
                test_coref_evaluator = metrics.CorefEvaluator()
                for result in estimator.predict(file_based_input_fn_builder(FLAGS.test_file, 
                    num_window=FLAGS.num_window, window_size=FLAGS.window_size, max_num_mention=FLAGS.max_num_mention, 
                    is_training=False, drop_remainder=False),steps=698, checkpoint_path=checkpoint_path, yield_single_examples=False):
                    predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(result["topk_span_starts"], result["topk_span_ends"], result["top_antecedent"], 
                        result["cluster_ids"], result["gold_starts"], result["gold_ends"])
                    test_coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

                test_pre, test_rec, test_f1 = test_coref_evaluator.get_prf()
                test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = test_f1, test_pre, test_rec 
                tf.logging.info("***** EVAL ON TEST SET *****")
                tf.logging.info("***** [TEST EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(test_pre, test_rec, test_f1))

        tf.logging.info("*"*20)
        tf.logging.info("- @@@@@ the path to the BEST DEV result is : {}".format(best_ckpt_path))
        tf.logging.info("- @@@@@ BEST DEV F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(best_dev_f1, best_dev_prec, best_dev_rec))
        tf.logging.info("- @@@@@ TEST when DEV best F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best))


    if FLAGS.do_predict:
        coref_evaluator = metrics.CorefEvaluator()
        model = util.get_model(model_config, model_sign="corefqa")
        for result in estimator.predict(file_based_input_fn_builder(FLAGS.test_file, 
                    num_window=FLAGS.num_window, window_size=FLAGS.window_size, max_num_mention=FLAGS.max_num_mention, 
                    is_training=False, drop_remainder=False),steps=698, checkpoint_path=checkpoint_path, yield_single_examples=False):
            
            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(result["topk_span_starts"], result["topk_span_ends"], 
                result["top_antecedent"], result["cluster_ids"], result["gold_starts"], result["gold_ends"])
            coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        
        p, r, f = coref_evaluator.get_prf()
        tf.logging.info("Average precision: {:.4f}, Average recall: {:.4f}, Average F1 {:.4f}".format(p, r, f))



if __name__ == '__main__':
    # set the random seed. 
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    # start train/evaluate the model.
    tf.app.run()






