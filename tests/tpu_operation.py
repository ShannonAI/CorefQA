#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# descripiton:
# test math operations in tpu 


import tensorflow as tf 
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver



TPU_NAME = "tensorflow-tpu"
TPU_ZONE = "us-central1-f"
GCP_PROJECT = "xiaoyli-20-04-274510"



if __name__ == "__main__":
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME, zone=TPU_ZONE, project=GCP_PROJECT)   
    # tpu_cluster_resolver = TPUClusterResolver(tpu=['tensorflow-tpu']).get_master()
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

    scores = tf.constant([1.0, 2.3, 3.2, 4.3, 1.5, 1.8, 98, 2.9])
    k = 2

    def test_top_k():
        top_scores, top_index = tf.nn.top_k(scores, k)
        return top_scores, top_index 

    test_op = test_top_k

    # with tf.compat.v1.InteractiveSession(tpu_cluster_resolver) as sess:
    with tf.compat.v1.Session(tpu_cluster_resolver) as sess:
        sess.run(tpu.initialize_system())

        scores = tf.constant([1.0, 2.3, 3.2, 4.3, 1.5, 1.8, 98, 2.9])
        k = 2
        print("ALL Devices: ", tf.config.experimental_list_devices())

        top_scores, top_index = tf.nn.top_k(scores, k) 

        print(top_scores.eval())
        print(top_index.eval())

        sess.run(tpu.shutdown_system())





