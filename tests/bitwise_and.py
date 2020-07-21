#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


import tensorflow as tf 


if __name__ == "__main__":
    sess = tf.compat.v1.InteractiveSession()
    lhs = tf.constant([0, 5, 3, 14], dtype=tf.int32)
    rhs = tf.constant([5, 0, 7, 11], dtype=tf.int32)

    res = tf.bitwise.bitwise_and(lhs, rhs)
    res.eval()
    # array([ 0, 0, 3, 10], dtype=int32)
    sess.close()


