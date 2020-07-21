#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 




# author: xiaoy li 


import tensorflow as tf 



if __name__ == "__main__":
    sess = tf.compat.v1.InteractiveSession()
    lhs = tf.zeros((4, 3))

    slice_lhs = tf.gather(lhs, 1)
    # slice_lhs_nd = tf.gather_nd(lhs, 1)

    slice_lhs = tf.gather(lhs, [1, 2])

    slice_lhs.eval()
    # slice_lhs_nd.eval()
    # array([ 0,  0,  3, 10], dtype=int32)
    sess.close()