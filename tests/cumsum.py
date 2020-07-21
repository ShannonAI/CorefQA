#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

"""
input_a:
    array([[0, 1, 0, 1, 1],
       [0, 1, 1, 1, 1],
       [0, 1, 1, 0, 1],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0]], dtype=int32)
cum_input_b:
    array([[1, 2, 3, 4, 5],
       [1, 2, 3, 4, 5],
       [1, 2, 3, 4, 5],
       [1, 2, 3, 4, 5],
       [1, 2, 3, 4, 5]], dtype=int32)
input_c:
    array([[0, 2, 0, 4, 5],
       [0, 2, 3, 4, 5],
       [0, 2, 3, 0, 5],
       [1, 2, 3, 4, 5],
       [0, 2, 3, 4, 0]], dtype=int32)
input_c:
    array([[  1,   2,   3,   4,   5],
       [129, 130, 131, 132, 133],
       [257, 258, 259, 260, 261],
       [385, 386, 387, 388, 389],
       [513, 514, 515, 516, 517]], dtype=int32)
input_d:
    array([[  0,   2,   0,   4,   5],
       [  0, 130, 131, 132, 133],
       [  0, 258, 259,   0, 261],
       [385, 386, 387, 388, 389],
       [  0, 514, 515, 516,   0]], dtype=int32)
flat_input_d:
    array([  0,   2,   0,   4,   5,   0, 130, 131, 132, 133,   0, 258, 259,
        0, 261, 385, 386, 387, 388, 389,   0, 514, 515, 516,   0], dtype=int32)
boolean_mask:
    array([False,  True, False,  True,  True, False,  True,  True,  True,
        True, False,  True,  True, False,  True,  True,  True,  True,
        True,  True, False,  True,  True,  True, False])
input_f:
    array([  2,   4,   5, 130, 131, 132, 133, 258, 259, 261, 385, 386, 387,
       388, 389, 514, 515, 516], dtype=int32)
"""




import tensorflow as tf 


if __name__ == "__main__":
    sess = tf.compat.v1.InteractiveSession()
    input_a = tf.constant([
        [0, 1, 0, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    ones_input_b = tf.ones_like(input_a, tf.int32)
    cum_input_b = tf.math.cumsum(ones_input_b, axis=1)
    cum_input_b.eval()
    # input_c = tf.math.multiply(cum_input_b, input_a)
    # input_c.eval()
    seq_len = 128
    offset = tf.tile(tf.expand_dims(tf.range(5) * 128, 1), [1, 5])
    offset.eval()
    input_e = offset + cum_input_b 
    input_e.eval()

    input_d = tf.math.multiply(input_e, input_a)
    input_d.eval()
    flat_input_d = tf.reshape(input_d, [-1])
    flat_input_d.eval()

    boolean_mask = tf.math.greater(flat_input_d, tf.zeros_like(flat_input_d, tf.int32))
    boolean_mask.eval()

    input_f = tf.boolean_mask(flat_input_d, boolean_mask)
    input_f.eval()

    sess.close()


