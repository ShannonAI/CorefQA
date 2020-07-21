#!/usr/bin/env python3 
 
import numpy as np 
import tensorflow as tf 


a_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7], [8, 9, 10]])

print(a_np.shape)
# exit()

def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]


if __name__ == "__main__":

    original_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7], [8, 9, 10]])
    sess = tf.compat.v1.InteractiveSession()
    start_scores = tf.convert_to_tensor(tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7], [8, 9, 10]]))
    print(tf.shape(start_scores))
    # exit()
    expand_scores = tf.tile(tf.expand_dims(start_scores, 2), [1, 1, 3])
    # (6, 3, 3)
    print(expand_scores.eval())
    print(shape(expand_scores, 0))
    print(shape(expand_scores, 1))
    print(shape(expand_scores, 2))
    print("=*="*20)
    # tf.convert_to_tensor(data_np, np.float32)
    # ndarray_scores = tf.make_ndarray(expand_scores)
    # ndarray_scores = tf.convert_to_tensor(expand_scores, np.int32)
    # print(ndarray_scores)
    # exit()
    ndarray_scores = np.array([[[1, 1, 1], [ 2 , 2 , 2], [ 3 , 3 , 3]],
        [[ 4 , 4 , 4],[ 5 , 5 , 5],[ 6 , 6 , 6]],
        [[ 7 , 7 , 7],[ 8 , 8 , 8],[ 9 , 9 , 9]],
        [[ 2 , 2 , 2], [ 3 , 3 , 3], [ 4 , 4 , 4]],
        [[ 5 , 5 , 5], [ 6 , 6 , 6], [ 7 , 7 , 7]],
        [[ 8 , 8 , 8], [ 9 , 9 , 9], [10 , 10 ,10]]])
    print("$="*20)
    print("test_a is : {}".format(str(ndarray_scores[2, 2, 2])))
    print("test_b is : {}".format(str(original_array[2, 2])))
    print("^-"*20)
    print("test_a is : {}".format(str(ndarray_scores[2, 1, 1])))
    print("test_b is : {}".format(str(original_array[2, 1])))
    print("^-"*20)
    print("test_a is : {}".format(str(ndarray_scores[2, 0, 0])))
    print("test_b is : {}".format(str(original_array[2, 0])))
    sess.close() 
    # span_scores[k][i][j] = start_scores[k][i] + end_scores[k][j]
    # start_scores[k][i][j] = start_scores[k][i]
    # end_scores[k][i][j] = end_scores[k][j]

    # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7], [8, 9, 10]]

    """
    [[[1, 1, 1], [ 2 , 2 , 2], [ 3 , 3 , 3]],
    [[ 4 , 4 , 4],[ 5 , 5 , 5],[ 6 , 6 , 6]],
    [[ 7 , 7 , 7],[ 8 , 8 , 8],[ 9 , 9 , 9]],
    [[ 2 , 2 , 2], [ 3 , 3 , 3], [ 4 , 4 , 4]],
    [[ 5 , 5 , 5], [ 6 , 6 , 6], [ 7 , 7 , 7]],
    [[ 8 , 8 , 8], [ 9 , 9 , 9], [10 , 10 ,10]]]
    """

