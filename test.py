import matplotlib
import numpy as np
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from subprocess import call
import time
import os
import sys

import tensorflow.compat.v1 as tf

# x = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [4, 1]), dtype=np.float32))
# y = tf.Variable(np.random.normal(scale = 0.1), dtype=np.float32)
# z = tf.multiply(x,y)
# print(x)
# print(tf.transpose(x))
# print(tf.matmul(x, tf.transpose(x)))
# print(y)
# print (z)
# print(tf.constant(np.identity(4), dtype=np.float32))

# print(tf.constant([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))


n = 2
# graph = tf.Graph()
# with graph.as_default():

# 	z_ph = tf.placeholder(tf.float32,shape=[None,2*n]) # Here z_ph is a placeholder for the initial positions of the sample points
# 	b = tf.shape(z_ph)[0]
# 	G = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [4, 4]), dtype=np.float32))
# 	print(z_ph)
# 	print(tf.matmul(z_ph, G))

# z = tf.constant([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
# print(z[0,:])


graph = tf.Graph()
with graph.as_default():

    z_ph = tf.placeholder(tf.float32,shape=[1,2*n]) # Here z_ph is a placeholder for the initial positions of the sample points
    b = tf.shape(z_ph)[0]
    print(z_ph)
    print(b)




    depth = 10
    num_macro_steps = 10
    width = 50
    num_G = 3
    num_Sk = 1
    num_Sv = 1
    num_T = 1
    num_Lfg = 1
    n = 2
    activation = tf.math.tanh
    G_list = []
    neural_list = []
    Sk_list = []
    Sv_list = []
    T_list = []
    Lfg_neural_list = []

    for i in range(num_macro_steps):
                    

        u= tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_G, 4, 1]), dtype=np.float32))
        b= tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_G, 1]), dtype=np.float32))


        c= tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_T, 1, 4]), dtype=np.float32))
        
        inputM = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Sk + num_Sv, 2, width]), dtype=np.float32))
        inputBias = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Sk + num_Sv, 1, width]), dtype=np.float32))
        for j in range(depth - 1):
            w = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Sk + num_Sv, width, width]), dtype=np.float32))
            Bias = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Sk + num_Sv, 1, width]), dtype=np.float32))
            neural_list.append([w, Bias])
            # Sk_list.append(w[])
            # Sv_list = []
            # Lfg_list = []
        outputM = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Sk + num_Sv, width, 2]), dtype=np.float32))

        inputML = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [2*num_Lfg, 1, width]), dtype=np.float32))
        outputML = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [2*num_Lfg, width, 1]), dtype=np.float32))
        for j in range(depth - 1): 
            w = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [2*num_Lfg, width, width]), dtype=np.float32))
            Bias = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [2*num_Lfg, 1, width]), dtype=np.float32))
            Lfg_neural_list.append([w, Bias])


    def Compute_S(input_place, neural_list, inputM, outputM, i):
        input_place = activation(tf.add(inputBias[i, :, :], tf.matmul(input_place, inputM[i, :, :])))
        for j in range(depth - 1):
            w, Bias = neural_list[j]
            input_place = activation(tf.add(Bias[i, :, :], tf.matmul(input_place, w[i, :, :])))
        output = tf.matmul(input_place, outputM[i, :, :])
        return output
        
    z = tf.constant(np.array(np.random.normal(scale = 0.1, size = [1, 2]), dtype=np.float32))
    print(z)

        
    S = Compute_S(z, neural_list, inputM, outputM, 0)
    print(S)