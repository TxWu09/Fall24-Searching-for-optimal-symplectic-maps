import matplotlib
import numpy as np
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from subprocess import call
import time
import os
import sys

# Tensorflow compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():

	# Process input arguments
	inp_args = {}
	for arg in sys.argv[1:]:
		k,v = arg.split("=")
		inp_args[k] = v

	# Start time
	tic = time.time()

################################################################################
# Initialize parameters
################################################################################

	# Gradient descent parameters
	restore_session = False # Whether or not to recall the weights from the last training stage or re-initialize
	num_epochs = 1 # Number of epochs per script call
	current_lr = 0.001 # Initial learning rate
	decay_steps = 500 # Number of training steps before we decay the learning rate
	decay_rate = 0.5 # Decay factor of the learning rate
	num_samples_per_epoch_basic = 10 # Number of random sample points to draw from the domain at each epoch
	num_samples_per_epoch_check_accuracy = 5000 # Should be larger than num_samples_per_epoch_basic and is used to periodically assess the accuracy of the sampling process

	# Time integration parameters
	num_micro_steps = 1 # Number of numerical integration steps for each time discretization of the (time-dependent) Hamiltonian
	num_macro_steps = 10 # Number of time discretization steps of the Hamiltonian (e.g., the Hamiltonian is constant on each the intervals [0,1/5],[1/5,2/5],[2/5,3/5],[3/5,4/5],[4/5,1])
	num_steps_check_accuracy = num_micro_steps # Should be larger than num_micro_steps and is used to check the periodically assess the accuracy of the numerical integration

	# Movie parameters
	num_samples_per_epoch_movie = num_samples_per_epoch_basic # The value of num_samples_per_epoch used at the very end when creating a higher resolution movie
	num_micro_steps_movie = 100 # The number of micro (numerical integration) steps used per macro (Hamiltonian time discretization) step for the movie

	# Other training parameters
	save_steps = 500 # After how many epochs to save weights to memory
	check_accuracy_steps = 500 # After how many epochs to use "check_accuracy" values for number of samples and numerical integration steps
	write_steps = 5 # After how many epochs to write summary to tensorboard
	
	# Training plot parameters
	show_training_plot = True
	num_samples_per_epoch_training_plot = num_samples_per_epoch_basic # How many samples to plot
	update_training_plot_live_steps = 5 # After how many epochs to update the training plot if displaying in real time
	update_training_plot_save_steps = 5 #How often we update the training plot for save purposes.
	markersize = 1

	# Domain can be either ellipsoid, polydisk, and tori (target is always the smallest four ball)
	# The geometric parameters are defined within the domain definitions
	#
	# Hamiltonian_type can be either neural net, degree 5 polynomial, or degree 15 polynomial
	# The neural net parameters are defined within the model definition
	#
	n = 2 # half the total real dimension
	domain = 'ellipsoid'
	Hamiltonian_type = 'Week 4'

################################################################################
# Define the Hamiltonian model form
################################################################################

	assert (update_training_plot_save_steps * num_epochs * num_samples_per_epoch_training_plot * 2*n * 4) / (2.0**30) <= 16
	#This is a somewhat crude way to avoid a memory error when saving the training data, assuming 16gb of RAM. 

	graph = tf.Graph()
	with graph.as_default():

		z_ph = tf.placeholder(tf.float32,shape=[None,2*n]) # Here z_ph is a placeholder for the initial positions of the sample points
		b = tf.shape(z_ph)[0]
		
		num_steps_ph = tf.placeholder(tf.int32)
		dt = tf.reshape(1 / (num_macro_steps * tf.to_float(num_steps_ph)), [])


		if Hamiltonian_type == 'Week 4':
			depth = 2
			num_macro_steps = 1
			width = 2
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
			Lfg_neural_list_1 = []
			Lfg_neural_list_2 = []
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

				inputBiasL_1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, 1, width]), dtype=np.float32))
				inputML_1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, 1, width]), dtype=np.float32))
				outputML_1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, width, 1]), dtype=np.float32))
				inputBiasL_2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, 1, width]), dtype=np.float32))
				inputML_2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, 1, width]), dtype=np.float32))
				outputML_2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, width, 1]), dtype=np.float32))
				for j in range(depth - 1): 
					w1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, width, width]), dtype=np.float32))
					Bias1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, 1, width]), dtype=np.float32))
					Lfg_neural_list_1.append([w1, Bias1])
					w2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, width, width]), dtype=np.float32))
					Bias2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [num_Lfg, 1, width]), dtype=np.float32))
					Lfg_neural_list_2.append([w2, Bias2])

				T_list.append(c)

				def Compute_G(u, b):
					I = tf.constant(np.identity(4), dtype=np.float32)
					J = tf.constant([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
					output = tf.add(I ,tf.multiply(b, tf.matmul(u, tf.matmul(tf.transpose(u), J))))
					return output
				
				def Compute_S(input_place, neural_list, inputM, outputM, i, inputBias):
					input_place = activation(tf.add(inputBias[i, :, :], tf.matmul(input_place, inputM[i, :, :])))
					for j in range(depth - 1):
						w, Bias = neural_list[j]
						print(w)
						print(Bias)
						input_place = activation(tf.add(Bias[i, :, :], tf.matmul(input_place, w[i, :, :])))
					output = tf.matmul(input_place, outputM[i, :, :])
					return output
				
				def Compute_L(input_place, Lfg_neural_list, inputML, outputML, i, inputBiasL):
					input_place = activation(tf.add(inputBiasL[i, :, :], tf.matmul(input_place, inputML[i, :, :])))
					for j in range(depth - 1):
						w, Bias = Lfg_neural_list[j]
						input_place = activation(tf.add(Bias[i, :, :], tf.matmul(input_place, w[i, :, :])))
					output = tf.matmul(input_place, outputML[i, :, :])
					return output
				
					



		elif Hamiltonian_type == 'b':
			num_macro_steps = 10
			width = 50
			n = 2
			activation = tf.math.tanh
			G1b_list = []
			G1u_list = []
			G2b_list = []
			G2u_list = []
			Tc_list = []
			Sk_list = []
			for i in range(num_macro_steps):

				u1= tf.Variable(np.array(np.random.normal(scale = 0.1, size = [4, 1]), dtype=np.float32))
				b1= tf.Variable(np.random.normal(scale = 0.1), dtype=np.float32)

				u2= tf.Variable(np.array(np.random.normal(scale = 0.1, size = [4, 1]), dtype=np.float32))
				b2= tf.Variable(np.random.normal(scale = 0.1), dtype=np.float32)

				c= tf.Variable(np.array(np.random.normal(scale = 0.1, size = [1, 4]), dtype=np.float32))

				w1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [2, width]), dtype=np.float32))
				w2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, width]), dtype=np.float32))
				Bias1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [1, width]), dtype=np.float32))
				Bias2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [1, width]), dtype=np.float32))
				outputM = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, 2]), dtype=np.float32))
				# My neural network function code is different with original author as following
				# 1. I think his neural network function actually have 3 depth
				# 	 He generates 3 slope functions (1 input matrix and 2 layer matrix). I only generate 2 slope functions.
				#	 My w1 is his input matrix and w2 is his layer1 matrix. Similiar for bias
				
				# Reminder: 
				#    input matrix in the code is a 2 x w matrix, but in the paper, input matrix should be a w x 2 matrix
				# 	 Because 
				# x and y are 1 x 2 matrices in python instead of 2 x 1 in paper. 
				# 	 We do multiplication in "x * input matrix" in python, but in paper we do "input matrix * transpose (x)" in the paper. (x is 1 x 2 matrix)


				G1b_list.append(b1)
				G1u_list.append(u1)
				G2b_list.append(b2)
				G2u_list.append(u2)
				Tc_list.append(c)
				Sk_list.append([w1, w2, Bias1, Bias2, outputM])

			def compute_G(u, b):
				I = tf.constant(np.identity(4), dtype=np.float32)
				J = tf.constant([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
				output = tf.add(I ,tf.multiply(b, tf.matmul(u, tf.matmul(tf.transpose(u), J))))
				return output
			
			def compute_S(y_input, S_weight):
				w1, w2, Bias1, Bias2, outputM = S_weight
				depth_1_cal = activation(tf.add(Bias1, tf.matmul(y_input, w1)))
				depth_2_cal = activation(tf.add(Bias2, tf.matmul(depth_1_cal, w2)))
				output = tf.matmul(depth_2_cal, outputM)
				return output


		#The Hamiltonian is assumed to be of the form H(x,y) = F(x) + G(y).

		# Neural net (width, depth, and activation function specified below)
		elif Hamiltonian_type == 'neural net':

			width = 50
			#depth = 2 # broken for now, second-order gradient doesn't support loops
			activation = tf.math.tanh

			F_weights_list = []
			for i in range(num_macro_steps):

				input_matrix = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [2, width]), dtype=np.float32)) # dimensions width by 2*n
				input_bias = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width]), dtype=np.float32)) # dimensions width
				
				'''
				layer_matrices_list = []
				layer_biases_list = []
				for j in range(depth):
					layer_matrix = tf.Variable(np.zeros([width, width], dtype=np.float32)) # dimensions width by width
					layer_matrices_list.append(layer_matrix)
					layer_bias = tf.Variable(np.zeros([width], dtype=np.float32)) # dimensions width by width
					layer_biases_list.append(layer_bias)
				'''
				layer_matrix1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, width]), dtype=np.float32)) # dimensions width by width
				layer_bias1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width]), dtype=np.float32)) # dimensions width by width
				layer_matrix2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, width]), dtype=np.float32)) # dimensions width by width
				layer_bias2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width]), dtype=np.float32)) # dimensions width by width

				output_matrix = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, 2]), dtype=np.float32)) # dimensions 2*n by width

				#F_weight = [input_matrix, layer_matrices_list, output_matrix]
				F_weight = [input_matrix, layer_matrix1, layer_matrix2, layer_bias1, layer_bias2, output_matrix]
				F_weights_list.append(F_weight)
	
			G_weights_list = []
			for i in range(num_macro_steps):

				input_matrix = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [2, width]), dtype=np.float32)) # dimensions width by 2*n
				input_bias = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width]), dtype=np.float32)) # dimensions width
				
				'''
				layer_matrices_list = []
				layer_biases_list = []
				for j in range(depth):
					layer_matrix = tf.Variable(np.zeros([width, width], dtype=np.float32)) # dimensions width by width
					layer_matrices_list.append(layer_matrix)
					layer_bias = tf.Variable(np.zeros([width], dtype=np.float32)) # dimensions width by width
					layer_biases_list.append(layer_bias)
				'''
				layer_matrix1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, width]), dtype=np.float32)) # dimensions width by width
				layer_bias1 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width]), dtype=np.float32)) # dimensions width by width
				layer_matrix2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, width]), dtype=np.float32)) # dimensions width by width
				layer_bias2 = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width]), dtype=np.float32)) # dimensions width by width

				output_matrix = tf.Variable(np.array(np.random.normal(scale = 0.1, size = [width, 2]), dtype=np.float32)) # dimensions 2*n by width

				#G_weight = [input_matrix, layer_matrices_list, output_matrix]
				G_weight = [input_matrix, layer_matrix1, layer_matrix2, layer_bias1, layer_bias2, output_matrix]
				G_weights_list.append(F_weight)
	
			def compute_F(x_input, F_weight_input):

				'''
				input_matrix = F_weight_input[0]
				layer_matrices_list = F_weight_input[1]
				output_matrix = F_weight_input[2]
				'''
				input_matrix, layer_matrix1, layer_matrix2, layer_bias1, layer_bias2, output_matrix = F_weight_input

				layer_matrices_input = activation(tf.add(input_bias, tf.matmul(x_input, input_matrix)))
				x_layer = layer_matrices_input

				x_layer = activation(tf.add(layer_bias1, tf.matmul(x_layer, layer_matrix1)))
				x_layer = activation(tf.add(layer_bias2, tf.matmul(x_layer, layer_matrix2)))

				layer_matrices_output = x_layer
				x_output = tf.matmul(layer_matrices_output, output_matrix)

				return x_output
	
			def compute_G(y_input, G_weight_input):

				'''
				input_matrix = G_weight_input[0]
				layer_matrices_list = G_weight_input[1]
				output_matrix = G_weight_input[2]
				'''
				input_matrix, layer_matrix1, layer_matrix2, layer_bias1, layer_bias2, output_matrix = G_weight_input

				layer_matrices_input = activation(tf.add(input_bias, tf.matmul(y_input, input_matrix)))
				y_layer = layer_matrices_input

				y_layer = activation(tf.add(layer_bias1, tf.matmul(y_layer, layer_matrix1)))
				y_layer = activation(tf.add(layer_bias2, tf.matmul(y_layer, layer_matrix2)))

				layer_matrices_output = y_layer
				y_output = tf.matmul(layer_matrices_output, output_matrix)

				return y_output

		# Degree 5 polynomial
		elif Hamiltonian_type == 'degree 5 polynomial':

			F_weights_list = []
			for i in range(num_macro_steps):
				F_weight = tf.Variable(np.zeros([sum(n**(j+1) for j in range(5))],dtype=np.float32))
				F_weights_list.append(F_weight)
	
			G_weights_list = []
			for i in range(num_macro_steps):
				G_weight = tf.Variable(np.zeros([sum(n**(j+1) for j in range(5))],dtype=np.float32))
				G_weights_list.append(G_weight)
	
			def compute_F(x_input, F_weight_input):
				deg2  = tf.reshape(tf.reshape(x_input, [b,n,    1])*tf.reshape(x_input, [b,1,n]), [b,n**2]) # shape [b,n**2 ]
				deg3  = tf.reshape(tf.reshape(deg2,    [b,n**2, 1])*tf.reshape(x_input, [b,1,n]), [b,n**3]) # shape [b,n**3 ]
				deg4  = tf.reshape(tf.reshape(deg3,    [b,n**3, 1])*tf.reshape(x_input, [b,1,n]), [b,n**4]) # shape [b,n**4 ]	
				deg5  = tf.reshape(tf.reshape(deg4,    [b,n**4, 1])*tf.reshape(x_input, [b,1,n]), [b,n**5]) # shape [b,n**5 ]	
				total = tf.concat([x_input,deg2,deg3,deg4,deg5], axis=1)
				return tf.reshape(tf.matmul(total, tf.reshape(F_weight_input, [-1,1])), [b])			
	
			def compute_G(y_input, G_weight_input):
				deg2  = tf.reshape(tf.reshape(y_input, [b,n,    1])*tf.reshape(y_input, [b,1,n]), [b,n**2]) # shape [b,n**2 ]
				deg3  = tf.reshape(tf.reshape(deg2,    [b,n**2, 1])*tf.reshape(y_input, [b,1,n]), [b,n**3]) # shape [b,n**3 ]
				deg4  = tf.reshape(tf.reshape(deg3,    [b,n**3, 1])*tf.reshape(y_input, [b,1,n]), [b,n**4]) # shape [b,n**4 ]	
				deg5  = tf.reshape(tf.reshape(deg4,    [b,n**4, 1])*tf.reshape(y_input, [b,1,n]), [b,n**5]) # shape [b,n**5 ]	
				total = tf.concat([y_input,deg2,deg3,deg4,deg5], axis=1)
				return tf.reshape(tf.matmul(total, tf.reshape(G_weight_input, [-1,1])), [b])				

		# Degree 15 polynomial
		elif Hamiltonian_type == 'degree 15 polynomial':

			F_weights_list = []
			for i in range(num_macro_steps):
				F_weight = tf.Variable(np.zeros([sum(n**(j+1) for j in range(15))],dtype=np.float32))
				F_weights_list.append(F_weight)
	
			G_weights_list = []
			for i in range(num_macro_steps):
				G_weight = tf.Variable(np.zeros([sum(n**(j+1) for j in range(15))],dtype=np.float32))
				G_weights_list.append(G_weight)
	
			def compute_F(x_input, F_weight_input):
				deg2  = tf.reshape(tf.reshape(x_input, [b,n,    1])*tf.reshape(x_input, [b,1,n]), [b,n**2 ]) # shape [b,n**2 ]
				deg3  = tf.reshape(tf.reshape(deg2,    [b,n**2, 1])*tf.reshape(x_input, [b,1,n]), [b,n**3 ]) # shape [b,n**3 ]
				deg4  = tf.reshape(tf.reshape(deg3,    [b,n**3, 1])*tf.reshape(x_input, [b,1,n]), [b,n**4 ]) # shape [b,n**4 ]	
				deg5  = tf.reshape(tf.reshape(deg4,    [b,n**4, 1])*tf.reshape(x_input, [b,1,n]), [b,n**5 ]) # shape [b,n**5 ]	
				deg6  = tf.reshape(tf.reshape(deg5,    [b,n**5, 1])*tf.reshape(x_input, [b,1,n]), [b,n**6 ]) # shape [b,n**6 ]
				deg7  = tf.reshape(tf.reshape(deg6,    [b,n**6, 1])*tf.reshape(x_input, [b,1,n]), [b,n**7 ]) # shape [b,n**7 ]
				deg8  = tf.reshape(tf.reshape(deg7,    [b,n**7, 1])*tf.reshape(x_input, [b,1,n]), [b,n**8 ]) # shape [b,n**8 ]	
				deg9  = tf.reshape(tf.reshape(deg8,    [b,n**8, 1])*tf.reshape(x_input, [b,1,n]), [b,n**9 ]) # shape [b,n**9 ]	
				deg10 = tf.reshape(tf.reshape(deg9,    [b,n**9, 1])*tf.reshape(x_input, [b,1,n]), [b,n**10]) # shape [b,n**10]
				deg11 = tf.reshape(tf.reshape(deg10,   [b,n**10,1])*tf.reshape(x_input, [b,1,n]), [b,n**11]) # shape [b,n**11]
				deg12 = tf.reshape(tf.reshape(deg11,   [b,n**11,1])*tf.reshape(x_input, [b,1,n]), [b,n**12]) # shape [b,n**12]	
				deg13 = tf.reshape(tf.reshape(deg12,   [b,n**12,1])*tf.reshape(x_input, [b,1,n]), [b,n**13]) # shape [b,n**13]	
				deg12 = tf.reshape(tf.reshape(deg13,   [b,n**13,1])*tf.reshape(x_input, [b,1,n]), [b,n**14]) # shape [b,n**14]	
				deg13 = tf.reshape(tf.reshape(deg14,   [b,n**14,1])*tf.reshape(x_input, [b,1,n]), [b,n**15]) # shape [b,n**15]	
				total = tf.concat([x_input,deg2,deg3,deg4,deg5,deg6,deg7,deg8,deg9,deg10,deg11,deg12,deg13,deg14,deg15], axis=1)
				return tf.reshape(tf.matmul(total, tf.reshape(F_weight_input, [-1,1])), [b])			
	
			def compute_G(y_input, G_weight_input):
				deg2  = tf.reshape(tf.reshape(y_input, [b,n,    1])*tf.reshape(y_input, [b,1,n]), [b,n**2 ]) # shape [b,n**2 ]
				deg3  = tf.reshape(tf.reshape(deg2,    [b,n**2, 1])*tf.reshape(y_input, [b,1,n]), [b,n**3 ]) # shape [b,n**3 ]
				deg4  = tf.reshape(tf.reshape(deg3,    [b,n**3, 1])*tf.reshape(y_input, [b,1,n]), [b,n**4 ]) # shape [b,n**4 ]	
				deg5  = tf.reshape(tf.reshape(deg4,    [b,n**4, 1])*tf.reshape(y_input, [b,1,n]), [b,n**5 ]) # shape [b,n**5 ]	
				deg6  = tf.reshape(tf.reshape(deg5,    [b,n**5, 1])*tf.reshape(y_input, [b,1,n]), [b,n**6 ]) # shape [b,n**6 ]
				deg7  = tf.reshape(tf.reshape(deg6,    [b,n**6, 1])*tf.reshape(y_input, [b,1,n]), [b,n**7 ]) # shape [b,n**7 ]
				deg8  = tf.reshape(tf.reshape(deg7,    [b,n**7, 1])*tf.reshape(y_input, [b,1,n]), [b,n**8 ]) # shape [b,n**8 ]	
				deg9  = tf.reshape(tf.reshape(deg8,    [b,n**8, 1])*tf.reshape(y_input, [b,1,n]), [b,n**9 ]) # shape [b,n**9 ]	
				deg10 = tf.reshape(tf.reshape(deg9,    [b,n**9, 1])*tf.reshape(y_input, [b,1,n]), [b,n**10]) # shape [b,n**10]
				deg11 = tf.reshape(tf.reshape(deg10,   [b,n**10,1])*tf.reshape(y_input, [b,1,n]), [b,n**11]) # shape [b,n**11]
				deg12 = tf.reshape(tf.reshape(deg11,   [b,n**11,1])*tf.reshape(y_input, [b,1,n]), [b,n**12]) # shape [b,n**12]	
				deg13 = tf.reshape(tf.reshape(deg12,   [b,n**12,1])*tf.reshape(y_input, [b,1,n]), [b,n**13]) # shape [b,n**13]	
				deg12 = tf.reshape(tf.reshape(deg13,   [b,n**13,1])*tf.reshape(y_input, [b,1,n]), [b,n**14]) # shape [b,n**14]	
				deg13 = tf.reshape(tf.reshape(deg14,   [b,n**14,1])*tf.reshape(y_input, [b,1,n]), [b,n**15]) # shape [b,n**15]	
				total = tf.concat([y_input,deg2,deg3,deg4,deg5,deg6,deg7,deg8,deg9,deg10,deg11,deg12,deg13,deg14,deg15], axis=1)
				return tf.reshape(tf.matmul(total, tf.reshape(G_weight_input, [-1,1])), [b])				

		else:

			raise Exception('Error! Hamiltonian type not recognized')
		

################################################################################
# b Calculation
################################################################################
		if Hamiltonian_type == 'b':
			z_traj = tf.reshape(z_ph,[1,b,2*n]) 
			# shape_invariants = [tf.constant(0).get_shape(), x.get_shape(), y.get_shape(), tf.TensorShape([None,None,2*n])]
			z = z_ph
			for m in range(num_macro_steps):
				# First do G2
				G1 = compute_G(G1u_list[m], G1b_list[m])
				z = tf.matmul(z, G1)

				x = z[:,0:n]
				y = z[:,n:2*n]

				S = compute_S(y, Sk_list[m])
				dy = tf.gradients(S, y)[0]

				x = x + dy * dt
				y = y
				z = tf.concat([x,y], axis=1)
				z_traj = tf.concat([z_traj, tf.reshape(z, [1,b,2*n])], axis=0)

				G2 = compute_G(G2u_list[m], G2b_list[m])
				z = tf.matmul(z, G2)
				z = z + Tc_list[m]

				x = z[:,0:n]
				y = z[:,n:2*n]
		elif Hamiltonian_type == 'week 4':
			z_traj = tf.reshape(z_ph,[1,b,2*n]) 
			# shape_invariants = [tf.constant(0).get_shape(), x.get_shape(), y.get_shape(), tf.TensorShape([None,None,2*n])]
			z = z_ph
			x = z[:, 0:n]
			for m in range(num_macro_steps):
				print(x)
				S = Compute_S(x, neural_list, inputM, outputM, 0)
			

			



################################################################################
# Leapfrog integration
################################################################################

		# x = z_ph[:,0:n]
		# y = z_ph[:,n:2*n]
		# z_traj = tf.reshape(z_ph,[1,b,2*n]) 
		# shape_invariants = [tf.constant(0).get_shape(), x.get_shape(), y.get_shape(), tf.TensorShape([None,None,2*n])]

		# for m in range(num_macro_steps):

		# 	# First we increment x a half step forward, leaving y alone
		# 	G = compute_G(y, G_weights_list[m])
		# 	dG_y = tf.gradients(G, y)[0]

		# 	x = x + dG_y * 0.5 * dt
		# 	y = y

		# 	z = tf.concat([x,y], axis=1)
		# 	z_traj = tf.concat([z_traj, tf.reshape(z, [1,b,2*n])], axis=0)

		# 	# Next we increment y a full step, and then x a full step, and we repeat this num_steps_ph-1 times
		# 	while_condition = lambda ctr, x, y, z_traj: tf.less(ctr, num_steps_ph-1) 
		# 	def body(ctr, x, y, z_traj):

		# 		# z = tf.concat([x,y], axis=1) # shape [b,2*n]	
		# 		F = compute_F(x, F_weights_list[m])
		# 		dF_x = tf.gradients(F, x)[0]

		# 		# Increment y by a full step:
		# 		x = x
		# 		y = y - dF_x * dt

		# 		G = compute_G(y, G_weights_list[m])
		# 		dG_y = tf.gradients(G, y)[0]			

		# 		# Increment x by a full step:
		# 		x = y + dG_y * dt
		# 		x = x

		# 		z = tf.concat([x,y], axis=1)
		# 		z_traj = tf.concat([z_traj,tf.reshape(z, [1,b,2*n])], axis=0)

		# 		return ctr+1, x, y, z_traj	

		# 	_,x,y,z_traj = tf.while_loop(while_condition,body, [0,x,y,z_traj], shape_invariants)

		# 	# Finally, we increment y one more full step, and then x one more half step
			
		# 	# Increment y by a full step
		# 	F = compute_F(x, F_weights_list[m])
		# 	dF_x = tf.gradients(F, x)[0]
		# 	x = x
		# 	y = y - dF_x * dt

		# 	# Increment x by a half step
		# 	G = compute_G(y, G_weights_list[m])
		# 	dG_y = tf.gradients(G, y)[0]
		# 	x = x + dG_y * 0.5 * dt
		# 	y = y

		# 	z = tf.concat([x,y], axis=1)
		# 	z_traj = tf.concat([z_traj, tf.reshape(z, [1,b,2*n])], axis=0)

		# 	# Note: currently z_traj is pretty screwed up since it sometimes updates by a half step and sometimes by a full step

################################################################################
# Gradient descent setup
################################################################################

		lr_ph = tf.placeholder(tf.float32)

		last_z = tf.concat([x,y],axis=1)

		enclosing_area = np.pi*tf.reduce_max(tf.reduce_sum(last_z*last_z,axis=-1))

		more_accurate_enclosing_area = np.pi*tf.reduce_max(tf.reduce_sum(last_z*last_z,axis=-1))

		loss = enclosing_area

		summ1 = tf.summary.scalar('enclosing_area',enclosing_area)
		summ3 = tf.summary.scalar('loss',loss)
		summ4 = tf.summary.scalar('lr',lr_ph)

		one_timers = []
		one_timers.append(tf.summary.scalar('num_micro_steps', num_micro_steps))
		one_timers.append(tf.summary.scalar('num_macro_steps', num_macro_steps))
		one_timers.append(tf.summary.scalar('num_samples_per_epoch_basic', num_samples_per_epoch_basic))
		one_timers.append(tf.summary.scalar('num_steps_check_accuracy', num_steps_check_accuracy))
		one_timers.append(tf.summary.scalar('num_samples_per_epoch_check_accuracy', num_samples_per_epoch_check_accuracy))

		annotation_string = domain + ' ' + Hamiltonian_type

		one_timers.append(tf.summary.text('annotations', tf.make_tensor_proto(annotation_string, dtype=tf.string)))		
		summ_one_timers = tf.summary.merge(one_timers)

		more_accurate_summ = tf.summary.scalar('more_accurate_enclosing_area', more_accurate_enclosing_area)

		global_step = tf.Variable(0)

		optimizer = tf.train.AdamOptimizer(lr_ph).minimize(loss, global_step=global_step)

		merged = tf.summary.merge([summ1,summ3,summ4])
		saver = tf.train.Saver()

################################################################################
# Define possible source domains (target is always the smallest four ball)
################################################################################

	if domain == 'ellipsoid':

		# Geometric parameters
		# The equation for the ellipsoid is (\pi * |z_1|^2 / aa) + (\pi * |z_2|^2 / bb) \leq 1
		aa = 1.0
		bb = 3.0

		def F(z):
			z1sq = z[:,0]**2 + z[:,2]**2
			z2sq = z[:,1]**2 + z[:,3]**2
			return np.pi*(z1sq/aa + z2sq/bb)

		# Randomly generate sample points in the (boundary of the) domain.
		def get_z_init(num_samples_per_epoch = num_samples_per_epoch_basic):

			z_init = np.random.uniform(-5,5, [num_samples_per_epoch, 4]).astype(np.float32)
			div_fact = np.sqrt(F(z_init).reshape([num_samples_per_epoch, 1]))
			z_init = z_init/div_fact
			print(z_init[:, 0])
			print(z_init[:, 2])
			print(z_init[:, 1])
			print(z_init[:, 3])

			return z_init

	elif domain == 'polydisk':

		# Geometric parameters
		# The equations for the polydisk are (\pi * |z_1|^2 / aa) \leq 1 and (\pi * |z_2|^2 / bb) \leq 1
		aa = 1.0
		bb = 5.0

		def get_z_init(num_samples_per_epoch = num_samples_per_epoch_basic):

			# Sample from the unit polydisk
			r = np.sqrt(np.random.uniform(0, 1, [num_samples_per_epoch, 2]))
			theta = np.pi * np.random.uniform(0, 2, [num_samples_per_epoch, 2])

			for i in range(num_samples_per_epoch // 2):
				r[i,0] = 1
			for j in range(num_samples_per_epoch // 2, num_samples_per_epoch):
				r[j,1] = 1

			# Convert from polar to cartesian coordinates
			x = r * np.cos(theta)
			y = r * np.sin(theta)

			# Transform normalized samples from unit polydisk into the proper domain
			z_a = np.sqrt(aa) * np.concatenate([x[:,:1], y[:,:1]], axis=1) / np.sqrt(np.pi)
			z_b = np.sqrt(bb) * np.concatenate([x[:,1:], y[:,1:]], axis=1) / np.sqrt(np.pi)

			# Each row is a sample e.g., x_a[i] x_b[i] y_a[i] y_b[i]
			samples = np.concatenate([z_a,z_b], axis=-1)[:, [0,2,1,3]]

			return samples

	elif domain == 'tori':

		# Number of disjoint tori
		num_tori = 2

		# Geometric parameters
		# We write each T2 = S1_a \times S1_b
		# It is the user's responsibility to ensure that the tori are disjoint
		S1_a_radii = [0.5,0.5,]
		S1_a_x_centers = [0.6,-0.6,]
		S1_a_y_centers = [0.6,-0.6,]
		S1_b_radii = [0.5,0.5,]
		S1_b_x_centers = [0.6,-0.6,]
		S1_b_y_centers = [0.6,-0.6,]

		# Check that all tori are fully specified
		list_sizes = [len(S1_a_radii), len(S1_a_x_centers), len(S1_a_y_centers),
			len(S1_b_radii), len(S1_b_x_centers), len(S1_b_y_centers)]
		if not all(list_size == list_sizes[0] for list_size in list_sizes):
			raise Exception('Error! Need radii and centers for each torus')

		def get_z_init(num_samples_per_epoch = num_samples_per_epoch_basic):

			theta = np.pi * np.random.uniform(0, 2, [num_samples_per_epoch, 2, num_tori])
			x = np.transpose(np.concatenate(np.transpose(np.add(np.multiply(
				np.cos(theta),
				np.stack([S1_a_radii, S1_b_radii])),
				np.stack([S1_a_x_centers, S1_b_x_centers]))), axis=-1))
			y = np.transpose(np.concatenate(np.transpose(np.add(np.multiply(
				np.sin(theta),
				np.stack([S1_a_radii, S1_b_radii])),
				np.stack([S1_a_y_centers, S1_b_y_centers]))), axis=-1))

			# Each row is a sample e.g., x_a[i] x_b[i] y_a[i] y_b[i] (the samples for each torus are then stacked vertically)
			samples = np.concatenate([x,y], axis=-1)

			return samples

	else:

		raise Exception('Error! Domain not recognized')

################################################################################
# Main gradient descent loop and tensorflow boilerplate
################################################################################

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config, graph=graph) as sess:
		
		if restore_session:

			folders = os.listdir(os.curdir + '/summaries')
			for f in folders:
				assert f[:3] == 'run'
			run_num = np.max([int(f[3:]) for f in folders])
			tensorboard_writer = tf.summary.FileWriter('summaries' + '/run' + str(run_num))		

			print('Restoring session...')
			new_saver = tf.train.import_meta_graph('./save_variables.meta')
			new_saver.restore(sess,tf.train.latest_checkpoint('./'))

		else:

			if not os.path.isdir('summaries'):
				
				run_num = 0

			else:

				folders = os.listdir(os.curdir + '/summaries')

				for f in folders:
					assert f[:3] == 'run'

				run_num = np.max([int(f[3:]) for f in folders]) + 1	
				print ('run_num: %s' %run_num)

			tensorboard_writer = tf.summary.FileWriter('summaries' + '/run' + str(run_num))
			print('summaries' + '/run' + '/' + str(run_num))

			print('Initializing variables...')
			sess.run(tf.global_variables_initializer())

		summary = sess.run(summ_one_timers)
		tensorboard_writer.add_summary(summary)

		training_plot_points_list = []

		if show_training_plot == True:

			fig = plt.figure()
			ax = fig.add_subplot(111,autoscale_on=False,xlim=(-0.5, 2), ylim=(-0.5, 2))
			fig.show()

			training_plot_points = []

			for i in range(num_samples_per_epoch_training_plot):
				# new_point, = ax.plot([],[],'o',markersize=1,color=plt.get_cmap('viridis')(i))
				new_point, = ax.plot([],[],'o',markersize=markersize)				
				training_plot_points.append(new_point)

			training_plot_line, = ax.plot([],[], color='black',alpha=0.5,linewidth=2, markersize=12)
			training_plot_title = ax.text(1.5,1.5,'')

		for step in range(num_epochs):

			_,current_lst_z,lss,enc_ar,gstep,summary = sess.run([optimizer, last_z, loss, enclosing_area, global_step,merged],{z_ph:get_z_init(), lr_ph:current_lr, num_steps_ph:num_micro_steps})	
		
			if step % update_training_plot_save_steps == update_training_plot_save_steps - 1:
				training_plot_points_list.append(current_lst_z)

			if show_training_plot == True and step % update_training_plot_live_steps == update_training_plot_live_steps - 1:

				# Update training plot:
				sq = current_lst_z**2
				p_x = np.pi*(sq[:,0] + sq[:,2])
				p_y = np.pi*(sq[:,1] + sq[:,3])

				for i in range(num_samples_per_epoch_training_plot):
					training_plot_points[i].set_data([p_x,p_y])

				training_plot_line.set_data([0,enc_ar],[enc_ar,0])
				training_plot_title.set_text('training step:' +str(gstep))

				plt.pause(5)
				fig.show()

			print('using num steps: %s, local step: %s, global step: %s, current_lr: %s, loss: %s, enclosing area: %s' %(num_micro_steps, step, gstep, current_lr, lss, enc_ar))

			if step % save_steps == save_steps - 1:

				saver.save(sess,'./save_variables')
				print('Saved variables.')

			if step % write_steps == write_steps - 1:

				tensorboard_writer.add_summary(summary,gstep)			
				print('Wrote variables to tensorboard.')

			if gstep % decay_steps == decay_steps - 1:

				current_lr = decay_rate*current_lr

			if step % check_accuracy_steps == check_accuracy_steps - 1:

				print('using num steps: %s'%num_steps_check_accuracy)
				print('enclosing area for b = %s:' %num_samples_per_epoch_check_accuracy)
				lss,summary = sess.run([more_accurate_enclosing_area,more_accurate_summ],{z_ph:get_z_init(num_samples_per_epoch_check_accuracy),num_steps_ph:num_steps_check_accuracy})
				tensorboard_writer.add_summary(summary,gstep)	
				print(lss)

		saver.save(sess,'./save_variables')
		print('Saved variables.')

		trj = sess.run(z_traj,{z_ph:get_z_init(num_samples_per_epoch_movie),num_steps_ph:num_micro_steps_movie})

	toc = time.time()
	print('Total time elapsed: %s' %(toc-tic))

	trj_name = 'trj_run' + str(run_num) + '.npy'
	np.save(trj_name,trj)
	print('saved ' + trj_name + ' (this is the flow for the time dependent Hamiltonian found at the end of training)')

	trn = np.array(training_plot_points_list)
	trn_name = 'trn_run' + str(run_num) + '.npy'
	np.save(trn_name,trn)
	print('saved ' + trn_name + ' (this is the current best embedding as a function of training time)')	
	print('note: trn.npy only records the training progress every %s steps' %update_training_plot_save_steps)

if __name__ == "__main__":
	main()