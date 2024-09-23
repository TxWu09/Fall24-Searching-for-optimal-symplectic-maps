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
tf.disable_v2_behavior()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
	inp_args = {}
	for arg in sys.argv[1:]:
		k,v = arg.split("=")
		inp_args[k] = v

	tic = time.time()

	restore_session = False #Whether or not to recall the weights from the last training stage or re-initialize.
	local_training_steps = 2000 #The number of training steps that will be performed when this script is run.
	current_lr = .01#0.005 #The learning rate to be used. 
	decay_steps = 500 #Number of training steps before we decay the learning rate.
	decay_rate = 0.5 #The decay factor of the learning rate.
	num_steps_basic = 1 #The number of numerical integration steps for each time discretization of the (time-dependent) Hamiltonian.
	num_macro_steps = 10 #The number of time discretization steps of the Hamiltonian (e.g. the Hamiltonian is constant each the intervals [0,1/5],[1/5,2/5],[2/5,3/5],[3/5,4/5],[4/5,1].)
	b_basic = 2000 #The number of sample domains in the domain (e.g. on the boundary of E(1,5)) used for most of the training.

	num_steps_check_accuracy = num_steps_basic#10 #The number of numerical integration steps used for each time discretization. This number should be larger than num_steps_basic and is used to check the periodically assess the accuracy of the numerical integration.
	b_check_accuracy = 5000 #Similarly, this number should be larger than b_basic and is used to periodically assess the accuracy of the sampling process.

	b_movie = b_basic#The value of b used at the very end when creating a higher resolution movie.
	num_steps_movie = 100 #The number of integration steps used (per time discretization step) for hte movie.

	save_steps = 500
	check_accuracy_steps = 500
	write_steps = 5 #how often to write summary to tensorboard

	b_training_plot = b_basic
	update_training_plot_live_steps = 5 #How often we update the training plot if displaying in real time.
	show_training_plot = True

	update_training_plot_save_steps = 5 #How often we update the training plot for save purposes.

	markersize = 1

	n = 2 #Half the total real dimension.
	tot_secs = 1

	#Parameters for polydisk or ellipsoid (i.e. E(a,b) or P(a,b)). This is the domain, and the target is the four-ball of minimal area.
	aa = 1.0 
	bb = 5.0 

	domain = 'ellipsoid' #Currently can switch between ellipsoid or polydisk.

	assert (update_training_plot_save_steps * local_training_steps * b_training_plot * 2*n * 4) / (2.0**30) <= 16
	#This is a somewhat crude way to avoid a memory error when saving the training data, assuming 16gb of RAM. 

	graph = tf.Graph()
	with graph.as_default():
		z_ph = tf.placeholder(tf.float32,shape=[None,2*n]) #Here z_ph is a placeholder for the initial positions of the sample points.
		lr_ph = tf.placeholder(tf.float32)
		num_steps_ph = tf.placeholder(tf.int32)
		b = tf.shape(z_ph)[0]
		dt = tf.reshape(tot_secs / (num_macro_steps*tf.to_float(num_steps_ph)),[])


		V_weights_list = []
		for i in range(num_macro_steps):
			V_wt = tf.Variable(np.zeros([n + (n)**2 + (n)**3 + (n)**4 + (n)**5],dtype=np.float32))
			V_weights_list.append(V_wt)

		K_weights_list = []
		for i in range(num_macro_steps):
			K_wt = tf.Variable(np.zeros([n + (n)**2 + (n)**3 + (n)**4 + (n)**5],dtype=np.float32))
			K_weights_list.append(K_wt)



		#The Hamiltonian is assumed to be of the form H(q,p) = V(q) + K(p). Currently it's set to be a quintic polynomial.
		def compute_V(q_input,V_wt_input):
			quad = tf.reshape(tf.reshape(q_input,[b,n,1])*tf.reshape(q_input,[b,1,n]),[b,(n)**2]) #shape [b,n*n]
			cub = tf.reshape(tf.reshape(quad,[b,n*n,1])*tf.reshape(q_input,[b,1,n]),[b,(n)**3]) #shape [b,n*n,n]
			quart = tf.reshape(tf.reshape(cub,[b,n*n*n,1])*tf.reshape(q_input,[b,1,n]),[b,(n)**4]) #shape [b,n*n,n]	
			quint = tf.reshape(tf.reshape(quart,[b,n*n*n*n,1])*tf.reshape(q_input,[b,1,n]),[b,(n)**5]) #shape [b,n*n,n]	
			tot = tf.concat([q_input,quad,cub,quart,quint],axis=1)
			return tf.reshape(tf.matmul(tot,tf.reshape(V_wt_input,[-1,1])),[b])			

		def compute_K(p_input,K_wt_input):
			quad = tf.reshape(tf.reshape(p_input,[b,n,1])*tf.reshape(p_input,[b,1,n]),[b,(n)**2]) #shape [b,n*n]
			cub = tf.reshape(tf.reshape(quad,[b,n*n,1])*tf.reshape(p_input,[b,1,n]),[b,(n)**3]) #shape [b,n*n,n]
			quart = tf.reshape(tf.reshape(cub,[b,n*n*n,1])*tf.reshape(p_input,[b,1,n]),[b,(n)**4]) #shape [b,n*n,n]	
			quint = tf.reshape(tf.reshape(quart,[b,n*n*n*n,1])*tf.reshape(p_input,[b,1,n]),[b,(n)**5]) #shape [b,n*n,n]	
			tot = tf.concat([p_input,quad,cub,quart,quint],axis=1)
			return tf.reshape(tf.matmul(tot,tf.reshape(K_wt_input,[-1,1])),[b])		
	

		#Leapfrog:
		q = z_ph[:,0:n]
		p = z_ph[:,n:2*n]
		z_traj = tf.reshape(z_ph,[1,b,2*n]) 
		shape_invariants = [tf.constant(0).get_shape(),q.get_shape(),p.get_shape(),tf.TensorShape([None,None,2*n])]

		for m in range(num_macro_steps):
			#first we increment q a half step forward, leaving p alone:.
			K = compute_K(p,K_weights_list[m])
			dK_p = tf.gradients(K,p)[0]

			q = q + dK_p*0.5*dt
			p = p

			z = tf.concat([q,p],axis=1)
			z_traj = tf.concat([z_traj,tf.reshape(z,[1,b,2*n])],axis=0)

			#Next we increment p a full step and then q a full step, and we repeat this num_steps_ph-1 times:
			while_condition = lambda ctr,q,p,z_traj: tf.less(ctr, num_steps_ph-1) 
			def body(ctr,q,p,z_traj):
				# z = tf.concat([q,p],axis=1) #shape [b,2*n]	
				V = compute_V(q,V_weights_list[m])
				dV_q = tf.gradients(V,q)[0]

				#Increment p by a full step:
				q = q
				p = p - dV_q*dt

				K = compute_K(p,K_weights_list[m])
				dK_p = tf.gradients(K,p)[0]			

				#Increment q by a full step:
				q = q + dK_p*dt
				p = p

				z = tf.concat([q,p],axis=1)
				z_traj = tf.concat([z_traj,tf.reshape(z,[1,b,2*n])],axis=0)

				return ctr+1,q,p,z_traj	

			_,q,p,z_traj = tf.while_loop(while_condition,body,[0,q,p,z_traj],shape_invariants)


			#Finally, we increment p one more full step, and then q one more half step.
			#Increment p by a full step:
			V = compute_V(q,V_weights_list[m])
			dV_q = tf.gradients(V,q)[0]
			q = q
			p = p - dV_q*dt

			#Increment q by a half step:
			K = compute_K(p,K_weights_list[m])
			dK_p = tf.gradients(K,p)[0]
			q = q + dK_p*0.5*dt
			p = p

			z = tf.concat([q,p],axis=1)
			z_traj = tf.concat([z_traj,tf.reshape(z,[1,b,2*n])],axis=0)
			#Note: currently z_traj is pretty screwed up since it sometimes updates by a half step and sometimes by a ful step.


		last_z = tf.concat([q,p],axis=1)

		enclosing_area = np.pi*tf.reduce_max(tf.reduce_sum(last_z*last_z,axis=-1))

		more_accurate_enclosing_area = np.pi*tf.reduce_max(tf.reduce_sum(last_z*last_z,axis=-1))

		loss = enclosing_area

		summ1 = tf.summary.scalar('enclosing_area',enclosing_area)
		summ3 = tf.summary.scalar('loss',loss)
		summ4 = tf.summary.scalar('lr',lr_ph)

		one_timers = []
		one_timers.append(tf.summary.scalar('num_steps_basic',num_steps_basic))
		one_timers.append(tf.summary.scalar('num_macro_steps',num_macro_steps))
		one_timers.append(tf.summary.scalar('b_basic',b_basic))
		one_timers.append(tf.summary.scalar('num_steps_check_accuracy',num_steps_check_accuracy))
		one_timers.append(tf.summary.scalar('b_check_accuracy',b_check_accuracy))

		annotation_string = domain +' aa = '+str(aa)+' bb = '+str(bb) + '\nsplit quartic Hamiltonian'

		one_timers.append(tf.summary.text('annotations',tf.make_tensor_proto(annotation_string,dtype=tf.string)))		
		summ_one_timers = tf.summary.merge(one_timers)

		more_accurate_summ = tf.summary.scalar('more_accurate_enclosing_area',more_accurate_enclosing_area)

		global_step = tf.Variable(0)

		optimizer = tf.train.AdamOptimizer(lr_ph).minimize(loss,global_step=global_step)

		merged = tf.summary.merge([summ1,summ3,summ4])
		saver = tf.train.Saver()


	if domain == 'ellipsoid':
		def F(z):
			z1sq = z[:,0]**2 + z[:,2]**2
			z2sq = z[:,1]**2 + z[:,3]**2
			return np.pi*(z1sq/aa + z2sq/bb)

		# Randomly generate sample points in the (boundary of the) domain.
		def get_z_init(b=b_basic):
			z_init = np.random.uniform(-5,5,[b,4]).astype(np.float32)
			div_fact = np.sqrt(F(z_init).reshape([b,1]))
			z_init = z_init/div_fact
			return z_init

	elif domain == 'polydisk':
		def get_z_init(b=b_basic):
			length = np.sqrt(np.random.uniform(0, 1,[b,2]))
			angle = np.pi * np.random.uniform(0, 2,[b,2])

			for i in range(b//2):
				length[i,0] = 1
			for j in range(b//2,b):
				length[j,1] = 1

			q = length * np.cos(angle)
			p = length * np.sin(angle)

			z1 = np.sqrt(aa)*np.concatenate([q[:,0:1],p[:,0:1]],axis=1)/np.sqrt(np.pi)
			z2 = np.sqrt(bb)*np.concatenate([q[:,1:],p[:,1:]],axis=1)/np.sqrt(np.pi)

			out = np.concatenate([z1,z2],axis=-1)[:,[0,2,1,3]]

			return out

	else:
		raise Exception('Error! Domain not recognized.')

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config,graph=graph) as sess:
		
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
				run_num = np.max([np.int(f[3:]) for f in folders]) + 1	
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
			ax = fig.add_subplot(111,autoscale_on=False,xlim=(-1, 10), ylim=(-1, 10))
			fig.show()

			training_plot_points = []
			for i in range(b_training_plot):  
				# new_point, = ax.plot([],[],'o',markersize=1,color=plt.get_cmap('viridis')(i))
				new_point, = ax.plot([],[],'o',markersize=markersize)				
				training_plot_points.append(new_point)
			training_plot_line, = ax.plot([],[], color='black',alpha=0.5,linewidth=2, markersize=12)
			training_plot_title = ax.text(5,5,'')

		for step in range(local_training_steps):
			_,current_lst_z,lss,enc_ar,gstep,summary = sess.run([optimizer,last_z,loss,enclosing_area,global_step,merged],{z_ph:get_z_init(),lr_ph:current_lr,num_steps_ph:num_steps_basic})	
		
			if step % update_training_plot_save_steps == update_training_plot_save_steps - 1:
				training_plot_points_list.append(current_lst_z)

			if show_training_plot == True and step % update_training_plot_live_steps == update_training_plot_live_steps - 1:
				#update training plot:
				sq = current_lst_z**2
				p_x = np.pi*(sq[:,0] + sq[:,2])
				p_y = np.pi*(sq[:,1] + sq[:,3])
				for i in range(b_training_plot):
					training_plot_points[i].set_data([p_x[i],p_y[i]])	

				training_plot_line.set_data([0,enc_ar],[enc_ar,0])
				training_plot_title.set_text('training step:' +str(gstep))

				plt.pause(0.05)
				fig.show()


			print('using num steps: %s, local step: %s, global step: %s, current_lr: %s, loss: %s, enclosing area: %s' %(num_steps_basic,step,gstep,current_lr,lss,enc_ar))

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
				print('enclosing area for b = %s:' %b_check_accuracy)
				lss,summary = sess.run([more_accurate_enclosing_area,more_accurate_summ],{z_ph:get_z_init(b_check_accuracy),num_steps_ph:num_steps_check_accuracy})
				tensorboard_writer.add_summary(summary,gstep)	
				print(lss)

		saver.save(sess,'./save_variables')
		print('Saved variables.')

		trj = sess.run(z_traj,{z_ph:get_z_init(b_movie),num_steps_ph:num_steps_movie})

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