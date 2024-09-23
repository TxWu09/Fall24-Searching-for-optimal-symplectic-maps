from __future__ import division

import matplotlib
# matplotlib.use('Agg')
import numpy as np 
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from subprocess import call
import sys
import mpl_toolkits.mplot3d.axes3d as p3
import timeit

"""
Usage: python trj_movie_2d_leapfrog_experiments trj17.npy cool_movie.mp4 
"""

movie_time = 20
ms = 1
fps = 25 #Frames per second. Might want to decrease this to create the movie faster.
default_movie_name = 'output_2d.mp4'

def main():

	tic = timeit.default_timer()

	assert len(sys.argv) >= 2

	if len(sys.argv) >= 3:
		movie_name = sys.argv[2]
	else:
		movie_name = default_movie_name

	trj = np.load(sys.argv[1])

	int_steps = trj.shape[0]
	skip_rate = np.max([int_steps//(fps*movie_time),1])
	print('target move time: %s' %movie_time)
	print('integration steps: %s' %int_steps)
	print('skip rate: %s' %skip_rate)
	print('target fps: %s' %fps)

	trj = trj[list(range(0,int_steps,skip_rate))]#+[trj.shape[0]-1]]

	time_factor = movie_time/(trj.shape[0]/fps)
	print ('actual fps: %s'%(fps/time_factor))
	assert fps/time_factor <= 50

	b = trj.shape[1]

	fig = plt.figure()
	ax = fig.add_subplot(111,autoscale_on=False,xlim=(-1, 9), ylim=(-1, 9))
	# ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 6), ylim=(-1, 6))
	ax.grid()

	cmin = np.min(trj[0,:,0])
	cmax = np.max(trj[0,:,0])
	gap = cmax-cmin

	points = []
	for i in range(b):
		new_point, = ax.plot([],[],'o',markersize=ms,color=plt.get_cmap('viridis')((trj[0,i,0]-cmin)/gap))
		points.append(new_point)

	def proj(data):
		return np.pi*(data[0]**2+data[2]**2),np.pi*(data[1]**2+data[3]**2)

	def animate(i):
		for j in range(b):
			pt = points[j]

			p_x,p_y = proj(trj[i,j])		
			pt.set_data([p_x,p_y])


	ani = animation.FuncAnimation(fig, animate,frames=trj.shape[0],repeat=0)

	#Plotting the theoretical optimal ball:
	a_optimal = 2.5
	ax.plot([0,a_optimal], [a_optimal,0], color='black',alpha=0.5,linewidth=2, markersize=12)

	print('Saving movie...')

	ani.save('./'+movie_name, writer='ffmpeg',fps=fps/time_factor)#,bitrate=1800) #another option would be imagemagick
	print('Saved as %s' %movie_name)
	call(['open',movie_name])

	toc = timeit.default_timer()

	print('Time elapsed: %s'%(toc-tic))

if __name__ == "__main__":
    main()






