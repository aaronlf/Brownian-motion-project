#!/usr/bin/env python3

__author__ = "Aaron Lalor-Fitzpatrick"

from scipy import special
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
from randomwalk2d import RandomWalk2D

class RandomWalk3D(RandomWalk2D):
	"""This class implements from RandomWalk2D and can produce Monte Carlo simulations of a 3D random
	walk. 
	"""
	
	###################### CONSTRUCTOR ######################
	
	def __init__(self,n,delta_x=1,x_initial=0,y_initial=0,z_initial=0,stats=False):
		self.z_initial = z_initial
		super().__init__(n,delta_x,x_initial,y_initial,stats)
	
	
	###################### USER-CALLABLE METHODS ######################
	
	def run_monte_carlo(self,number_of_trials=2000,plot=True,show=True): #NEEDS TO BE COMPLETELY CHANGED
		"""Runs a Monte Carlo simulation of the random walk for a specified number of trials.
		It then plots the results as a frequency distribution.
		
		Mean and variance values of the Monte Carlo simulation can be retrieved by calling mc.mean
		and mc.variance, respectively.
		
		#####  Parameters  #####
		- number_of_trials: Integer number of Monte Carlo trials to be conducted. 2000 by default.
		- plot: If true, plot the monte carlo trial results
		- show: If true, plt.show() will be called for each figure, one at a time.
		- overlay: If true, the theoretical pdf will be plotted on top of the Monte Carlo trial
		"""
		
		trial_data = []
		for _ in range(number_of_trials):
			x_steps,y_steps,z_steps = self._random_walk_simulation()
			trial_data.append( sum(x_steps) + self.x_initial )
		x_n, counts = np.unique(trial_data, return_counts=True)
		
		self.mc_mean = np.mean(trial_data)
		self.mc_variance = np.var(trial_data)
		
		mean_total = 0
		for i in range(len(x_n)):
			x,count = x_n[i],counts[i]
			weighted_distance = abs(x - self.x_initial) * count
			mean_total += weighted_distance
		self.mc_mean_distance = mean_total/number_of_trials
			
		
		if plot == False:
			return trial_data
			
		plt.figure("Monte Carlo Simulation of Random Walk")
		plt.scatter(x_n,counts,s=4)
		plt.xlim((-self.n-1,self.n+1))
		plt.xlabel("x\u2099 - Position after n jumps")
		plt.ylabel("Frequency")
		plt.suptitle("Monte Carlo Simulation of Random Walk: p={}, n={}, \u0394x={}, N={}".format(
													self.p,self.n,self.delta_x,number_of_trials))	
		
		if show == True:
			plt.show()		
								
		return trial_data
		
		
	def random_walk_draw(self,num_plots,animated=False,show=True):
		"""This method draws an animated 3D random walk simulation.
		"""
		x_y_z_arrays = []
		for _ in range(num_plots):
			current_x = self.x_initial
			current_y = self.y_initial
			current_z = self.z_initial
			x_array = [current_x]
			y_array = [current_y]
			z_array = [current_z]
			x_steps,y_steps,z_steps = self._random_walk_simulation()
			for i in range(len(x_steps)-1):
				current_x += x_steps[i]
				current_y += y_steps[i]
				current_z += z_steps[i]
				x_array.append(current_x)
				y_array.append(current_y)
				z_array.append(current_z)
			x_y_z_arrays.append([x_array,y_array,z_array])
				
		fig = plt.figure('Random walk live simulation, {} trials, n = {}'.format(num_plots,self.n))
		ax=Axes3D(fig)
		ax.set_ylim([-np.sqrt(self.n)*1.75,np.sqrt(self.n)*1.75])
		ax.set_xlim([-np.sqrt(self.n)*1.75,np.sqrt(self.n)*1.75])
		ax.set_zlim([-np.sqrt(self.n)*1.75,np.sqrt(self.n)*1.75])
		
		if animated == True:
			fig.suptitle('Simulation of 3D random walk, live')
			self.index = 0
			def update(i):
				ax.clear()
				ax.set_ylim([-np.sqrt(self.n)*1.75,np.sqrt(self.n)*1.75])
				ax.set_xlim([-np.sqrt(self.n)*1.75,np.sqrt(self.n)*1.75])
				ax.set_zlim([-np.sqrt(self.n)*1.75,np.sqrt(self.n)*1.75])

				for i in x_y_z_arrays:
					x_arr,y_arr,z_arr = i
					ax.plot(x_arr[:self.index], y_arr[:self.index],z_arr[:self.index],linewidth=0.5)
				self.index += 1
			a = anim.FuncAnimation(fig, update, frames=self.n, repeat=False,interval=10)
		else:
			fig.suptitle('Simulation of 3D random walk, static')
			for i in x_y_z_arrays:
				x_vals,y_vals,z_vals = i
				ax.plot(x_vals, y_vals,z_vals,linewidth=0.5)
	
		
		plt.xlabel("x")
		plt.ylabel("y")
		
		if show == True:
			plt.show()
		return fig
		
		
	###################### PRIVATE METHODS ######################
	
	def _random_walk_simulation(self):
		z_coordinates = [random.random()*random.choice([-1,1]) for _ in range(self.n)]
		x_coordinates = []
		y_coordinates = []
		angles = [random.random()*2*np.pi for _ in range(self.n)]
		
		for index,angle in enumerate(angles):
			multiplier = np.sqrt(1-(z_coordinates[index])**2) 	# By Pythagoras
			x_coordinates.append(np.cos(angle)*multiplier)
			y_coordinates.append(np.sin(angle)*multiplier)
			
		return [x_coordinates,y_coordinates,z_coordinates]
