#!/usr/bin/env python3

__author__ = "Aaron Lalor-Fitzpatrick"

from scipy import special
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from randomwalk import RandomWalk

class RandomWalk2D(RandomWalk):
	"""This class implements from RandomWalk and can produce Monte Carlo simulations of a 2D random
	walk. 
	
	This class does not support calculation of theoretical values, as the random walk is in 2D 
	space, as opposed to along a 1D line, so there are many directions.
	"""
	
	###################### CONSTRUCTOR ######################
	
	def __init__(self,p,n,delta_x=1,x_initial=0,stats=True,y_initial=0):
		self.y_initial = y_initial
		super().__init__(p,n,delta_x,x_initial,stats)
	
	
	###################### USER-CALLABLE METHODS ######################
	
	def run_monte_carlo(self,number_of_trials=2000,plot=True,show=True):
		"""Runs a Monte Carlo simulation of the random walk for a specified number of trials.
		It then plots the results as a frequency distribution.
		
		Mean and variance values of the Monte Carlo simulation can be retrieved by calling mc.mean
		and mc.variance, respectively.
		
		#####  Parameters  #####
		- number_of_trials: Integer number of Monte Carlo trials to be conducted. 2000 by default.
		- plot: If true, plot the monte carlo trial results
		- histogram: If true, plot a histogram of the Monte Carlo results
		- show: If true, plt.show() will be called for each figure, one at a time.
		- overlay: If true, the theoretical pdf will be plotted on top of the Monte Carlo trial
		"""
		
		trial_data = []
		for _ in range(number_of_trials):
			x_steps,y_steps = self._random_walk_simulation()
			trial_data.append( sum(steps) + self.x_initial )
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
		"""This method draws an animated 2D random walk simulation.
		"""
		x_y_arrays = []
		for _ in range(num_plots):
			current_x = self.x_initial
			current_y = self.y_initial
			x_array = [current_x]
			y_array = [current_y]
			x_steps,y_steps = self._random_walk_simulation()
			for i in range(len(x_steps)):
				current_x += x_steps[i]
				current_y += y_steps[i]
				x_array.append(current_x)
				y_array.append(current_y)
			x_y_arrays.append([x_array,y_array])
				
		fig = plt.figure('Random walk live simulation')
		ax = fig.add_subplot(1,1,1)
		ax.set_ylim([-np.sqrt(self.n)*2,np.sqrt(self.n)*2])
		ax.set_xlim([-np.sqrt(self.n)*2,np.sqrt(self.n)*2])
		
		if animated == True:
			fig.suptitle('Simulation of 2D random walk, live')
			self.index = 0
			def update(i):
				ax.clear()
				ax.set_ylim([-np.sqrt(self.n)*2,np.sqrt(self.n)*2])
				ax.set_xlim([-np.sqrt(self.n)*2,np.sqrt(self.n)*2])
				for i in x_y_arrays:
					x_arr,y_arr = i
					ax.plot(x_arr[:self.index], y_arr[:self.index])
				self.index += 1
			a = anim.FuncAnimation(fig, update, frames=self.n, repeat=False,interval=10)
		else:
			fig.suptitle('Simulation of 2D random walk, static')
			for i in x_y_arrays:
				x_vals,y_vals = i
				ax.plot(x_vals, y_vals)
		
		if show == True:
			plt.show()
		
		
	###################### PRIVATE METHODS ######################
	
	def _random_walk_simulation(self):
		r = random.random() * 2 * np.pi
		angles = [random.random()*2*np.pi for _ in range(self.n)]
		x_coordinates = [np.cos(angle) for angle in angles]
		y_coordinates = [np.sin(angle) for angle in angles]
		return [x_coordinates,y_coordinates]

