#!/usr/bin/env python3

__author__ = "Aaron Lalor-Fitzpatrick"

from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


class RandomWalk:
	"""
	Each RandomWalk instance is an object of the random walk model that can be created requiring 
	only the values of n, p, delta_x  and x_initial (delta_x is 1 by default and x_initial is 0).
	
	- By calling the plot_distribution() method, one can plot the random walk's probability 
	  distribution.
	- By calling the get_confidence_interval(a,b) method, one can get the probability that the 
	  final value, x_n, will end up between the limits a and b.
	- By calling the plot_monte_carlo() method, one can plot a Monte Carlo simulation of the
	  random walk. Two graphs can be plotted: A histogram (optional) with N on the y-axis, and/or
	  a normalised version of this (all N-values divided by n) that resembles a pdf.
	"""
	
	def __init__(self,p,n,delta_x=1,x_initial=0,stats=True):
		
		if type(n) != int:
			if type(n) == float:
				n = int(round(n))
			else:
				print("You must enter a numerical value for n (integer).")
				print("Closing...")
				quit()
				
		if (type(p) not in [int,float]) or (p < 0) or (p > 1):
			print("You must enter a numerical value for p between 0 and 1.")
			print("Closing...")
			quit()
			
		self.n = n
		self.p = p
		self.delta_x = delta_x 
		self.x_initial = x_initial
		
		if stats == True:
			self.mean = self._calculate_mean()
			self.variance = self._calculate_variance()
			self.tuple_of_probabilities = self._get_tuple_of_probabilities()
			self.mean_distance = self._calculate_mean_distance_theoretical()
		
	def get_confidence_interval(self,a,b):
		"""Returns the theoretical probability that x_n will be between the values a and b.
		"""
		k_vals,prob_vals = self.tuple_of_probabilities
		working_indices = [i for i,v in enumerate(k_vals) if (v >= a and v<= b)]
		working_prob_vals = [prob_vals[i] for i in working_indices]
		return sum(working_prob_vals)
		
	def plot_distribution(self):
		"""Plots the theoretical probability distribution for the random walk.
		"""
		k_vals,prob_vals = self.tuple_of_probabilities
		
		plt.figure("Probability Distribution of Random Walk, Theoretical")
		plt.scatter(k_vals,prob_vals,s=4)
		plt.xlim((-self.n-1,self.n+1))
		
		plt.xlabel("x\u2099 - Position after n jumps")
		plt.ylabel("Probability")
		plt.suptitle("Random Walk: p={}, n={}, \u0394x={}".format(self.p,self.n,self.delta_x))

	def plot_monte_carlo(self,number_of_trials=2000,histogram=False,show=True): # Deal with histogram later
		"""Runs a Monte Carlo simulation of the random walk for a specified number of trials.
		It then plots the results as a frequency distribution.
		
		Mean and variance values of the Monte Carlo simulation can now be retrieved.
		"""
		trial_data = []
		for _ in range(number_of_trials):
			steps = self._random_walk_simulation()
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
		
		if show == False:
			return trial_data
			
		plt.figure("Monte Carlo Simulation of Random Walk")
		plt.scatter(x_n,counts,s=4)
		plt.xlim((-self.n-1,self.n+1))
		
		plt.xlabel("x\u2099 - Position after n jumps")
		plt.ylabel("Frequency")
		plt.suptitle("Monte Carlo Simulation of Random Walk: p={}, n={}, \u0394x={}, N={}".format(
													self.p,self.n,self.delta_x,number_of_trials))
		return trial_data
		
	def factorial(self,num):
		# Gosper approximation was meant to fix num > 1000 case but this number computes to 0.0
		# = np.sqrt((2*num+(1/3))*np.pi) * (num**num) * (np.e**(-num))
		res = 1
		for i in range(1,int(num+1)):
			res += i
		return res 
			
	def _calculate_binom(self,n,r):
		#From formula: nCr = n! / r!(n-r)!
		return self.factorial(n) / ( self.factorial(r) * self.factorial(n-r) )
			
	def _calculate_mean(self):
		return (self.p - (1-self.p)) * self.n * self.delta_x
		
	def _calculate_variance(self):
		return 4 * self.p * (1-self.p) * self.n * (self.delta_x)**2
		
	def _calculate_probability(self,k):
		"""Calculates the probability that x_n = k * delta_x.
		
		This method uses the values of n and p in its calculations.
		"""
		if abs(k * self.delta_x) > (3 * np.sqrt(self.variance)):
			return 0.0
		binom_coeff = special.binom(self.n,(self.n + k)/2)
		b_value = binom_coeff * ((self.p) ** ((self.n + k)/2)) * ((1-self.p) ** ((self.n - k)/2))
		return b_value
		
	def _get_tuple_of_probabilities(self):
		"""Gets a tuple of the form (k-values,probabilities) in the range [-n,n].
		"""
		k_array = np.arange(-self.n,self.n+1,2)
		probability_array = []
		
		for k in k_array:
			probability_array.append(self._calculate_probability(k))
			
		return (k_array,probability_array)
		
	def _calculate_mean_distance_theoretical(self):
		x_mean_distance = 0
		x_vals,prob_vals = self.tuple_of_probabilities
		for i in range(len(x_vals)):
			x_val, prob = x_vals[i], prob_vals[i]
			x_distance = abs(x_val - self.x_initial)
			x_weighted = x_distance * prob
			x_mean_distance += x_weighted
		return x_mean_distance
		
	def _random_walk_simulation(self):
		steps = np.random.choice( [-1,1], self.n, p = [1-self.p,self.p] )
		return steps

	def random_walk_draw(self,num_plots):
		t_x_arrays = []
		t_max = self.n
		for _ in range(num_plots):
			current_x = self.x_initial
			x_array = [current_x]
			t_array = range(t_max + 1)
			steps = self._random_walk_simulation()
			for s in steps:
				current_x += s
				x_array.append(current_x)
			t_x_arrays.append( [x_array,t_array] )
		
		
		fig = plt.figure('Random walk live simulation')
		ax = fig.add_subplot(1,1,1)

		self.index = 0
		def update(i):
			ax.set_ylim([-self.n,self.n])
			ax.set_xlim([0,self.n])
			for i in t_x_arrays:
				x_vals,t_vals = i
				ax.plot(t_vals[:self.index], x_vals[:self.index])
			self.index += 1
		a = anim.FuncAnimation(fig, update, frames=t_max+1, repeat=False)
		plt.show()
		
	def __getattr__(self,name):
		if name in ['mc_mean','mc_variance']:
			print("You must run a Monte Carlo simulation first before calling '{}'".format(name))
		raise AttributeError
		
