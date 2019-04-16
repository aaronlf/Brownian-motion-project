#!/usr/bin/env python3

__author__ = "Aaron Lalor-Fitzpatrick"

from scipy import special
import random
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
	
	###################### CONSTRUCTOR ######################
	
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
		
		
	###################### USER-CALLABLE METHODS ######################
	
	def get_confidence_interval(self,a,b):
		"""Returns the theoretical probability that x_n will be between the values a and b.
		"""
		k_vals,prob_vals = self.tuple_of_probabilities
		working_indices = [i for i,v in enumerate(k_vals) if (v >= a and v<= b)]
		working_prob_vals = [prob_vals[i] for i in working_indices]
		return sum(working_prob_vals)
		
		
	def plot_distribution(self,show=True):
		"""Plots the theoretical probability distribution for the random walk.
		"""
		k_vals,prob_vals = self.tuple_of_probabilities
		
		plt.figure("Probability distribution of Random Walk, theoretical")
		plt.scatter(k_vals,prob_vals,s=4)
		plt.xlim((-self.n-1,self.n+1))
		
		plt.xlabel("x\u2099 - Position after n jumps")
		plt.ylabel("Probability")
		plt.suptitle("Random Walk: p={}, n={}, \u0394x={}".format(self.p,self.n,self.delta_x))
		if show == True:
			plt.show()


	def run_monte_carlo(self,number_of_trials=2000,plot=True,histogram=False,show=True,overlay=False):
		"""Runs a Monte Carlo simulation of the random walk for a specified number of trials.
		It then plots the results as a frequency distribution.
		
		Mean and variance values of the Monte Carlo simulation can be retrieved by calling mc.mean
		and mc.variance, respectively.
		
		#####  Method parameters  #####
		- number_of_trials: Integer number of Monte Carlo trials to be conducted. 2000 by default.
		- plot: If true, plot the monte carlo trial results
		- histogram: If true, plot a histogram of the Monte Carlo results
		- show: If true, plt.show() will be called for each figure, one at a time.
		- overlay: If true, the theoretical pdf will be plotted on top of the Monte Carlo trial
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
		
		if histogram == True:
			plt.figure("Monte Carlo simulation of random walk - results")
			plt.hist( trial_data, bins=int(round(np.sqrt(self.n))) )
			plt.suptitle("Histogram of Monte Carlo simulation results: p={},n={}, \u0394x={}, N={}".format(
													self.p,self.n,self.delta_x,number_of_trials))
			if show == True:
				plt.show()
			
		
		if plot == False:
			return trial_data
			
		plt.figure("Monte Carlo simulation of Random Walk")
		plt.scatter(x_n,counts,s=4)
		plt.xlim((-self.n-1,self.n+1))
		plt.xlabel("x\u2099 - Position after n jumps")
		plt.ylabel("Frequency")
		plt.suptitle("Monte Carlo simulation of random walk: p={}, n={}, \u0394x={}, N={}".format(
													self.p,self.n,self.delta_x,number_of_trials))
		
		if overlay == True: # IF TRUE, PLOT THEORETICAL RESULTS OVER MONTE CARLO RESULTS
			k_vals,prob_vals = self.tuple_of_probabilities
			prob_vals = [p*number_of_trials for p in prob_vals]
			plt.scatter(k_vals,prob_vals,s=4)	
		
		if show == True:
			plt.show()		
								
		return trial_data

	
	def random_walk_draw(self,num_plots,animated=False,show=True):
		"""This method produces an animated simulation of a 1D random walk.
		"""
		
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
		
		
		fig = plt.figure('Random walk simulation')
		ax = fig.add_subplot(1,1,1)
		ax.set_ylim([(round(min(x_array) - np.sqrt(self.n)*3)),round(max(x_array) + np.sqrt(self.n)*3)])
		ax.set_xlim([-(round(np.sqrt(self.n))),self.n+(round(np.sqrt(self.n)))])
		
		if animated == True: # THIS CASE CURRENTLY HAS BUG FOR SOME REASON. CODE IS IDENTICAL TO 2D ANIMATION?
			fig.suptitle('Simulation of 1D random walk, live')
			self.index = 0
			def update(i):
				ax.clear()
				ax.set_ylim([(round(min(x_array) - np.sqrt(self.n)*3)), round(max(x_array) + np.sqrt(self.n)*3)])
				ax.set_xlim([-(round(np.sqrt(self.n))), self.n+(round(np.sqrt(self.n)))])
				for i in t_x_arrays:
					x_vals,t_vals = i 
					ax.plot(t_vals[:self.index], x_vals[:self.index])
				self.index += 1
			a = anim.FuncAnimation(fig, update, frames=self.n, repeat=False,interval=10)
		else:
			fig.suptitle('Simulation of 1D random walk, static')
			for i in t_x_arrays:
				x_vals,t_vals = i
				ax.plot(t_vals, x_vals)
			
		if show == True:
			plt.show()

			
	###################### PRIVATE METHODS ######################
			
	def _calculate_binom(self,n,r):
		#From formula: nCr = n! / r!(n-r)!
		return self._factorial(n) / ( self._factorial(r) * self._factorial(n-r) )

			
	def _calculate_mean(self):
		return (self.p - (1-self.p)) * self.n * self.delta_x
		
	
	def _calculate_mean_distance_theoretical(self):
		"""Returns the theoretical average distance from x_initial.
		"""
		x_mean_distance = 0
		x_vals,prob_vals = self.tuple_of_probabilities
		for i in range(len(x_vals)):
			x_val, prob = x_vals[i], prob_vals[i]
			x_distance = abs(x_val - self.x_initial)
			x_weighted = x_distance * prob
			x_mean_distance += x_weighted
		return x_mean_distance
		
		
	def _calculate_probability(self,k):
		"""Calculates the probability that x_n = k * delta_x.
		
		This method uses the values of n and p in its calculations.
		"""
		if abs(k * self.delta_x) > (3 * np.sqrt(self.variance)):
			return 0.0
		binom_coeff = special.binom(self.n,(self.n + k)/2)
		b_value = binom_coeff * ((self.p) ** ((self.n + k)/2)) * ((1-self.p) ** ((self.n - k)/2))
		return b_value
		
			
	def _calculate_variance(self):
		return 4 * self.p * (1-self.p) * self.n * (self.delta_x)**2

	
	def _factorial(self,num):
		res = 1
		for i in range(1,int(num+1)):
			res += i
		return res 
		
			
	def _get_tuple_of_probabilities(self):
		"""Gets a tuple of the form (k-values,probabilities) in the range [-n,n].
		"""
		k_array = np.arange(-self.n,self.n+1,2)
		probability_array = []
		
		for k in k_array:
			probability_array.append(self._calculate_probability(k))
			
		return (k_array,probability_array)
		

	def _random_walk_simulation(self):
		steps = np.random.choice( [-1,1], self.n, p = [1-self.p,self.p] )
		return steps

	
	###################### OPERATOR OVERLOADING ######################
		
	def __getattr__(self,name):
		if name in ['mc_mean','mc_variance']:
			print("You must run a Monte Carlo simulation first before calling '{}'".format(name))
		raise AttributeError

