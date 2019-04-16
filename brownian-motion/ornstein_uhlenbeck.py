#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Aaron Lalor-Fitzpatrick"


import random
import numpy as np
import matplotlib.pyplot as plt

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size


x_initial 		= 0
tau 			= 1
sigma 			= 1
dt 				= 0.001
t_max 			= 10
timesteps = np.arange(0,t_max,dt)
number_of_steps = len(timesteps)



def main():
	number_of_sims=100
	plot_ou()
	print("\nOrnstein-Uhlenbeck Brownian Motion Simuation, dt = {}\n".format(dt))
	print("Number of Monte Carlo trials, n = {}\n".format(number_of_sims))
	monte_carlo_arr = plot_ou_monte_carlo(number_of_sims)
	plot_average_path(monte_carlo_arr)


def ou_sim(x_init):
	x_vals = []
	x_old = x_init
	for ti in timesteps:
		x_new = x_old - ((dt/tau)*x_old) + (((sigma*np.sqrt(dt))/np.sqrt(tau))*random.gauss(0,1))
		x_vals.append(x_new)
		x_old = x_new
	return x_vals	
	
	
def plot_ou():
	fig = plt.figure("Simulation of Ornstein-Uhlenbeck Brownian Motion, single path")
	fig.suptitle("Ornstein-Uhlenbeck Brownian Motion: Single path \n\u03C4 = {}, \u03C3 = {}, dt = {}".format(tau,sigma,dt))
	ax = plt.subplot(111)
	ax.set_xlabel('t (time elapsed)')
	ax.set_ylabel('v(t) (velocity)')
	
	ou_vals = ou_sim(x_initial)
	plt.scatter(timesteps,ou_vals,s=0.05)
	plt.show()
	return ou_vals
	
	
def plot_ou_monte_carlo(number_of_sims):
	fig = plt.figure("Monte Carlo simulation of Ornstein-Uhlenbeck Brownian motion, {} trials".format(number_of_sims))
	fig.suptitle("Ornstein-Uhlenbeck Brownian Motion: Monte Carlo trials \n\u03C4 = {}, \u03C3 = {}, dt = {}".format(tau,sigma,dt))
	ax = plt.subplot(111)
	ou_array = []
	
	for _ in range(number_of_sims):
		ou_vals = ou_sim(x_initial)
		ax.set_xlabel('t (time elapsed)')
		ax.set_ylabel('v(t) (velocity)')
		plt.scatter(timesteps,ou_vals,s=0.1)
		ou_array.append(ou_vals)
	plt.show()
	return ou_array


def plot_average_path(ou_array):
	x_vals = [] # Will I keep this in the final plot?
	average_x_vals = []
	average_x_squared_vals = []
	std_vals = []
	single_path = ou_array[0] # Will I keep this in the final plot?
	for step in np.arange(0,number_of_steps):
		vals = []
		squared_vals = []
		for i in range(len(ou_array)):
			vals.append(ou_array[i][step])
			squared_vals.append((ou_array[i][step])**2)
		expected_val = np.average(vals)
		expected_val_squared = np.average(squared_vals)
		std = np.sqrt(np.average((np.array(vals)-expected_val)**2))
		x_vals.append(single_path[step])
		average_x_vals.append(expected_val)
		average_x_squared_vals.append(expected_val_squared)
		std_vals.append(std)
	std_minus = np.array(average_x_vals) - np.array(std_vals)
	std_plus = np.array(average_x_vals) +  np.array(std_vals)
	predicted_ss_v_squared = round(((sigma**2)*tau)/2,3)
	
	# PLOTTING <v(t)>
	fig = plt.figure("Ornstein-Uhlenbeck Brownian motion, expected path, {} trials".format(len(ou_array)))
	fig.suptitle("Ornstein-Uhlenbeck Brownian Motion: expected path \n\u03C4 = {}, \u03C3 = {}, dt = {}".format(tau,sigma,dt))
	ax = plt.subplot(111)
	ax.set_xlabel('t (time elapsed)')
	ax.set_ylabel('v(t) (average velocity)')
	#plt.scatter(timesteps,x_vals,s=0.1,label='')
	plt.scatter(timesteps,average_x_vals,s=0.1,label='⟨v(t)⟩')
	plt.scatter(timesteps,std_minus,s=0.2,label='⟨v(t)⟩-\u03C3(t)')
	plt.scatter(timesteps,std_plus,s=0.2,label='⟨v(t)⟩+\u03C3(t)')
	ax.legend(loc='best', bbox_to_anchor=(0.86, 1.00), shadow=True, ncol=1)
	plt.show()
	
	# PLOTTING <v(t)^2>
	fig = plt.figure("Ornstein-Uhlenbeck Brownian motion, average squared velocity, {} trials".format(len(ou_array)))
	fig.suptitle("Ornstein-Uhlenbeck Brownian Motion: average squared velocity, {} trials \n\u03C4 = {}, \u03C3 = {}, dt = {} \nPredicted ⟨$v^2$(t)⟩ at steady-state: ⟨$v^2$⟩$_e$$_q$ = {}".format(len(ou_array),tau,sigma,dt,predicted_ss_v_squared))
	ax = plt.subplot(111)
	ax.set_xlabel('t (time elapsed)')
	ax.set_ylabel('⟨$v^2$(t)⟩ (average velocity squared)')
	plt.scatter(timesteps,average_x_squared_vals,s=0.1)
	
	plt.show()
	return {'x':x_vals,'average_x_squared':average_x_squared_vals,'std_minus':std_minus,'std_plus':std_plus}


if __name__ == "__main__":
	main()
