#!/usr/bin/env python3

__author__ = "Aaron Lalor-Fitzpatrick"

import random
import numpy as np
import matplotlib.pyplot as plt


x_initial 		= 10
mu 				= 0.1
sigma 			= 0.2
dt 				= 0.01
t_max 			= 5
number_of_steps = int(round(t_max/dt))
timesteps = np.arange(0,t_max,dt)



def main():
	abm_arr = plot_abm(number_of_sims=100)
	compare_mu_vals(abm_arr)
	compare_sigma_vals(abm_arr)
	plt.show()
	
	
def random_term():
	return (mu * dt) + (sigma * (np.sqrt(dt)) * random.gauss(0,1))
	
	
def get_all_random_terms():
	random_terms = []
	for _ in range(number_of_steps-1):
		random_terms.append(random_term())
	return random_terms


def abm_sim(x_init):
	random_terms = get_all_random_terms()
	x_vals = [x_init]
	x_new = x_init
	for r_term in random_terms:
		x_new += (r_term)
		x_vals.append(x_new)
	return x_vals	
	
	
def plot_abm(number_of_sims):
	fig = plt.figure("Simulation of Arithmetic Brownian Motion, {} trials".format(number_of_sims))
	fig.suptitle("Arithmetic Brownian Motion, \u03BC = {}, \u03C3 = {}, dt = {}".format(mu,sigma,dt))
	abm_array = []
	
	for _ in range(number_of_sims):
		abm_vals = abm_sim(x_initial)
		plt.plot(timesteps,abm_vals,linewidth=0.5)
		abm_array.append(abm_vals)
	
	return abm_array

def compare_mu_vals(abm_array):
	computed_mu_vals = []
	for step in np.arange(1,number_of_steps):
		vals = []
		for i in range(len(abm_array)):
			vals.append(abm_array[i][step])
		expected_val = np.average(vals)
		computed_mu_vals.append( ((expected_val) - x_initial) / (step*dt) )
	computed_mu = np.mean(computed_mu_vals)
	print("μ given: {}	Average μ found: {}".format(mu,round(computed_mu,4)))
		
		
def compare_sigma_vals(abm_array):
	computed_sigma_vals = []
	for step in np.arange(1,number_of_steps):
		vals = []
		for i in range(len(abm_array)):
			vals.append(abm_array[i][step])
		variance = np.var(vals)
		computed_sigma_vals.append( np.sqrt(variance/(step*dt)) )
	computed_sigma = np.mean(computed_sigma_vals)
	print("σ given: {}	Average σ found: {}".format(sigma,round(computed_sigma,4)))
	


if __name__ == "__main__":
	main()
