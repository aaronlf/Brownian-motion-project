#!/usr/bin/env python3

__author__ = "Aaron Lalor-Fitzpatrick"

import random
import numpy as np
import matplotlib.pyplot as plt


s_initial 		= 10
mu 				= 0.05
sigma 			= 0.1
dt 				= 0.05
t_max 			= 10
number_of_steps = int(round(t_max/dt))
timesteps = np.arange(0,t_max,dt)



def random_term():
	return ((mu - ((sigma)**2)/2) * dt) + (sigma * (np.sqrt(dt)) * random.gauss(0,1))


def get_all_random_terms():
	random_terms = []
	for _ in range(number_of_steps-1):
		random_terms.append(random_term())
	return random_terms


def gbm_sim(s_init):
	random_terms = get_all_random_terms()
	s_vals = [s_init]
	s_new = s_init
	for i,r_term in enumerate(random_terms):
		s_new = s_new * (np.exp(r_term))
		s_vals.append(s_new)
	return s_vals	
	
	
def plot_gbm(number_of_sims):
	fig = plt.figure("Simulation of Geometric Brownian Motion, {} trials".format(number_of_sims))
	fig.suptitle("Geometric Brownian Motion, \u03BC = {}, \u03C3 = {}, dt = {}".format(mu,sigma,dt))
	gbm_array = []
	
	for _ in range(number_of_sims):
		gbm_vals = gbm_sim(s_initial)
		plt.plot(timesteps,gbm_vals,linewidth=0.5)
		gbm_array.append(gbm_vals)
	
	return gbm_array

def compare_mu_vals(gbm_array):
	computed_mu_vals = []
	for step in np.arange(1,number_of_steps):
		vals = []
		for i in range(len(gbm_arr)):
			vals.append(gbm_arr[i][step])
		expected_val = np.average(vals)
		computed_mu_vals.append( (np.log(expected_val) - np.log(s_initial)) / (step*dt) )
	computed_mu = np.mean(computed_mu_vals)
	print("μ given: {}	Average μ found: {}".format(mu,round(computed_mu,4)))
		
		
def compare_sigma_vals(gbm_array):
	computed_sigma_vals = []
	for step in np.arange(1,number_of_steps):
		vals = []
		for i in range(len(gbm_arr)):
			vals.append(gbm_arr[i][step])
		expected_val = np.average(vals)
		variance = np.var(vals)
		computed_sigma_vals.append(np.sqrt(np.log((variance/((expected_val)**2))+1)/(step*dt)))
	computed_sigma = np.mean(computed_sigma_vals)
	print("σ given: {}	Average σ found: {}".format(sigma,round(computed_sigma,4)))
	


gbm_arr = plot_gbm(number_of_sims=100)
compare_mu_vals(gbm_arr)
compare_sigma_vals(gbm_arr)
plt.show()
