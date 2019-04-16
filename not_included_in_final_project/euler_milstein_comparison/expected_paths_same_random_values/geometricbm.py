#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Aaron Lalor-Fitzpatrick"

import random
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import data_storage as ds

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
	
	
class Analytical:
	
	def __init__(self,s_initial=10,mu=0.2,sigma=0.2,dt=0.04,t_max=20,number_of_sims=1000):
		self.s_initial 			= s_initial
		self.mu 				= mu
		self.sigma 				= sigma
		self.dt 				= dt
		self.t_max 				= t_max
		self.timesteps 			= np.arange(0,t_max,dt)
		self.number_of_steps 	= len(self.timesteps)
		self.gbm_arr 			= []
		self.number_of_sims 	= number_of_sims
		self.compared_mu_vals 	= []
		self.compared_sigma_vals= []
		self.name 				= 'Analytical Method'
		self.theoretical_ev 	= self.theoretical_ev()
		
		
	def prepare_random_numbers(self,list_num=1):
		# Assign random variables to data_storage
		if list_num == 1:
			ds.random_vals1 = [[random.gauss(0,1) for _ in range(10000)] for i in range(self.number_of_sims)]
			self.random_vals = ds.random_vals1
		elif list_num == 2:
			ds.random_vals2 = [[random.gauss(0,1) for _ in range(10000)] for i in range(self.number_of_sims)]
			self.random_vals = ds.random_vals2
		elif list_num == 3:
			ds.random_vals3 = [[random.gauss(0,1) for _ in range(10000)] for i in range(self.number_of_sims)]
			self.random_vals = ds.random_vals3 
			
				
	def main(self):
		print('ANALYTICAL GEOMETRIC BROWNIAN MOTION')
		print("\nGeomteric Brownian Motion Simuation, dt = {}\n".format(self.dt))
		print("Number of Monte Carlo trials, n = {}\n\n".format(self.number_of_sims))
		self.prepare_random_numbers()
		self.sim_for_comparison()
		self.plot_gbm()

		
	def theoretical_ev(self):
		return self.s_initial * np.exp(self.mu * self.t_max)
		
			
	def random_term(self,i,sim_num):
		return ((self.mu - ((self.sigma)**2)/2) * self.dt) + (self.sigma * (np.sqrt(self.dt)) * self.random_vals[sim_num][i])


	def get_all_random_terms(self,sim_num):
		random_terms = []
		for i,_ in enumerate(self.timesteps):
			if _ == self.timesteps[-1]:
				break
			random_terms.append(self.random_term(i,sim_num))
		return random_terms


	def gbm_sim(self,sim_num):
		random_terms = self.get_all_random_terms(sim_num)
		s_vals = [self.s_initial]
		s_new = self.s_initial
		for r_term in random_terms:
			s_new = s_new * (np.exp(r_term))
			s_vals.append(s_new)
		return s_vals	
		
		
	def plot_gbm(self):
		fig = plt.figure("Simulation of Geometric Brownian Motion, {} - {} trials".format(self.name,self.number_of_sims))
		fig.suptitle("Geometric Brownian Motion, \u03BC = {}, \u03C3 = {}, dt = {}, {} trials".format(self.mu,self.sigma,self.dt,self.number_of_sims))
		if self.gbm_arr != []:
			for gbm_sub_arr in self.gbm_arr:
				plt.plot(self.timesteps,gbm_sub_arr,linewidth=0.5)
				gbm_array = self.gbm_arr
		else:
			gbm_array = []
			for num in range(self.number_of_sims):
				gbm_vals = self.gbm_sim(num)
				plt.plot(self.timesteps,gbm_vals,linewidth=0.5)
				gbm_array.append(gbm_vals)
		plt.show()
		return gbm_array
	

	def multiple_gbm_sims(self,number_of_sims):
		results_from_sims = []
		for num in range(number_of_sims):
			gbm_vals = self.gbm_sim(num)
			results_from_sims.append(gbm_vals)
			self.gbm_arr.append(gbm_vals)
		return results_from_sims
			
			
	def compare_mu_vals(self,gbm_array,print_res=False):
		computed_mu_vals = []
		computed_mu_rel_errors = []
		expected_vals = []
		for step in np.arange(1,self.number_of_steps):
			vals = []
			for i in range(len(gbm_array)):
				vals.append(gbm_array[i][step])
			expected_val = np.average(vals)
			mu = (np.log(expected_val) - np.log(self.s_initial)) / (step*self.dt)
			single_rel_error = 100 * (abs(mu - self.mu) / self.mu)
			expected_vals.append(expected_val)
			computed_mu_vals.append(mu)
			computed_mu_rel_errors.append(single_rel_error)
			if step == np.arange(1,self.number_of_steps)[-1]:
				strong_convergence_term = np.mean(np.absolute(np.array(vals) - self.theoretical_ev))
		computed_mu = np.mean(computed_mu_vals)
		mu_std = np.std(computed_mu_vals)
		mu_mrg_err = 1.959964 * mu_std / np.sqrt(len(gbm_array))
		rel_error = round(100*(abs(self.mu-computed_mu)/self.mu),4)
		rel_mrg_err = round(100*(abs(mu_std)/computed_mu),4)
		compared_mu_vals = {
							'obj':self,
							'expected_vals':expected_vals,
							'strong_convergence_term':strong_convergence_term,
							'computed_mu_vals':computed_mu_vals,
							'computed_mu_rel_errors':computed_mu_rel_errors,
							'mu':computed_mu,
							'mu_mrg_err':mu_mrg_err,
							'rel_error':rel_error,
							'rel_mrg_err':rel_mrg_err,
							'mu_std':mu_std
							}
		ds.compared_mu_vals_a.append(compared_mu_vals)
		if print_res == True:
			print("μ given: {}	Average μ found: {}".format(self.mu,round(computed_mu,4)))
			print("95% CI Margin of error of μ results: {}".format(round(mu_mrg_err,6)))
			print("95% confidence interval range: [{},{}]".format(
									round(computed_mu - mu_mrg_err,5), round(computed_mu + mu_mrg_err,5)))
			print("Relative error from theoretical result: {}%\n".format(rel_error))
		return compared_mu_vals
		
			
	def compare_sigma_vals(self,gbm_array,print_res=False):
		computed_sigma_vals = []
		computed_sigma_rel_errors = []
		expected_vals = []
		for step in np.arange(1,self.number_of_steps):
			vals = []
			for i in range(len(gbm_array)):
				vals.append(gbm_array[i][step])
			expected_val = np.average(vals)
			variance = np.var(vals)
			sigma = np.sqrt(np.log((variance/((expected_val)**2))+1)/(step*self.dt))
			single_rel_error = 100 * (abs(sigma-self.sigma) / self.sigma)
			expected_vals.append(expected_val)
			computed_sigma_vals.append(sigma)
			computed_sigma_rel_errors.append(single_rel_error)
		computed_sigma = np.mean(computed_sigma_vals)
		sigma_std = np.std(computed_sigma_vals)
		sigma_mrg_err = 1.959964 * sigma_std / np.sqrt(len(gbm_array))
		rel_error = round(100*(abs(self.sigma-computed_sigma)/self.sigma),4)
		rel_mrg_err = round(100*(abs(sigma_std)/computed_sigma),4)
		compared_sigma_vals = {
								'obj':self,
								'expected_vals':expected_vals,
								'computed_sigma_vals':computed_sigma_vals,
								'computed_sigma_rel_errors':computed_sigma_rel_errors,
								'sigma':computed_sigma,
								'sigma_mrg_err':sigma_mrg_err,
								'rel_error':rel_error,
								'rel_mrg_err':rel_mrg_err,
								'sigma_std':sigma_std
								}
		ds.compared_sigma_vals_a.append(compared_sigma_vals)
		if print_res == True:
			print("σ given: {}	Average σ found: {}".format(self.sigma,round(computed_sigma,4)))
			print("95% CI Margin of error of σ results: {}".	format(round(sigma_mrg_err,6)))
			print("95% confidence interval range: [{},{}]".format(
						round(computed_sigma - sigma_mrg_err,5), round(computed_sigma + sigma_mrg_err,5)))
			print("Relative error from theoretical result: {}%\n".format(rel_error))
		return compared_sigma_vals
	
	def sim_for_comparison(self,print_res=False):
		gbm_arr = self.multiple_gbm_sims(self.number_of_sims)
		self.compare_mu_vals(gbm_arr,print_res)
		self.compare_sigma_vals(gbm_arr,print_res)

if __name__ == "__main__":
	a = Analytical()
	a.main()
