#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Aaron Lalor-Fitzpatrick"

import random
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from threading import Lock
from gbm_euler import Euler
import data_storage as ds

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size



class Milstein(Euler):
			
	def __init__(self,s_initial=10,mu=0.2,sigma=0.2,dt=0.04,t_max=20,number_of_sims=1000):
		super().__init__(s_initial,mu,sigma,dt,t_max,number_of_sims)
		self.name = 'Milstein Method'
		self.random_vals = ds.random_vals1 
		
		
	def main(self):
		print('MILSTEIN METHOD')
		print("\nGeomteric Brownian Motion Simuation, dt = {}\n".format(self.dt))
		print("Number of Monte Carlo trials, n = {}\n\n".format(self.number_of_sims))
		self.prepare_random_numbers()
		self.sim_for_comparison(print_res=True)
		self.plot_gbm()
		
		
	def random_milstein_term(self,s_i):
		return (0.5 * (self.sigma ** 2) * s_i) * ( (self.dt * (random.gauss(0,1)  ** 2)) - self.dt)


	def gbm_sim(self,sim_num):
		s_vals = [self.s_initial]
		s_new = self.s_initial
		for i,_ in enumerate(self.timesteps):
			if _ == self.timesteps[-1]:
				break
			r_euler_term = self.random_euler_term(s_new,i,sim_num)
			r_milstein_term = self.random_milstein_term(s_new)
			s_new = s_new + r_euler_term + r_milstein_term
			s_vals.append(s_new)
		return s_vals	
		
		
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
		ds.compared_mu_vals_m.append(compared_mu_vals)
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
		ds.compared_sigma_vals_m.append(compared_sigma_vals)
		if print_res == True:
			print("σ given: {}	Average σ found: {}".format(self.sigma,round(computed_sigma,4)))
			print("95% CI Margin of error of σ results: {}".	format(round(sigma_mrg_err,6)))
			print("95% confidence interval range: [{},{}]".format(
						round(computed_sigma - sigma_mrg_err,5), round(computed_sigma + sigma_mrg_err,5)))
			print("Relative error from theoretical result: {}%\n".format(rel_error))
		return compared_sigma_vals


if __name__ == "__main__":
	m = Milstein()
	m.main()

