#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import time
import numpy as np
import pickle
from bisect import bisect_left

	
#---------------------------------------------------------------------------------------------------
### CLASS TO DEMONSTRATE FEL TEMPORAL COHERENCE/INCOHERECE FOR UNIFORMLY DISTRIBUTED ELECTRON PULSE	
#---------------------------------------------------------------------------------------------------	
	
class FEL_Uniform:
	
	def __init__(self,sigma,omega,num_data_points=500):
		self.sigma = sigma
		self.omega = omega
		self.num_data_points = num_data_points
		
		
	def coherent_rad(self,t,sigma,omega):
		return np.exp(-(t**2)/(4*(sigma**2))) * np.cos(omega*t)
		
	def model_coherent_rad_once(self,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		time_vals = np.linspace(-20,20,self.num_data_points)
		e_vals = []
		for t in time_vals:
			e_vals.append(self.coherent_rad(t,sigma,omega))
		return {'x_vals':time_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9 = {}'.format(sigma,omega)}
		
	def incoherent_rad(self,t,tj,sigma,omega,bunch_length):
		return (np.exp((-(t-tj)**2)/(4*(sigma**2)))) * np.cos(omega*(t-tj))
	
	def model_incoherent_rad_once(self,bunch_length,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		time_vals = np.linspace(-(bunch_length*0.5)-(4*sigma),(bunch_length*0.5)+(4*sigma),self.num_data_points)
		e_vals = []
		tj = random.uniform(-(bunch_length*0.5),(bunch_length*0.5))
		for t in time_vals:
			e_vals.append(self.incoherent_rad(t,tj,sigma,omega,bunch_length))
		return {'x_vals':time_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9 = {}'.format(sigma,omega)}
		
	def get_superposition(self,input_vals_list):
		output_vals = np.zeros(shape=len(input_vals_list[0]['y_vals']),dtype=complex)
		for input_vals in input_vals_list:
			output_vals += np.array(input_vals['y_vals'])
		return {
				'x_vals':input_vals_list[0]['x_vals'],
				'y_vals':output_vals,
				'label':None
				}
	
	def get_intensity(self,e_superposition):
		e_vals = e_superposition['y_vals']
		conj_e_vals = np.conjugate(e_vals)
		intensity_vals = abs(e_vals * conj_e_vals)
		return {
				'x_vals':e_superposition['x_vals'],
				'y_vals':intensity_vals,
				'label':None
				}			
				
	def get_num_l_modes(self,bunch_length,sigma=None):
		if sigma==None:
			sigma = self.sigma
		M_L = round(bunch_length/(4*sigma))
		return M_L

	def incoherent_rad_omega(self,tj,sigma,omega,bunch_length,omega1=None):
		if omega1 == None:
			omega1 = self.omega
		sigma_FT = 1/(2*sigma)
		return (np.exp(-((omega-omega1)**2)/(4*(sigma_FT**2))) * np.cos(omega*tj)) * (np.sqrt(np.pi)/sigma_FT) 
	
	def model_incoherent_rad_once_omega(self,bunch_length,sigma=None,omega1=None):
		if sigma == None:
			sigma = self.sigma
		if omega1 == None:
			omega1 = self.omega
		omega_vals = np.linspace(omega1-1,omega1+1,self.num_data_points)
		e_vals = []
		tj = random.uniform(-(bunch_length*0.5),(bunch_length*0.5))
		for omega in omega_vals:
			e_vals.append(self.incoherent_rad_omega(tj,sigma,omega,bunch_length))
		delta_omega_vals = omega_vals - omega1
		return {'x_vals':delta_omega_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9$_1 = {}'.format(sigma,omega1)}
		
			
	def autocorrelation_time(self,t,num_pulses,bunch_length,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
			
		incoherent_vals_list = [
			self.model_incoherent_rad_once(bunch_length,sigma=None,omega=None) for i in range(num_pulses)
			]
		superposition_pulses = self.get_superposition(incoherent_vals_list)
		time_vals = superposition_pulses['x_vals']
		E = superposition_pulses['y_vals'][bisect_left(time_vals, t)]
		E_conj = np.conjugate(E)
		
		c1_vals = []	# 1st order AC function
		c2_vals = []	# 2nd order AC function
		tau_vals = []
		for i,time_val in enumerate(time_vals):
			E_tau = superposition_pulses['y_vals'][i]
			E_tau_conj = np.conjugate(E_tau)
			tau_vals.append(time_val - t)
			c1_vals.append( E * E_tau_conj )
			c2_vals.append( E * E_conj * E_tau * E_tau_conj )
		c1_dict = {'x_vals':tau_vals,'y_vals':c1_vals,'label':None}
		c2_dict = {'x_vals':tau_vals,'y_vals':c2_vals,'label':None}
		return {'c1_dict':c1_dict,'c2_dict':c2_dict}
		
		
	def autocorrelation_freq(self,delta_omega,num_pulses,bunch_length,sigma=None,omega1=None):
		if sigma == None:
			sigma = self.sigma
		if omega1 == None:
			omega1 = self.omega
			
		incoherent_vals_list = [
			self.model_incoherent_rad_once_omega(bunch_length,sigma=None,omega1=None) for i in range(num_pulses)
			]
		superposition_pulses = self.get_superposition(incoherent_vals_list)
		delta_omega_vals = superposition_pulses['x_vals']
		E = superposition_pulses['y_vals'][bisect_left(delta_omega_vals, delta_omega)]
		E_conj = np.conjugate(E)
		
		c1_vals = []	# 1st order AC function
		c2_vals = []	# 2nd order AC function
		tau_freq_vals = []
		for i,delta_omega_val in enumerate(delta_omega_vals):
			E_tau = superposition_pulses['y_vals'][i]
			E_tau_conj = np.conjugate(E_tau)
			tau_freq_vals.append(delta_omega_val - delta_omega)
			c1_vals.append( E * E_tau_conj )
			c2_vals.append( E * E_conj * E_tau * E_tau_conj )
		c1_dict = {'x_vals':delta_omega_vals,'y_vals':c1_vals,'label':None}
		c2_dict = {'x_vals':delta_omega_vals,'y_vals':c2_vals,'label':None}
		return {'c1_dict':c1_dict,'c2_dict':c2_dict}

class FEL_Uniform_Complex(FEL_Uniform):
	
	def incoherent_rad(self,t,tj,sigma,omega,bunch_length):
		# Models (1.19) in notes from Lampros A.A Nikolopoulos
		X = 1 + 1j/np.sqrt(3)
		return np.exp( (-X*(t-tj)**2)/(4*(sigma**2)) + (1j*omega*tj) )
	
	def incoherent_rad_omega(self,tj,sigma,omega,bunch_length,omega1=None):
		if omega1 == None:
			omega1 = self.omega
		sigma_FT = 1/(2*sigma)
		return (np.exp(-((omega-omega1)**2)/(4*(sigma_FT**2))) * np.cos(omega*tj)) * (np.sqrt(np.pi)/sigma_FT) 
