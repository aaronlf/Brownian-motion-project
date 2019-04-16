#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import time
import numpy as np
import pickle
from bisect import bisect_left
import pylab as pl

	
#---------------------------------------------------------------------------------------------------
### CLASS TO DEMONSTRATE FEL TEMPORAL COHERENCE/INCOHERECE FOR UNIFORMLY DISTRIBUTED ELECTRON PULSE	
#---------------------------------------------------------------------------------------------------	
	
class FEL_Uniform:
	
	def __init__(self,sigma,omega,num_data_points=500):
		self.sigma = sigma
		self.omega = omega
		self.num_data_points = num_data_points
	
	def coherence_time(self,autocorrelation_data):
		# NOTE: Autocorrelation argument should be already normalised (and from Wigner function if 
		#       not time invariant, i.e., not a WSS)
		tau_vals, ac_vals = autocorrelation_data['x_vals'], autocorrelation_data['y_vals']
		ac_squared = np.abs(np.array(ac_vals)) ** 2
		coherence_time = self.integral(tau_vals, ac_squared)
		print(f'Coherence time = {round(coherence_time,3)}')
		return coherence_time
			
	def fourier_transform(self,signal,n=None,return_power=False,ref_freq=None,normalise=False,scatter=True):
		if n == None:
			n = self.num_data_points
		else:
			n = len(signal['x_vals'])
		t_vals,y = signal['x_vals'],signal['y_vals']
		dt = abs(t_vals[1] - t_vals[0])
		dw = 2.0*np.pi/(t_vals[-1] - t_vals[0])
		w_vals = np.arange(-n/2,n/2)*dw
		fourier_transform = pl.fftshift(pl.fft(y)) * (dt)
		magnitude_spec = abs(fourier_transform)
		phase_spec = np.arctan(fourier_transform.imag/fourier_transform.real)
		if normalise == True:
			magnitude_spec = magnitude_spec/sqrt(2.0*pi)
		if ref_freq != None:
			w_vals = w_vals - ref_freq
		if return_power == True:
			magnitude_spec = (magnitude_spec ** 2)
		return {'x_vals':w_vals,'y_vals':magnitude_spec,'phase':phase_spec,
				'label':'Discrete Fourier transform','scatter':scatter}
		
	def integral(self,x_vals,y_vals,dx=None): #  may add lower and upper limits (only if necessary)
		riemann_sum = 0
		if dx == None:
			dx = abs(x_vals[1] - x_vals[0])
		y_vals = np.array(y_vals) * dx
		riemann_sum = np.sum(y_vals)
		return riemann_sum
	
	def get_num_l_modes(self,bunch_length,coherence_time=None,sigma=None):
		if sigma==None:
			sigma = self.sigma
		M_L = int(round(bunch_length/(2*np.sqrt(np.pi)*sigma)))
		if coherence_time:
			M_L = int(round(bunch_length/coherence_time))
		return M_L
	
	def PSD(self,autocorrelation): # Power Spectral Density		
		S = self.fourier_transform(autocorrelation,1)
		S['label'] = 'Power Spectral Density'
		return S
		
	def coherent_rad(self,t,sigma,omega):
		return np.exp(-((t**2)/(4*(sigma**2))) + (1j*omega*t))
		
	def coherent_rad_freq(self,omega,sigma):
		sigma_w = 1/(2*sigma)
		return ((np.sqrt(np.pi))/sigma_w) * np.exp(-((omega-self.omega)**2)/(4*(sigma_w**2)))
		
	def model_coherent_rad_once(self,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		n = self.num_data_points
		time_vals = np.linspace(-20,20,n)	#Set upper and lower values to (-100,100) for good DFT sample
		e_vals = []
		for t in time_vals:
			e_vals.append(self.coherent_rad(t,sigma,omega))
		return {'x_vals':time_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9 = {}'.format(sigma,omega)}
	
	def model_coherent_rad_once_freq(self,sigma=None,return_power=False):
		if sigma == None:
			sigma = self.sigma
		n = self.num_data_points
		time_vals = np.linspace(-20,20,n)
		dw = 2.0*np.pi/abs(time_vals[-1] - time_vals[0])
		w_vals = np.arange(-n/2,n/2)*dw
		y_vals = []
		for w in w_vals:
			y_vals.append(abs(self.coherent_rad_freq(w,sigma)))
		if return_power == True:
			y_vals = np.array(y_vals) ** 2
		return {'x_vals':w_vals,'y_vals':y_vals,
				'label':'Analytical frequency profile','scatter':False}
			
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

	def incoherent_rad_omega(self,tj,sigma,omega,bunch_length,omega1=None):
		if omega1 == None:
			omega1 = self.omega
		sigma_FT = 1/(2*sigma)
		return (np.exp(-((omega-omega1)**2)/(4*(sigma_FT**2))) * np.cos(omega*tj)) * (np.sqrt(np.pi)/sigma_FT) 
	
	def model_incoherent_rad_once_omega(self,bunch_length,sigma=None,omega1=None):
		if sigma == None:
			sigma = self.sigma
		if omega1 == None:	# Resonance frequency
			omega1 = self.omega
		omega_vals = np.linspace(omega1-10,omega1+10,self.num_data_points)
		e_vals = []
		tj = random.uniform(-(bunch_length*0.5),(bunch_length*0.5))
		for omega in omega_vals:
			e_vals.append(self.incoherent_rad_omega(tj,sigma,omega,bunch_length,omega1))
		delta_omega_vals = omega_vals - omega1
		return {'x_vals':delta_omega_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9$_1 = {}'.format(sigma,omega1)}
		
	def autocorrelation(self,t,num_pulses,bunch_length,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		incoherent_vals_list = [
			self.model_incoherent_rad_once(bunch_length,sigma=None,omega=None) for i in range(num_pulses)
			]
		superposition_pulses = self.get_superposition(incoherent_vals_list)
		time_vals = superposition_pulses['x_vals']
		intensity_vals = self.get_intensity(superposition_pulses)
		E = superposition_pulses['y_vals'][bisect_left(time_vals, t)]
		E_conj = np.conjugate(E)
		intensity_ref = abs(E * E_conj)
		
		c1_vals = []		# 1st order AC function
		c2_vals = []		# 2nd order AC function
		intensity_vals = [] # Intensity values for denominator of final AC result
		tau_vals = []
		for i,time_val in enumerate(time_vals):
			tau_vals.append(time_val - t)
			E_tau = superposition_pulses['y_vals'][i]
			E_tau_conj = np.conjugate(E_tau)
			c1_vals.append(E * E_tau_conj)
			c2_vals.append(abs(E * E_conj) * abs(E_tau * E_tau_conj))
			intensity_vals.append(abs(E_tau * E_tau_conj))
			
		c1_approx = np.exp(-((np.array(tau_vals))**2)/(8*(self.sigma**2)))
		c1_dict = {'x_vals':tau_vals,'y_vals':c1_vals,'label':'1st-order correlation'}
		c2_dict = {'x_vals':tau_vals,'y_vals':c2_vals,'label':'2nd-order correlation'}
		c1_approx_dict = {'x_vals':tau_vals,'y_vals':c1_approx,'label':'Approximation'}
		intensity_dict = {'x_vals':tau_vals,'y_vals':intensity_vals,'label':'Intensity'}
		return {'c1_dict':c1_dict,'c2_dict':c2_dict,'c1_approx_dict':c1_approx_dict,
							'intensity_dict':intensity_dict,'intensity_ref':intensity_ref}
		
	def autocorrelation_integral_method(self,e_field,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		time_vals = e_field['x_vals']
		
		dt = abs(time_vals[1] - time_vals[0])
		tau_indices = np.arange(-len(time_vals),len(time_vals))
		bottom_vals = []
		top_vals = []
		c1_vals = []		# 1st order AC function
		tau_vals = tau_indices * dt
		
		for tau_index in tau_indices:
			zeros = list(np.zeros(len(time_vals)))
			E = np.array(zeros + list(e_field['y_vals']) + zeros)
			E_shifted = np.roll(E, tau_index)
			E_product = E * np.conjugate(E_shifted)
			top_integral = self.integral(time_vals,E_product)
			
			intensity = abs(E * np.conjugate(E))
			bottom_integral = self.integral(time_vals,intensity)
			
			bottom_vals.append(bottom_integral)
			top_vals.append(top_integral)
			c1_vals.append(top_integral / bottom_integral)
			c1_approx = np.exp(-((np.array(tau_vals))**2)/(8*(self.sigma**2)))

		c1_dict = {'x_vals':tau_vals,'y_vals':c1_vals,'label':'1st-order normalised correlation'}
		c1_approx_dict = {'x_vals':tau_vals,'y_vals':c1_approx,'label':'Approximation'}
		return {'c1_dict':c1_dict,'c1_approx_dict':c1_approx_dict,
				'top_vals':top_vals,'bottom_vals':bottom_vals} # <-- for ensemble averages

class FEL_Uniform_Complex(FEL_Uniform):
	
	def incoherent_rad(self,t,tj,sigma,omega,bunch_length):
		# Models (1.19) in notes from Lampros A.A Nikolopoulos
		X = 1 + 1j/np.sqrt(3)
		return np.exp( (-X*(t-tj)**2)/(4*(sigma**2)) + (1j*omega*tj) )
	
	def incoherent_rad_omega(self,tj,sigma,omega,bunch_length,omega1=None):
		if omega1 == None:
			omega1 = self.omega
		X = 1 + 1j/np.sqrt(3)
		first_term = np.sqrt((np.pi*4*self.sigma)/X)
		second_term = np.exp(-(self.sigma**2)*(omega**2)/X)
		third_term = np.exp(1j*(omega1+omega)*tj)
		return first_term * second_term * third_term
