#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import pickle
from bisect import bisect_left

from FEL_uniform import FEL_Uniform

	
#---------------------------------------------------------------------------------------------------
### CLASS TO DEMONSTRATE FEL TEMPORAL COHERENCE/INCOHERECE FOR NORMALLY DISTRIBUTED ELECTRON PULSE	
#---------------------------------------------------------------------------------------------------	
	
class FEL_Gaussian(FEL_Uniform):
	
	def model_incoherent_rad_once(self,bunch_length,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		time_vals = np.linspace(-(bunch_length*0.5)-(4*sigma),(bunch_length*0.5)+(4*sigma),self.num_data_points)
		e_vals = []
		tj = np.random.normal(0,(bunch_length*0.5)/(np.pi*2))
		for t in time_vals:
			e_vals.append(self.incoherent_rad(t,tj,sigma,omega,bunch_length))
		return {'x_vals':time_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9 = {}'.format(sigma,omega)}
	
	def model_incoherent_rad_once_omega(self,bunch_length,sigma=None,omega1=None):
		if sigma == None:
			sigma = self.sigma
		if omega1 == None:
			omega1 = self.omega
		omega_vals = np.linspace(omega1-10,omega1+10,self.num_data_points)
		e_vals = []
		tj = np.random.normal(0,(bunch_length*0.5)/(np.pi*2))
		for omega in omega_vals:
			e_vals.append(self.incoherent_rad_omega(tj,sigma,omega,bunch_length,omega1))
		delta_omega_vals = omega_vals - omega1
		return {'x_vals':delta_omega_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9$_1 = {}'.format(sigma,omega1)}


class FEL_Gaussian_Complex(FEL_Gaussian):
	
	def incoherent_rad(self,t,tj,sigma,omega,bunch_length):
		# Models (1.19) in notes from Lampros A.A Nikolopoulos
		X = 1 + 1j/np.sqrt(3)
		return np.exp( (-X*((t-tj)**2))/(4*(sigma**2)) + (1j*omega*(t-tj) ))
	
	def incoherent_rad_omega(self,tj,sigma,omega,bunch_length,omega1=None):
		if omega1 == None:
			omega1 = self.omega
		X = 1 + 1j/np.sqrt(3)
		first_term = np.sqrt((np.pi*4*self.sigma)/X)
		second_term = np.exp(-((self.sigma**2)*(omega**2)/X))
		third_term = np.exp(1j*(omega1+omega)*tj)
		return first_term * second_term * third_term
