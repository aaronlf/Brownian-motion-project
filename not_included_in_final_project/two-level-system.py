#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import FEL_gaussian as FEL

temp = FEL.Temporal(sigma=2,omega=6)

class TwoLevelSystem:
	
	def __init__(self,rabi_param,e1=-13.6,e2=-3.40,delta=0,omega=None):
		# all physical quantities are presented here in SI units
		self.a0 = 5.29*1e-11 		# first Bohr radius
		self.hbar = 1.054*1e-34 
		self.e1 = e1 * 1.602*1e-19 	# 1s orbital energy
		self.e2 = e2 * 1.602*1e-19 	# 2p orbital energy
		self.omega = omega 			# omega is picked arbitrary, but close to (e2-e1)/hbar
		if omega != None:	# allows user to manually set delta based on laser frequency omega
			self.delta = (self.e2/self.hbar) - (self.e1/self.hbar) - self.omega
		else: 
			self.delta = delta
		self.rabi_param = rabi_param
		self.det_rabi_param = np.sqrt( (self.delta**2) + (4*(abs(rabi_param)**2)) )
		
	def u1(self,t):
		e_term = np.exp(-1j * self.delta * t / 2) / self.det_rabi_param
		cos_term = self.det_rabi_param * np.cos(self.det_rabi_param * t / 2)
		sin_term = 1j * self.delta * np.sin(self.det_rabi_param * t / 2)
		return e_term * (cos_term + sin_term)
		
	def u1_squared(self,t):
		return 1 - self.u2_squared(t)
		
	def u2(self,t):
		rabi_term = -(2j * np.conjugate(self.rabi_param)) / self.det_rabi_param
		e_term = np.exp(-1j * self.delta * t / 2)
		sin_term = np.sin(self.det_rabi_param * t / 2)
		return rabi_term * e_term * sin_term
		
	def u2_squared(self,t):
		first_term = 4 * (abs(self.rabi_param)**2) / (abs(self.det_rabi_param)**2)
		second_term = (np.sin(self.det_rabi_param * t / 2))**2
		return first_term * second_term
		
	def wavefunc_1s(self):
		pass
	
	def wavefunc_2p(self):
		pass


def SIMPLE_AMPLITUDES():
	time_vals = np.linspace(0,10,500)
	u2_sq_vals = np.array([])
	for t in time_vals:
		u2_sq_vals = np.append(u2_sq_vals,tls.u2_squared(t))
	u1_sq_vals = 1 - u2_sq_vals
	u1_data = {
				'x_vals':time_vals,
				'y_vals':u1_sq_vals,
				'label':'$|u_1(t)|^2$'
				}
	u2_data = {
				'x_vals':time_vals,
				'y_vals':u2_sq_vals,
				'label':'$|u_2(t)|^2$'
				}
	FEL.plot_graph(
					data_dicts=[u1_data,u2_data],
					fig_name='Populations of states vs time',
					title='TDSE amplitude coefficients vs time, \u03B4 = {}'.format(tls.delta),
					xlabel='Time',
					ylabel='Probability of electron found in state',
					savefig_name=None
				)
					
		
if __name__ == '__main__':
	tls = TwoLevelSystem(rabi_param=1,delta=1)
	SIMPLE_AMPLITUDES()
	
	
