#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import FEL_gaussian as FEL


class Test_EV(FEL.Temporal):
	'''Redefining the function for E-field to be that of the simple test as outlined in work
	'''
	def incoherent_rad(self,t,tj,sigma,omega,bunch_length):
		return np.exp((-(t-tj)**2))


test = Test_EV(sigma=2,omega=6)


def SHOW_INCOHERENT_RADIATION(num_pulses,bunch_length=100):
	incoherent_vals_list = [test.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]
	superposition_pulses = test.get_superposition(incoherent_vals_list)
	intensity = test.get_intensity(superposition_pulses)
	return {'superposition_pulses':superposition_pulses,'intensity':intensity}

def MONTE_CARLO_TEST_TIME(num_trials,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		incoherent_radiation = SHOW_INCOHERENT_RADIATION(num_pulses,bunch_length)
		e_field = incoherent_radiation['superposition_pulses']
		intensity = incoherent_radiation['intensity']
		
		if i == 0:
			e_field_vals = np.zeros(shape=len(e_field['y_vals']))
			intensity_vals = np.zeros(shape=len(intensity['y_vals']))
			time_vals = e_field['x_vals']
			
		e_field_vals += (e_field['y_vals'])
		intensity_vals += (intensity['y_vals'])
		
	average_e_field_vals = e_field_vals / num_trials
	average_intensity_vals = intensity_vals / num_trials
	
	plot_dict_e = {
				'x_vals':time_vals,
				'y_vals':average_e_field_vals,
				'label':None
				}
	plot_dict_i = {
				'x_vals':time_vals,
				'y_vals':average_intensity_vals,
				'label':None
				}
				
	with open(f'test_e_field_N={num_pulses}_T={bunch_length}_trials={num_trials}pickle', 'wb') as f1:
			pickle.dump(plot_dict_e,f1)
	with open(f'test_intensity_N={num_pulses}_T={bunch_length}_trials={num_trials}.pickle', 'wb') as f2:
			pickle.dump(plot_dict_i,f2)
	
	# TO GET THE EXPECTED VALUES:
	valid_time_indices = np.where(abs(time_vals) < bunch_length/2)
	e_vals = [average_e_field_vals[index] for index in valid_time_indices]
	i_vals = [average_intensity_vals[index] for index in valid_time_indices]
	EV_e_field = round(np.mean(e_vals),2)
	EV_intensity = round(np.mean(i_vals),2)
	print(f'E-FIELD EXPECTED VALUE: {EV_e_field}')
	print(f'INTENSITY EXPECTED VALUE: {EV_intensity}')
	
	FEL.plot_graph([plot_dict_e],'Monte Carlo Test for EV. Shown: Average E-Field in Time Domain',
					f'Average E-field vs Time, N = {num_pulses}, T = {bunch_length}, Number of trials = {num_trials}',
					'time ','simulated average E-field',
					f'test_e_field_N={num_pulses}_T={bunch_length}_trials={num_trials}',
					show=show_result,scatter=False)
	FEL.plot_graph([plot_dict_i],'Monte Carlo Test for EV. Shown: Average Intensity in Time Domain',
					f'Average Intensity vs Time, N = {num_pulses}, T = {bunch_length}, Number of trials = {num_trials}',
					'time ','simulated average intensity',
					f'test_intensity_N={num_pulses}_T={bunch_length}_trials={num_trials}',
					show=show_result,scatter=False)
	
	
				
MONTE_CARLO_TEST_TIME(num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
MONTE_CARLO_TEST_TIME(num_trials=500,num_pulses=150,bunch_length=100,show_result=True)
MONTE_CARLO_TEST_TIME(num_trials=500,num_pulses=200,bunch_length=70,show_result=True)
MONTE_CARLO_TEST_TIME(num_trials=500,num_pulses=60,bunch_length=130,show_result=True)
