#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pickle
from plotting import Plotting
from FEL_uniform import FEL_Uniform,FEL_Uniform_Complex
from FEL_gaussian import FEL_Gaussian,FEL_Gaussian_Complex

uniform_pulse_FEL = FEL_Uniform(sigma=2,omega=6)
gaussian_pulse_FEL = FEL_Gaussian(sigma=2,omega=6)
uniform_pulse_complex_FEL = FEL_Uniform_Complex(sigma=2,omega=6)
gaussian_pulse_complex_FEL = FEL_Gaussian_Complex(sigma=2,omega=6)


#---------------------------------------------------------------------------------------------------
### 									MAIN FUNCTION 											 ###
#---------------------------------------------------------------------------------------------------

def main():
	FEL = uniform_pulse_FEL
	#show_coherent_radiation(FEL)
	#show_incoherent_radiation(FEL,num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#incoherent_average_intensity_vs_num_sources(FEL)
	#monte_carlo_intensity_time(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#monte_carlo_intensity_omega(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_time(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_autocorrelation_time(FEL,num_trials=10,t=0,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_freq(FEL,delta_omega=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_autocorrelation_freq(FEL,num_trials=500,delta_omega=0,num_pulses=100,bunch_length=100,show_result=True)
	
	FEL = gaussian_pulse_FEL
	#show_coherent_radiation(FEL)
	#show_incoherent_radiation(FEL,num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#incoherent_average_intensity_vs_num_sources(FEL)
	#monte_carlo_intensity_time(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#monte_carlo_intensity_omega(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_time(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_autocorrelation_time(FEL,num_trials=10,t=0,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_freq(FEL,delta_omega=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_autocorrelation_freq(FEL,num_trials=500,delta_omega=0,num_pulses=100,bunch_length=100,show_result=True)
	
	FEL = uniform_pulse_complex_FEL
	#show_coherent_radiation(FEL)
	#show_incoherent_radiation(FEL,num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#incoherent_average_intensity_vs_num_sources(FEL)
	#monte_carlo_intensity_time(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#monte_carlo_intensity_omega(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_time(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	monte_carlo_autocorrelation_time(FEL,num_trials=500,t=0,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_freq(FEL,delta_omega=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_autocorrelation_freq(FEL,num_trials=500,delta_omega=0,num_pulses=100,bunch_length=100,show_result=True)
	
	FEL = gaussian_pulse_complex_FEL
	#show_coherent_radiation(FEL)
	#show_incoherent_radiation(FEL,num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#incoherent_average_intensity_vs_num_sources(FEL)
	#monte_carlo_intensity_time(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#monte_carlo_intensity_omega(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_time(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_autocorrelation_time(FEL,num_trials=500,t=0,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation_freq(FEL,delta_omega=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_autocorrelation_freq(FEL,num_trials=500,delta_omega=0,num_pulses=100,bunch_length=100,show_result=True)
	


#---------------------------------------------------------------------------------------------------
### 								SUBROUTINES OF MAIN											 ###
#---------------------------------------------------------------------------------------------------

def show_coherent_radiation(FEL):
	coherent_vals0 = FEL.model_coherent_rad_once(sigma=2)
	coherent_vals1 = FEL.model_coherent_rad_once(sigma=1)
	Plotting.plot_graph([coherent_vals0,coherent_vals1],'Coherent Radiation','Coherent Radiation','t',
				'E/e$_0$','coherent_radiation')
	
	
def show_incoherent_radiation(FEL,num_pulses,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True,print_mode_num=True):
	incoherent_vals_list = [FEL.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]
	if subplots == True:
		Plotting.plot_subplots(incoherent_vals_list[:10],'Incoherent Radiation','Incoherent Radiation',
				'Shown: 10 randomly spaced pulses','time','incoherent_radiation_individual_pulses')			
	superposition_pulses = FEL.get_superposition(incoherent_vals_list)
	if plot_superposition == True:
		Plotting.plot_graph([superposition_pulses],'Incoherent Radiation',
					'Incoherent Radiation: Superposition of {} pulses'.format(num_pulses),
					'time','E/e$_0$','incoherent_radiation_superposition_{}_pulses'.format(num_pulses))
	intensity = FEL.get_intensity(superposition_pulses)
	if plot_intensity == True:
		Plotting.plot_graph([intensity],'Incoherent Radiation Intensity Profile',
					'Incoherent Radiation: Intensity profile for {} pulses'.format(num_pulses),
					'time','intensity','intensity_incoherent_radiation_{}_pulses'.format(num_pulses))
	if print_mode_num == True:
		print('Approximate number of longitudinal modes = {}'.format(FEL.get_num_l_modes(bunch_length)))
	return {'superposition_pulses':superposition_pulses,'intensity':intensity}


def incoherent_average_intensity_vs_num_sources(FEL,bunch_length=100):
	num_pulse_vals = np.arange(10,400,10)
	average_intensities = []
	for num_pulses in num_pulse_vals:
		print(num_pulses)
		incoherent_vals_list = [FEL.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]
		superposition_pulses = FEL.get_superposition(incoherent_vals_list)
		intensity_profile = FEL.get_intensity(superposition_pulses)['y_vals']
		average_intensity = np.mean(intensity_profile)
		average_intensities.append(average_intensity)
	plot_dict = {
				'x_vals':num_pulse_vals,
				'y_vals':average_intensities,
				'label':None
				}
	Plotting.plot_graph([plot_dict],'Incoherent Radiation Average Intensity',
					'Incoherent Radiation: Average intensity vs number of sources',
					'number of sources','average intensity',
					'average_intensity_vs_num_pulses',scatter=True)
	

def monte_carlo_intensity_time(FEL,num_trials,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		intensity = show_incoherent_radiation(FEL,num_pulses,bunch_length,False,False,False,False)['intensity']
		if i == 0:
			intensity_vals = np.zeros(shape=len(intensity['y_vals']),dtype=complex)
			time_vals = intensity['x_vals']
		intensity_vals += (intensity['y_vals'])
	average_intensity_vals = intensity_vals / num_trials
	plot_dict = {
				'x_vals':time_vals,
				'y_vals':average_intensity_vals,
				'label':None
				}
	with open('monte_carlo_time_average_intensity_{}_trials.pickle'.format(num_trials), 'wb') as f:
			pickle.dump(plot_dict,f)
	Plotting.plot_graph([plot_dict],'Monte Carlo Trialed Average Intensity Profile in Time Domain',
					'Average Intensity vs Time',
					'time ','simulated average intensity',
					'monte_carlo_time_average_intensity_{}_trials'.format(num_trials),
					show=show_result,scatter=False)
					
	
def show_incoherent_radiation_omega(FEL,num_pulses,bunch_length=100,plot_superposition=True,plot_intensity=True):
	incoherent_vals_list_omega = [FEL.model_incoherent_rad_once_omega(bunch_length) for i in range(num_pulses)]
	superposition_pulses_omega = FEL.get_superposition(incoherent_vals_list_omega)
	if plot_superposition == True:
		Plotting.plot_graph([superposition_pulses_omega],'Incoherent Radiation',
					'Incoherent Radiation: Superposition of {} pulses in terms of \u03C9'.format(num_pulses),
					'\u03C9 - \u03C9$_1$','E/e$_0$','omega_incoherent_radiation_superposition_{}_pulses'.format(num_pulses))
	#'E\u03C3/e$_0\sqrt{\u03C0}$'
	intensity_omega = FEL.get_intensity(superposition_pulses_omega)
	if plot_intensity == True:
		Plotting.plot_graph([intensity_omega],'Incoherent Radiation Intensity Spectrum',
					'Incoherent Radiation: Intensity spectrum for {} pulses in terms of \u03C9'.format(num_pulses),
					'\u03C9 - \u03C9$_1$','P(\u03C9)','omega_intensity_incoherent_radiation_{}_pulses'.format(num_pulses))
	FEL.get_num_l_modes(bunch_length)
	return {'superposition_pulses':superposition_pulses_omega,'intensity':intensity_omega}
					
					
def monte_carlo_intensity_omega(FEL,num_trials,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		intensity = show_incoherent_radiation_omega(FEL,num_pulses,bunch_length,False,False)['intensity']
		if i == 0:
			intensity_vals = np.zeros(shape=len(intensity['y_vals']),dtype=complex)
			time_vals = intensity['x_vals']
		intensity_vals += (intensity['y_vals'])
	average_intensity_vals = intensity_vals / num_trials
	plot_dict = {
				'x_vals':time_vals,
				'y_vals':average_intensity_vals,
				'label':None
				}
	with open('monte_carlo_frequency_average_intensity_{}_trials.pickle'.format(num_trials), 'wb') as f:
			pickle.dump(plot_dict,f)
	Plotting.plot_graph([plot_dict],'Monte Carlo Trialed Average Intensity Profile in Frequency Domain',
					'Average Intensity Spectrum',
					'\u03C9 - \u03C9$_1$','simulated average intensity',
					'monte_carlo_frequency_average_intensity_{}_trials'.format(num_trials),
					show=show_result,scatter=False)	
		

def show_autocorrelation_time(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True):
	autocorrelation_dicts = FEL.autocorrelation_time(t,num_pulses,bunch_length)
	c1_data = autocorrelation_dicts['c1_dict']
	c2_data = autocorrelation_dicts['c2_dict']
	if plot_e == True:
		Plotting.plot_graph([c1_data],'Autocorrelation function, E-field',
					'Autocorrelation Function: E-Field in Time Domain',
					'\u03C4','$C_1$(\u03C4)',f'E_autocorrelation_{num_pulses}_pulses_t={t}')
	if plot_intensity == True:
		Plotting.plot_graph([c2_data],'Autocorrelation function, Intensity',
					'Autocorrelation Function: Intensity in Time Domain',
					'\u03C4','$C_2$(\u03C4)',f'I_autocorrelation_{num_pulses}_pulses_t={t}')
	

def monte_carlo_autocorrelation_time(FEL,num_trials,t=0,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		autocorrelation_dicts = FEL.autocorrelation_time(t,num_pulses,bunch_length)
		c1_data = autocorrelation_dicts['c1_dict']
		c2_data = autocorrelation_dicts['c2_dict']
		if i == 0:
			c1_vals = np.zeros(shape=len(c1_data['y_vals']),dtype=complex)
			c2_vals = np.zeros(shape=len(c2_data['y_vals']),dtype=complex)
			tau_vals = c1_data['x_vals']
		c1_vals += (c1_data['y_vals'])
		c2_vals += (c2_data['y_vals'])
	average_c1_vals = c1_vals / num_trials
	average_c2_vals = c2_vals / num_trials
	c1_plot_dict = {
				'x_vals':tau_vals,
				'y_vals':average_c1_vals,
				'label':None
				}
	c2_plot_dict = {
				'x_vals':tau_vals,
				'y_vals':average_c2_vals,
				'label':None
				}
	with open(f'autocorrelation_c1_monte_carlo_{num_trials}_trials.pickle', 'wb') as f:
			pickle.dump(c1_plot_dict,f)
	with open(f'autocorrelation_c2_monte_carlo_{num_trials}_trials.pickle', 'wb') as f:
			pickle.dump(c2_plot_dict,f)		
	Plotting.plot_graph([c1_plot_dict],'Monte Carlo Trialed $C_1$ in Time Domain',
					'First-Order Autocorrelation vs Time',
					'\u03C4','simulated average $C_1$(\u03C4)',
					f'C1_monte_carlo_time_{num_trials}_trials',
					show=show_result,scatter=False)
	Plotting.plot_graph([c2_plot_dict],'Monte Carlo Trialed $C_2$ in Time Domain',
					'First-Order Autocorrelation vs Time',
					'\u03C4','simulated average $C_2$(\u03C4)',
					f'C2_monte_carlo_time_{num_trials}_trials',
					show=show_result,scatter=False)
					

def show_autocorrelation_freq(FEL,delta_omega=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True):
	autocorrelation_dicts = FEL.autocorrelation_freq(delta_omega,num_pulses,bunch_length)
	c1_data = autocorrelation_dicts['c1_dict']
	c2_data = autocorrelation_dicts['c2_dict']
	if plot_e == True:
		Plotting.plot_graph([c1_data],'Autocorrelation function, E-field',
					'Autocorrelation Function: E-Field in Frequency Domain',
					'Variation in \u03C9','$C_1$',f'E_autocorrelation_{num_pulses}_pulses_delta_omega={delta_omega}')
	if plot_intensity == True:
		Plotting.plot_graph([c2_data],'Autocorrelation function, Intensity',
					'Autocorrelation Function: Intensity in Frequency Domain',
					'Variation in \u03C9','$C_2$',f'I_autocorrelation_{num_pulses}_pulses_delta_omega={delta_omega}')	


def monte_carlo_autocorrelation_freq(FEL,num_trials,delta_omega=0,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		autocorrelation_dicts = FEL.autocorrelation_freq(delta_omega,num_pulses,bunch_length)
		c1_data = autocorrelation_dicts['c1_dict']
		c2_data = autocorrelation_dicts['c2_dict']
		if i == 0:
			c1_vals = np.zeros(shape=len(c1_data['y_vals']),dtype=complex)
			c2_vals = np.zeros(shape=len(c2_data['y_vals']),dtype=complex)
			d_omega_vals = c1_data['x_vals']
		c1_vals += (c1_data['y_vals'])
		c2_vals += (c2_data['y_vals'])	
	average_c1_vals = c1_vals / num_trials
	average_c2_vals = c2_vals / num_trials
	c1_plot_dict = {
				'x_vals':d_omega_vals,
				'y_vals':average_c1_vals,
				'label':None
				}
	c2_plot_dict = {
				'x_vals':d_omega_vals,
				'y_vals':average_c2_vals,
				'label':None
				}
	with open(f'freq_autocorrelation_c1_monte_carlo_{num_trials}_trials.pickle', 'wb') as f:
			pickle.dump(c1_plot_dict,f)
	with open(f'freq_autocorrelation_c2_monte_carlo_{num_trials}_trials.pickle', 'wb') as f:
			pickle.dump(c2_plot_dict,f)		
	Plotting.plot_graph([c1_plot_dict],'Monte Carlo Trialed $C_1$ in Frequency Domain',
					'First-Order Autocorrelation vs Change in Frequency',
					'Variation in \u03C9','simulated average $C_1$',
					f'C1_monte_carlo_freq_{num_trials}_trials',
					show=show_result,scatter=False)
	Plotting.plot_graph([c2_plot_dict],'Monte Carlo Trialed $C_2$ in Frequency Domain',
					'First-Order Autocorrelation vs Change in Frequency',
					'Variation in \u03C9','simulated average $C_2$',
					f'C2_monte_carlo_freq_{num_trials}_trials',
					show=show_result,scatter=False)					
			
								
#---------------------------------------------------------------------------------------------------


if __name__ == '__main__':
	main()

