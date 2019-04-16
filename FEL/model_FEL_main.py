#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pickle
from plotting import Plotting
from FEL_uniform import FEL_Uniform,FEL_Uniform_Complex
from FEL_gaussian import FEL_Gaussian,FEL_Gaussian_Complex

uniform_pulse_FEL = FEL_Uniform(sigma=3,omega=6)
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
	#monte_carlo_intensity_time(FEL,num_trials=100,num_pulses=100,bunch_length=100,show_result=True)
	#show_incoherent_radiation_FT(FEL,num_pulses=100,bunch_length=100,plot_FT=True,plot_intensity=True)
	#monte_carlo_intensity_FT(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#	(unused)	show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#	(unused)	monte_carlo_intensity_omega(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_normalised_autocorrelation(FEL,num_trials=500,t=0,num_pulses=100,bunch_length=100,show_result=True)

	
	#FEL = gaussian_pulse_FEL
	#show_incoherent_radiation(FEL,num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#incoherent_average_intensity_vs_num_sources(FEL)
	#monte_carlo_intensity_time(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#	(unused)	show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#	(unused)	monte_carlo_intensity_omega(FEL,num_trials=50,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_normalised_autocorrelation(FEL,num_trials=10,t=0,num_pulses=100,bunch_length=100,show_result=True)

	
	#FEL = uniform_pulse_complex_FEL
	#show_incoherent_radiation(FEL,num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#incoherent_average_intensity_vs_num_sources(FEL)
	#monte_carlo_intensity_time(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#	(unused)	show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#	(unused)	monte_carlo_intensity_omega(FEL,num_trials=20,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_normalised_autocorrelation(FEL,num_trials=500,t=0,num_pulses=100,bunch_length=100,show_result=True)

	
	FEL = gaussian_pulse_complex_FEL
	#show_incoherent_radiation(FEL,num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#incoherent_average_intensity_vs_num_sources(FEL)
	#monte_carlo_intensity_time(FEL,num_trials=1,num_pulses=100,bunch_length=100,show_result=True)
	#show_incoherent_radiation_FT(FEL,num_pulses=100,bunch_length=100,plot_FT=True,plot_intensity=True)
	#monte_carlo_intensity_FT(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#	(unused)	show_incoherent_radiation_omega(FEL,100,plot_superposition=True,plot_intensity=True)
	#	(unused)	monte_carlo_intensity_omega(FEL,num_trials=100,num_pulses=100,bunch_length=100,show_result=True)
	#show_autocorrelation(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#monte_carlo_normalised_autocorrelation(FEL,num_trials=500,t=10,num_pulses=100,bunch_length=100,show_result=True)
	#monte_carlo_integral_autocorrelation(FEL,num_trials=500,num_pulses=100,bunch_length=100,show_result=True)

	


#---------------------------------------------------------------------------------------------------
### 								SUBROUTINES OF MAIN											 ###
#---------------------------------------------------------------------------------------------------

def show_coherent_radiation(FEL):
	coherent_vals0 = FEL.model_coherent_rad_once()
	coherent_vals1 = FEL.model_coherent_rad_once(sigma=1)
	Plotting.plot_graph([coherent_vals0,coherent_vals1],'Coherent Radiation','Coherent Radiation','t',
				'E/e$_0$','coherent_radiation')
	coherent_vals0_freq = FEL.model_coherent_rad_once_freq(sigma=2,return_power=True)
	FT0 = FEL.fourier_transform(coherent_vals0,return_power=True,scatter=True)
	Plotting.scatter_and_plot([FT0,coherent_vals0_freq],'Coherent pulse in frequency space','Coherent pulse in frequency space: \u03C9$_r$ = 3','\u03C9 ',
				'|$E_\u03C9/e_0$|$^2$') # FT was done with scatter=True in results section of report
	coherent_pulse_autocorrelation = FEL.autocorrelation_integral_method(coherent_vals0)
	c1,c1_approx = coherent_pulse_autocorrelation['c1_dict'],coherent_pulse_autocorrelation['c1_approx_dict']
	coherence_time = round(FEL.coherence_time(c1),3)
	Plotting.plot_graph([c1,c1_approx],'Coherent pulse autocorrelation',f'Coherent pulse autocorrelation \n\u03C3 = {FEL.sigma}, \u03C9$_r$ = {FEL.omega} \nCoherence time found: $\u03C4_c$ = {coherence_time}',
				'\u03C4','$C_1(\u03C4)$')
	
def show_incoherent_radiation(FEL,num_pulses,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True):
	incoherent_vals_list = [FEL.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]
	if subplots == True:
		Plotting.plot_subplots(incoherent_vals_list[:10],'Incoherent Radiation','Incoherent Radiation',
				'Shown: 10 randomly spaced pulses','time','incoherent_radiation_individual_pulses')			
	superposition_pulses = FEL.get_superposition(incoherent_vals_list)
	if plot_superposition == True:
		Plotting.plot_graph([superposition_pulses],'Incoherent Radiation',
					f'Incoherent Radiation: Superposition of {num_pulses} pulses \n\u03C3 = {FEL.sigma}, \u03C9$_r$ = {FEL.omega}',
					'time','E/e$_0$',f'incoherent_radiation_superposition_{num_pulses}_pulses')
	intensity = FEL.get_intensity(superposition_pulses)
	if plot_intensity == True:
		autocorrelation = FEL.autocorrelation_integral_method(superposition_pulses)['c1_dict']
		coherence_time = round(FEL.coherence_time(autocorrelation),3)
		M_L = FEL.get_num_l_modes(bunch_length,coherence_time) # This is different for Gaussian bunch density pulse!
		print(f'Approximate number of longitudinal modes = {FEL.get_num_l_modes(bunch_length,coherence_time)}')
		Plotting.plot_graph([intensity],'Incoherent Radiation Intensity Profile',
					f'Incoherent Radiation: Intensity profile for {num_pulses} pulses \n\u03C3 = {FEL.sigma}, \u03C9$_r$ = {FEL.omega} \nCoherence time found: $\u03C4_c$ = {coherence_time}, Predicted nuber of modes: M = {M_L}',
					'time','intensity',f'intensity_incoherent_radiation_{num_pulses}_pulses')
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
		incoherent_radiation = show_incoherent_radiation(FEL,num_pulses,bunch_length,False,False,False)
		e_field = incoherent_radiation['superposition_pulses']
		intensity = incoherent_radiation['intensity']
		if i == 0:
			e_field_vals = np.zeros(shape=len(e_field['y_vals']),dtype=complex)
			intensity_vals = np.zeros(shape=len(intensity['y_vals']),dtype=complex)
			time_vals = intensity['x_vals']
		e_field_vals += (e_field['y_vals'])
		intensity_vals += (intensity['y_vals'])
	average_e_field = e_field_vals / num_trials
	average_intensity_vals = intensity_vals / num_trials
	e_field_plot_dict = {
				'x_vals':time_vals,
				'y_vals':average_e_field,
				'label':None
				}
	intensity_plot_dict = {
				'x_vals':time_vals,
				'y_vals':average_intensity_vals,
				'label':None
				}
	with open('monte_carlo_time_average_e_field_{}_trials.pickle'.format(num_trials), 'wb') as f:
			pickle.dump(e_field_plot_dict,f)
	with open('monte_carlo_time_average_intensity_{}_trials.pickle'.format(num_trials), 'wb') as f:
			pickle.dump(intensity_plot_dict,f)
	Plotting.plot_graph([e_field_plot_dict],'Monte Carlo Trialed Average E-Field in Time Domain',
					'Average E-Field vs Time',
					'time ','simulated average E-field',
					'monte_carlo_time_average_e_field_{}_trials'.format(num_trials),
					show=show_result,scatter=False)
	Plotting.plot_graph([intensity_plot_dict],'Monte Carlo Trialed Average Intensity Profile in Time Domain',
					'Average Intensity vs Time',
					'time ','simulated average intensity',
					'monte_carlo_time_average_intensity_{}_trials'.format(num_trials),
					show=show_result,scatter=False)
					

					

def show_incoherent_radiation_FT(FEL,num_pulses,bunch_length=100,plot_FT=True,plot_intensity=True):		
	incoherent_vals_list = [FEL.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]	
	superposition_pulses = FEL.get_superposition(incoherent_vals_list)
	FT = FEL.fourier_transform(superposition_pulses)
	if plot_FT == True:
		Plotting.plot_graph([FT],'Incoherent Radiation in Frequency Space',
					f'Incoherent Radiation: E-field spectrum of {num_pulses} pulses \n\u03C3 = {FEL.sigma}, \u03C9$_r$ = {FEL.omega}',
					'\u03C9' ,'E/e$_0$',f'incoherent_radiation_in_freq_space_{num_pulses}_pulses')
	intensity = FEL.get_intensity(FT)
	if plot_intensity == True:
		autocorrelation = FEL.autocorrelation_integral_method(superposition_pulses)['c1_dict']
		coherence_time = round(FEL.coherence_time(autocorrelation),3)
		M_L = FEL.get_num_l_modes(bunch_length,coherence_time) # This is different for Gaussian bunch density pulse!
		print(f'Approximate number of longitudinal modes = {FEL.get_num_l_modes(bunch_length,coherence_time)}')
		Plotting.plot_graph([intensity],'Incoherent Radiation Intensity Profile',
					f'Incoherent Radiation: Intensity spectrum for {num_pulses} pulses \n\u03C3 = {FEL.sigma}, \u03C9$_r$ = {FEL.omega}',
					'\u03C9 ','intensity',f'intensity_spectrum_incoherent_radiation_{num_pulses}_pulses')
	return {'superposition_pulses':FT,'intensity':intensity}
	

def monte_carlo_intensity_FT(FEL,num_trials,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		incoherent_radiation = show_incoherent_radiation(FEL,num_pulses,bunch_length,False,False,False)
		e_field = incoherent_radiation['superposition_pulses']
		intensity = incoherent_radiation['intensity']
		if i == 0:
			e_field_vals = np.zeros(shape=len(e_field['y_vals']),dtype=complex)
			intensity_vals = np.zeros(shape=len(intensity['y_vals']),dtype=complex)
		e_field_vals += e_field['y_vals']
		intensity_vals += intensity['y_vals']
	average_e_field = e_field_vals / num_trials
	average_intensity_vals = intensity_vals / num_trials
	e_field['y_vals'] = average_e_field
	intensity['y_vals'] = average_intensity_vals
	e_field_FT = FEL.fourier_transform(e_field)
	intensity_FT = e_field_FT.copy()
	intensity_FT['y_vals'] = np.array(e_field_FT['y_vals']) ** 2
	e_field_phase_dict = e_field_FT.copy()
	e_field_phase_dict['y_vals'] = e_field_phase_dict['phase']

	with open('monte_carlo_freq_average_e_field_{}_trials.pickle'.format(num_trials), 'wb') as f:
			pickle.dump(e_field_FT,f)
	with open('monte_carlo_freq_average_intensity_{}_trials.pickle'.format(num_trials), 'wb') as f:
			pickle.dump(intensity_FT,f)
	Plotting.plot_graph([e_field_FT],'Magnitude Monte Carlo Trialed Average E-Field in Frequency Domain',
					'Average E-Field vs frequency (magnitude spectrum)',
					'\u03C9  ','E(\u03C9) ',
					'monte_carlo_freq_average_e_field_{}_trials'.format(num_trials),
					show=show_result,scatter=False)
	Plotting.plot_graph([intensity_FT],'Magnitude Monte Carlo Trialed Average Intensity Profile in Frequency Domain',
					'Average intensity vs frequency (magnitude spectrum)',
					'\u03C9 ','I(\u03C9) ',
					'monte_carlo_freq_average_intensity_{}_trials'.format(num_trials),
					show=show_result,scatter=False)
	Plotting.plot_graph([e_field_phase_dict],'Phase Monte Carlo Trialed Average E-Field in Frequency Domain',
					'Average E-Field vs frequency (phase spectrum)',
					'\u03C9  ','\u03C6 ',
					'monte_carlo_freq_average_e_field_{}_trials'.format(num_trials),
					show=show_result,scatter=False)
								

		

def show_autocorrelation(FEL,t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True):
	autocorrelation_dicts = FEL.autocorrelation(t,num_pulses,bunch_length)
	c1_data = autocorrelation_dicts['c1_dict']
	c2_data = autocorrelation_dicts['c2_dict']
	intensity_data = autocorrelation_dicts['intensity_dict']
	intensity_ref = autocorrelation_dicts['intensity_ref']
	final_c1_vals = np.array(c1_data['y_vals']) / np.sqrt(np.array(intensity_data['y_vals']) * intensity_ref)
	final_c2_vals = np.array(c2_data['y_vals']) / (np.array(intensity_data['y_vals']) * intensity_ref)
	c1_data['y_vals'] = final_c1_vals
	c2_data['y_vals'] = final_c2_vals
	if plot_e == True:
		Plotting.plot_graph([c1_data],'Autocorrelation function, E-field',
					'Autocorrelation Function: E-Field in Time Domain',
					'\u03C4','$C_1$(\u03C4)',f'E_autocorrelation_{num_pulses}_pulses_t={t}')
	if plot_intensity == True:
		Plotting.plot_graph([c2_data],'Autocorrelation function, Intensity',
					'Autocorrelation Function: Intensity in Time Domain',
					'\u03C4','$C_2$(\u03C4)',f'I_autocorrelation_{num_pulses}_pulses_t={t}')
	

def monte_carlo_normalised_autocorrelation(FEL,num_trials,t=0,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		autocorrelation_dicts = FEL.autocorrelation(t,num_pulses,bunch_length)
		c1_data = autocorrelation_dicts['c1_dict']
		c2_data = autocorrelation_dicts['c2_dict']
		intensity_data = autocorrelation_dicts['intensity_dict']
		intensity_ref = autocorrelation_dicts['intensity_ref']
		c1_approx_dict = autocorrelation_dicts['c1_approx_dict']
		c2_prediction_dict = autocorrelation_dicts['c1_approx_dict'].copy()
		if i == 0:
			c1_vals = np.zeros(shape=len(c1_data['y_vals']),dtype=complex)
			c2_vals = np.zeros(shape=len(c2_data['y_vals']),dtype=complex)
			intensity_vals = np.zeros(shape=len(intensity_data['y_vals']),dtype=complex)
			intensity_ref_val = 0
			tau_vals = c1_data['x_vals']
		c1_vals += (c1_data['y_vals'])
		c2_vals += (c2_data['y_vals'])
		intensity_vals += (intensity_data['y_vals'])
		intensity_ref_val += intensity_ref
	average_c1_vals = c1_vals / num_trials
	average_c2_vals = c2_vals / num_trials
	average_intensity_vals = intensity_vals / num_trials
	average_intensity_ref = intensity_ref_val / num_trials
	
	final_c1_vals = average_c1_vals / np.sqrt(average_intensity_vals * average_intensity_ref)
	final_c2_vals = average_c2_vals / (average_intensity_vals * average_intensity_ref)
	c2_prediction_dict['y_vals'] = 1 + (np.abs(np.array(final_c1_vals)) ** 2)
	c2_prediction_dict['label'] =  '$C_2 = 1 + |C_1|^2$'
	c1_plot_dict = {
				'x_vals':tau_vals,
				'y_vals':final_c1_vals,
				'label':'1st-order correlation'
				}
	c2_plot_dict = {
				'x_vals':tau_vals,
				'y_vals':final_c2_vals,
				'label':'2nd-order correlation'
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
					'Second-Order Autocorrelation vs Time',
					'\u03C4','simulated average $C_2$(\u03C4)',
					f'C2_monte_carlo_time_{num_trials}_trials',
					show=show_result,scatter=False)
	
	PSD_dict = FEL.PSD(c1_plot_dict)
	Plotting.plot_graph([PSD_dict],'Power spectral density (magnitude)',
					'Power spectral density  (magnitude)',
					'\u03C9 ','S(\u03C9)',
					f'PSD_mag_{num_trials}_trials',
					show=show_result,scatter=False)
					
	phase_PSD_dict = PSD_dict.copy()
	phase_PSD_dict['y_vals'] = phase_PSD_dict['phase']
	Plotting.plot_graph([phase_PSD_dict],'Power spectral density (phase)',
					'Power spectral density (phase)',
					'\u03C9 ','\u03C9(\u03C9)',
					f'PSD_phase_{num_trials}_trials',
					show=show_result,scatter=False)


def monte_carlo_integral_autocorrelation(FEL,num_trials,num_pulses=100,bunch_length=100,show_result=True):
	top_vals = []
	bottom_vals = []
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		incoherent_radiation = show_incoherent_radiation(FEL,num_pulses,bunch_length,False,False,False)
		e_field = incoherent_radiation['superposition_pulses']			
		integral_autocorrelation = FEL.autocorrelation_integral_method(e_field)
		top_vals.append(integral_autocorrelation['top_vals']) 
		bottom_vals.append(integral_autocorrelation['bottom_vals']) 
		if i == 0:
			tau_vals = integral_autocorrelation['c1_dict']['x_vals']
		
	for i,top in enumerate(top_vals):
		if i == 0:
			total_top_vals = np.zeros(shape=len(top),dtype=complex)
			total_bottom_vals = np.zeros(shape=len(top),dtype=complex)
		total_top_vals += top
		total_bottom_vals += bottom_vals[i]
	average_top_vals = total_top_vals/num_trials
	average_bottom_vals = total_bottom_vals/num_trials
	autocorrelation = average_top_vals/average_bottom_vals
	c1_plot_dict = {
				'x_vals':tau_vals,
				'y_vals':autocorrelation,
				'label':'1st-order correlation'
				}
	
	Plotting.plot_graph([c1_plot_dict],'Monte Carlo Trialed $C_1$ in Time Domain (integral method)',
					'First-Order Autocorrelation vs Time (integral method)',
					'\u03C4','simulated average $C_1$(\u03C4)',
					f'C1_monte_carlo_time_{num_trials}_trials',
					show=show_result,scatter=False)
	
	PSD_dict = FEL.PSD(c1_plot_dict)
	print(len(PSD_dict['x_vals']),len(PSD_dict['y_vals']))
	Plotting.plot_graph([PSD_dict],'Power spectral density (magnitude)',
					'Power spectral density  (magnitude)',
					'\u03C9 ','S(\u03C9)',
					f'PSD_mag_{num_trials}_trials',
					show=show_result,scatter=False)
					
	phase_PSD_dict = PSD_dict.copy()
	phase_PSD_dict['y_vals'] = phase_PSD_dict['phase']
	Plotting.plot_graph([phase_PSD_dict],'Power spectral density (phase)',
					'Power spectral density (phase)',
					'\u03C9 ','\u03C9(\u03C9)',
					f'PSD_phase_{num_trials}_trials',
					show=show_result,scatter=False)	


#### OLD FUNCTIONS ####
''' These are analytical frequency profiles. If time allows, these can be tested against the 
	DFT results
'''

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
								
#---------------------------------------------------------------------------------------------------


if __name__ == '__main__':
	main()

