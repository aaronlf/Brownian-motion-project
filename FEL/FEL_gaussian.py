#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from bisect import bisect_left

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size


#---------------------------------------------------------------------------------------------------
### 									MAIN FUNCTION 											 ###
#---------------------------------------------------------------------------------------------------

def main():
	#SHOW_COHERENT_RADIATION()
	#SHOW_INCOHERENT_RADIATION(num_pulses=100,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True)
	#INCOHERENT_AVERAGE_INTENSITY_VS_NUM_SOURCES()
	#SHOW_INCOHERENT_RADIATION_OMEGA(100,plot_superposition=True,plot_intensity=True)
	MONTE_CARLO_INTENSITY_TIME(num_trials=100,num_pulses=100,bunch_length=24,show_result=True)
	#MONTE_CARLO_INTENSITY_OMEGA(num_trials=500,num_pulses=100,bunch_length=100,show_result=True)
	#SHOW_AUTOCORRELATION_TIME(t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	#MONTE_CARLO_AUTOCORRELATION_TIME(num_trials=500,t=0,num_pulses=100,bunch_length=100,show_result=True)
	
	#SHOW_AUTOCORRELATION_FREQUENCY(delta_omega=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True)
	
	#MONTE_CARLO_AUTOCORRELATION_FREQUENCY(num_trials=500,delta_omega=0,num_pulses=100,bunch_length=100,show_result=True)
	
#---------------------------------------------------------------------------------------------------
### 					CLASS TO DEMONSTRATE TEMPORAL COHERENCE/INCOHERECE 						 ###
#---------------------------------------------------------------------------------------------------	
	
class Temporal:
	
	def __init__(self,sigma,omega):
		self.sigma = sigma
		self.omega = omega
		
	def coherent_rad(self,t,sigma,omega):
		return np.exp(-(t**2)/(4*(sigma**2))) * np.cos(omega*t)
		
	def model_coherent_rad_once(self,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		time_vals = np.linspace(-20,20,500)
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
		time_vals = np.linspace(-(bunch_length*0.5)-(4*sigma),(bunch_length*0.5)+(4*sigma),500)
		e_vals = []
		tj = random.uniform(-(bunch_length*0.5),(bunch_length*0.5))
		for t in time_vals:
			e_vals.append(self.incoherent_rad(t,tj,sigma,omega,bunch_length))
		return {'x_vals':time_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9 = {}'.format(sigma,omega)}
		
	def get_superposition(self,incoherent_vals_list):
		output_vals = np.zeros(shape=len(incoherent_vals_list[0]['y_vals']))
		for incoherent_vals in incoherent_vals_list:
			output_vals += np.array(incoherent_vals['y_vals'])
		return {
				'x_vals':incoherent_vals_list[0]['x_vals'],
				'y_vals':output_vals,
				'label':None
				}
				
	def get_intensity(self,e_superposition):
		intensity_vals = e_superposition['y_vals'] ** 2
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
		omega_vals = np.linspace(omega1-1,omega1+1,5000)
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
	

#---------------------------------------------------------------------------------------------------
### 								PLOTTING FUNCTIONS											 ###
#---------------------------------------------------------------------------------------------------

def plot_graph(data_dicts,fig_name,title,xlabel,ylabel,savefig_name=None,show=True,scatter=False):
	fig = plt.figure(fig_name)
	fig.suptitle(title)
	ax = plt.subplot(111)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	legend = True
	for data_dict in data_dicts:
		if scatter == True:
			ax.scatter(data_dict['x_vals'],data_dict['y_vals'],label=data_dict['label'])
		else:
			ax.plot(data_dict['x_vals'],data_dict['y_vals'],label=data_dict['label'])
		if data_dict['label'] == None:
			legend = False
	if legend:
		ax.legend(loc='upper center', bbox_to_anchor=(0.92, 1.00), shadow=True, ncol=1)
	if savefig_name != None:
		plt.savefig(savefig_name.format('.png'))
	if show == True:
		plt.show()

def plot_subplots(data_dicts,fig_name,title,small_title,xlabel,savefig_name=None,show=True):
	fig = plt.figure(fig_name)
	fig.suptitle(title)
	for i,data_dict in enumerate(data_dicts):
		ax = plt.subplot(len(data_dicts), 1, i+1)
		ax.axes.yaxis.set_ticklabels([])
		plt.plot(data_dict['x_vals'],data_dict['y_vals'],label=data_dict['label'])
		if i == 0:
			plt.title(small_title)
		if i == len(data_dicts) - 1:
			plt.xlabel(xlabel)
		else:
			ax.axes.xaxis.set_ticklabels([])
	if savefig_name != None:
		plt.savefig(savefig_name.format('.png'))
	if show == True:
		plt.show()
		
		
#---------------------------------------------------------------------------------------------------
### 								SUBROUTINES OF MAIN											 ###
#---------------------------------------------------------------------------------------------------


def SHOW_COHERENT_RADIATION():
	coherent_vals0 = temp.model_coherent_rad_once(sigma=2)
	coherent_vals1 = temp.model_coherent_rad_once(sigma=1)
	plot_graph([coherent_vals0,coherent_vals1],'Coherent Radiation','Coherent Radiation','t',
				'E/e$_0$','coherent_radiation')
	
	
def SHOW_INCOHERENT_RADIATION(num_pulses,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True,print_mode_num=True):
	
	incoherent_vals_list = [temp.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]
	if subplots == True:
		plot_subplots(incoherent_vals_list[:10],'Incoherent Radiation','Incoherent Radiation',
				'Shown: 10 randomly spaced pulses','time','incoherent_radiation_individual_pulses')
				
	superposition_pulses = temp.get_superposition(incoherent_vals_list)
	if plot_superposition == True:
		plot_graph([superposition_pulses],'Incoherent Radiation',
					'Incoherent Radiation: Superposition of {} pulses'.format(num_pulses),
					'time','E/e$_0$','incoherent_radiation_superposition_{}_pulses'.format(num_pulses))
	
	intensity = temp.get_intensity(superposition_pulses)
	if plot_intensity == True:
		plot_graph([intensity],'Incoherent Radiation Intensity Profile',
					'Incoherent Radiation: Intensity profile for {} pulses'.format(num_pulses),
					'time','intensity','intensity_incoherent_radiation_{}_pulses'.format(num_pulses))
	
	if print_mode_num == True:
		print('Approximate number of longitudinal modes = {}'.format(temp.get_num_l_modes(bunch_length)))
	
	return {'superposition_pulses':superposition_pulses,'intensity':intensity}


def INCOHERENT_AVERAGE_INTENSITY_VS_NUM_SOURCES(bunch_length=100):
	num_pulse_vals = np.arange(10,400,10)
	average_intensities = []
	for num_pulses in num_pulse_vals:
		print(num_pulses)
		incoherent_vals_list = [temp.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]
		superposition_pulses = temp.get_superposition(incoherent_vals_list)
		intensity_profile = temp.get_intensity(superposition_pulses)['y_vals']
		average_intensity = np.mean(intensity_profile)
		average_intensities.append(average_intensity)
	plot_dict = {
				'x_vals':num_pulse_vals,
				'y_vals':average_intensities,
				'label':None
				}
	plot_graph([plot_dict],'Incoherent Radiation Average Intensity',
					'Incoherent Radiation: Average intensity vs number of sources',
					'number of sources','average intensity',
					'average_intensity_vs_num_pulses',scatter=True)
	

def MONTE_CARLO_INTENSITY_TIME(num_trials,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		intensity = SHOW_INCOHERENT_RADIATION(num_pulses,bunch_length,False,False,False,False)['intensity']
		if i == 0:
			intensity_vals = np.zeros(shape=len(intensity['y_vals']))
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
	plot_graph([plot_dict],'Monte Carlo Trialed Average Intensity Profile in Time Domain',
					'Average Intensity vs Time',
					'time ','simulated average intensity',
					'monte_carlo_time_average_intensity_{}_trials'.format(num_trials),
					show=show_result,scatter=False)
	
										
	
def SHOW_INCOHERENT_RADIATION_OMEGA(num_pulses,bunch_length=100,plot_superposition=True,plot_intensity=True):
	
	incoherent_vals_list_omega = [temp.model_incoherent_rad_once_omega(bunch_length) for i in range(num_pulses)]
		
	superposition_pulses_omega = temp.get_superposition(incoherent_vals_list_omega)
	if plot_superposition == True:
		plot_graph([superposition_pulses_omega],'Incoherent Radiation',
					'Incoherent Radiation: Superposition of {} pulses in terms of \u03C9'.format(num_pulses),
					'\u03C9 - \u03C9$_1$','E/e$_0$','omega_incoherent_radiation_superposition_{}_pulses'.format(num_pulses))
	#'E\u03C3/e$_0\sqrt{\u03C0}$'
	intensity_omega = temp.get_intensity(superposition_pulses_omega)
	if plot_intensity == True:
		plot_graph([intensity_omega],'Incoherent Radiation Intensity Spectrum',
					'Incoherent Radiation: Intensity spectrum for {} pulses in terms of \u03C9'.format(num_pulses),
					'\u03C9 - \u03C9$_1$','P(\u03C9)','omega_intensity_incoherent_radiation_{}_pulses'.format(num_pulses))
	
	temp.get_num_l_modes(bunch_length)
	return {'superposition_pulses':superposition_pulses_omega,'intensity':intensity_omega}
					
					
def MONTE_CARLO_INTENSITY_OMEGA(num_trials,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		intensity = SHOW_INCOHERENT_RADIATION_OMEGA(num_pulses,bunch_length,False,False)['intensity']
		if i == 0:
			intensity_vals = np.zeros(shape=len(intensity['y_vals']))
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
	plot_graph([plot_dict],'Monte Carlo Trialed Average Intensity Profile in Frequency Domain',
					'Average Intensity Spectrum',
					'\u03C9 - \u03C9$_1$','simulated average intensity',
					'monte_carlo_frequency_average_intensity_{}_trials'.format(num_trials),
					show=show_result,scatter=False)	
		

def SHOW_AUTOCORRELATION_TIME(t=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True):
	
	autocorrelation_dicts = temp.autocorrelation_time(t,num_pulses,bunch_length)
	c1_data = autocorrelation_dicts['c1_dict']
	c2_data = autocorrelation_dicts['c2_dict']
	
	if plot_e == True:
		plot_graph([c1_data],'Autocorrelation function, E-field',
					'Autocorrelation Function: E-Field in Time Domain',
					'\u03C4','$C_1$(\u03C4)',f'E_autocorrelation_{num_pulses}_pulses_t={t}')

	if plot_intensity == True:
		plot_graph([c2_data],'Autocorrelation function, Intensity',
					'Autocorrelation Function: Intensity in Time Domain',
					'\u03C4','$C_2$(\u03C4)',f'I_autocorrelation_{num_pulses}_pulses_t={t}')
	
	

def MONTE_CARLO_AUTOCORRELATION_TIME(num_trials,t=0,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		autocorrelation_dicts = temp.autocorrelation_time(t,num_pulses,bunch_length)
		c1_data = autocorrelation_dicts['c1_dict']
		c2_data = autocorrelation_dicts['c2_dict']
		if i == 0:
			c1_vals = np.zeros(shape=len(c1_data['y_vals']))
			c2_vals = np.zeros(shape=len(c2_data['y_vals']))
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
			
	plot_graph([c1_plot_dict],'Monte Carlo Trialed $C_1$ in Time Domain',
					'First-Order Autocorrelation vs Time',
					'\u03C4','simulated average $C_1$(\u03C4)',
					f'C1_monte_carlo_time_{num_trials}_trials',
					show=show_result,scatter=False)
	plot_graph([c2_plot_dict],'Monte Carlo Trialed $C_2$ in Time Domain',
					'First-Order Autocorrelation vs Time',
					'\u03C4','simulated average $C_2$(\u03C4)',
					f'C2_monte_carlo_time_{num_trials}_trials',
					show=show_result,scatter=False)
					

def SHOW_AUTOCORRELATION_FREQUENCY(delta_omega=0,num_pulses=100,bunch_length=100,plot_e=True,plot_intensity=True):
	autocorrelation_dicts = temp.autocorrelation_freq(delta_omega,num_pulses,bunch_length)
	c1_data = autocorrelation_dicts['c1_dict']
	c2_data = autocorrelation_dicts['c2_dict']
	
	if plot_e == True:
		plot_graph([c1_data],'Autocorrelation function, E-field',
					'Autocorrelation Function: E-Field in Frequency Domain',
					'Variation in \u03C9','$C_1$',f'E_autocorrelation_{num_pulses}_pulses_delta_omega={delta_omega}')

	if plot_intensity == True:
		plot_graph([c2_data],'Autocorrelation function, Intensity',
					'Autocorrelation Function: Intensity in Frequency Domain',
					'Variation in \u03C9','$C_2$',f'I_autocorrelation_{num_pulses}_pulses_delta_omega={delta_omega}')	


def MONTE_CARLO_AUTOCORRELATION_FREQUENCY(num_trials,delta_omega=0,num_pulses=100,bunch_length=100,show_result=True):
	for i in range(num_trials):
		print('Trial number: {}/{}'.format(i+1,num_trials))
		autocorrelation_dicts = temp.autocorrelation_freq(delta_omega,num_pulses,bunch_length)
		c1_data = autocorrelation_dicts['c1_dict']
		c2_data = autocorrelation_dicts['c2_dict']
		if i == 0:
			c1_vals = np.zeros(shape=len(c1_data['y_vals']))
			c2_vals = np.zeros(shape=len(c2_data['y_vals']))
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
			
	plot_graph([c1_plot_dict],'Monte Carlo Trialed $C_1$ in Frequency Domain',
					'First-Order Autocorrelation vs Change in Frequency',
					'Variation in \u03C9','simulated average $C_1$',
					f'C1_monte_carlo_freq_{num_trials}_trials',
					show=show_result,scatter=False)
	plot_graph([c2_plot_dict],'Monte Carlo Trialed $C_2$ in Frequency Domain',
					'First-Order Autocorrelation vs Change in Frequency',
					'Variation in \u03C9','simulated average $C_2$',
					f'C2_monte_carlo_freq_{num_trials}_trials',
					show=show_result,scatter=False)					
									
#---------------------------------------------------------------------------------------------------
	
	
if __name__ == '__main__':
	temp = Temporal(sigma=2,omega=6)
	main()
