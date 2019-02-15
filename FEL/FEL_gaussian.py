#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size


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
		return np.exp(-((t-tj)**2)/(4*(sigma**2))) * np.cos(omega*(t-tj))
	
	def model_incoherent_rad_once(self,bunch_length,sigma=None,omega=None):
		if sigma == None:
			sigma = self.sigma
		if omega == None:
			omega = self.omega
		time_vals = np.linspace(-(bunch_length*0.5)-(4*sigma),(bunch_length*0.5)+(4*sigma),5000)
		e_vals = []
		tj = random.uniform(-(bunch_length*0.5),(bunch_length*0.5))
		for t in time_vals:
			e_vals.append(self.incoherent_rad(t,tj,sigma,omega,bunch_length))
		return {'x_vals':time_vals,'y_vals':e_vals,'label':'\u03C3 = {}, \u03C9 = {}'.format(sigma,omega)}
		
	def get_superposition(self,coherent_vals_list):
		output_vals = np.zeros(shape=len(coherent_vals_list[0]['y_vals']))
		for coherent_vals in coherent_vals_list:
			output_vals += np.array(coherent_vals['y_vals'])
		return {
				'x_vals':coherent_vals_list[0]['x_vals'],
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
		print('Approximate number of longitudinal modes = {}'.format(M_L))
		return M_L


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


temp = Temporal(sigma=2,omega=6)


def SHOW_COHERENT_RADIATION():
	coherent_vals0 = temp.model_coherent_rad_once()
	coherent_vals1 = temp.model_coherent_rad_once()
	plot_graph([coherent_vals0,coherent_vals1],'Coherent Radiation','Coherent Radiation','t',
				'E/e','coherent_radiation')
	
	
def SHOW_INCOHERENT_RADIATION(num_pulses,bunch_length=100,subplots=True,plot_superposition=True,plot_intensity=True):
	
	incoherent_vals_list = [temp.model_incoherent_rad_once(bunch_length) for i in range(num_pulses)]
	if subplots == True:
		plot_subplots(incoherent_vals_list[:10],'Incoherent Radiation','Incoherent Radiation',
				'Shown: 10 randomly spaced pulses','time','incoherent_radiation_individual_pulses')
				
	superposition_pulses = temp.get_superposition(incoherent_vals_list)
	if plot_superposition == True:
		plot_graph([superposition_pulses],'Incoherent Radiation',
					'Incoherent Radiation: Superposition of {} pulses'.format(num_pulses),
					'time','E/e','incoherent_radiation_superposition_{}_pulses'.format(num_pulses))
	
	intensity = temp.get_intensity(superposition_pulses)
	if plot_intensity == True:
		plot_graph([intensity],'Incoherent Radiation Intensity Profile',
					'Incoherent Radiation: Intensity profile for {} pulses'.format(num_pulses),
					'time','intensity','intensity_incoherent_radiation_{}_pulses'.format(num_pulses))
	
	temp.get_num_l_modes(bunch_length)


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
	
	
#---------------------------------------------------------------------------------------------------
	
	
if __name__ == '__main__':
	#SHOW_COHERENT_RADIATION()
	SHOW_INCOHERENT_RADIATION(100,subplots=True,plot_superposition=True,plot_intensity=True)
	#INCOHERENT_AVERAGE_INTENSITY_VS_NUM_SOURCES()
	
