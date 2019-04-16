#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size



#---------------------------------------------------------------------------------------------------
### 								PLOTTING METHODS											 ###
#---------------------------------------------------------------------------------------------------

class Plotting:
	
	@staticmethod
	def plot_graph(data_dicts,fig_name,title,xlabel,ylabel,savefig_name=None,show=True,scatter=False,log=False):
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
				if log == True:
					ax.set_yscale('log')
			if data_dict['label'] == None:
				legend = False
		if legend:
			ax.legend(loc='best', bbox_to_anchor=(0.88, 1.00), shadow=True, ncol=1)
		if savefig_name != None:
			plt.savefig(savefig_name.format('.png'))
		if show == True:
			plt.show()
			
	@staticmethod
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
	
	@staticmethod
	def scatter_and_plot(data_dicts,fig_name,title,xlabel,ylabel,savefig_name=None,show=True,log=False):
		fig = plt.figure(fig_name)
		fig.suptitle(title)
		ax = plt.subplot(111)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		legend = True
		for data_dict in data_dicts:
			if data_dict['scatter'] == True:
				ax.scatter(data_dict['x_vals'],data_dict['y_vals'],label=data_dict['label'],c='red')
			else:
				ax.plot(data_dict['x_vals'],data_dict['y_vals'],label=data_dict['label'])
				if log == True:
					ax.set_yscale('log')
			if data_dict['label'] == None:
				legend = False
		if legend:
			ax.legend(loc='best', bbox_to_anchor=(0.88, 1.00), shadow=True, ncol=1)
		if savefig_name != None:
			plt.savefig(savefig_name.format('.png'))
		if show == True:
			plt.show()
