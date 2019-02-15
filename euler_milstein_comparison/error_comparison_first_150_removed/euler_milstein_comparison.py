import random
import time
import numpy as np
from threading import Thread
from threading import Lock
import matplotlib.pyplot as plt

from geometricbm import Analytical
from gbm_euler import Euler
from gbm_milstein import Milstein
import data_storage as ds

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
	
	
def compare_drift():
	
	ds.compared_mu_vals_a = [] 
	ds.compared_mu_vals_e = [] 
	ds.compared_mu_vals_m = [] 
	ds.compared_sigma_vals_a = [] 
	ds.compared_sigma_vals_e = [] 
	ds.compared_sigma_vals_m = [] 
	
	drift_vals = np.arange(0.02,0.52,0.02)
	num_threads = len(drift_vals)*3
	threads_done = 0
	total_start_time = time.time()
	
	for drift in drift_vals:
		start_time = time.time()
		a = Analytical(mu=drift)
		e = Euler(mu=drift)
		m = Milstein(mu=drift)
		threads = [
					Thread(target=a.sim_for_comparison),
					Thread(target=e.sim_for_comparison),
					Thread(target=m.sim_for_comparison)
					]
		for t in threads:
			t.start()
			threads_done += 1
			print('Created thread for varying mu: {}/{}'.format(threads_done,num_threads))
		for th in threads:
			th.join()
		print('Mu = {}'.format(drift))
		print('Time elapsed = {}s'.format(round(time.time()-start_time)))
		print('Total time elapsed = {}s'.format(round(time.time()-total_start_time)))
		print('Progress: {}%\n'.format(round(100*(threads_done/num_threads),2)))

	sorted_mu_a = sorted(ds.compared_mu_vals_a,key=lambda x: x['obj'].mu)
	sorted_sigma_a = sorted(ds.compared_sigma_vals_a,key=lambda x: x['obj'].mu)
	analytical_mu_errors = [i['rel_error'] for i in sorted_mu_a]	
	analytical_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_a]
	analytical_sigma_errors = [i['rel_error'] for i in sorted_sigma_a]	
	analytical_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_a]
	
	sorted_mu_e = sorted(ds.compared_mu_vals_e,key=lambda x: x['obj'].mu)
	sorted_sigma_e = sorted(ds.compared_sigma_vals_e,key=lambda x: x['obj'].mu)
	euler_mu_errors = [i['rel_error'] for i in sorted_mu_e]
	euler_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_e]
	euler_sigma_errors = [i['rel_error'] for i in sorted_sigma_e]	
	euler_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_e]
	
	sorted_mu_m = sorted(ds.compared_mu_vals_m,key=lambda x: x['obj'].mu)
	sorted_sigma_m = sorted(ds.compared_sigma_vals_m,key=lambda x: x['obj'].mu)
	milstein_mu_errors = [i['rel_error'] for i in sorted_mu_m]
	milstein_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_m]
	milstein_sigma_errors = [i['rel_error'] for i in sorted_sigma_m]	
	milstein_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_m]
	
	fig = plt.figure("Comparison of Numerical Methods, varied drift, Geometric Brownian Motion - {} trials".format(a.number_of_sims))
	fig.suptitle("\u03C3 = {}, dt = {}, {} trials. Shown: Relative error in \u03BC, ".format(a.sigma,a.dt,a.number_of_sims))
	ax = plt.subplot(111)
	ax.set_xlabel('Drift value used')
	ax.set_ylabel('Relative error in \u03BC (%)')
	ax.errorbar(drift_vals, analytical_mu_errors, yerr=analytical_mu_error_bars,label='Analytical',fmt='o')
	ax.errorbar(drift_vals, euler_mu_errors, yerr=euler_mu_error_bars,label='Euler',fmt='o')
	ax.errorbar(drift_vals, milstein_mu_errors, yerr=milstein_mu_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('1_drift_mu.png')
	plt.show()
	
	fig = plt.figure("Comparison of Numerical Methods, varied drift, Geometric Brownian Motion - {} trials".format(a.number_of_sims))
	fig.suptitle("\u03C3 = {}, dt = {}, {} trials. Shown: Relative error in \u03C3, ".format(a.sigma,a.dt,a.number_of_sims))
	ax = plt.subplot(111)
	ax.set_xlabel('Drift value used')
	ax.set_ylabel('Relative error in \u03C3 (%)')
	ax.errorbar(drift_vals, analytical_sigma_errors, yerr=analytical_sigma_error_bars,label='Analytical',fmt='o')
	ax.errorbar(drift_vals, euler_sigma_errors, yerr=euler_sigma_error_bars,label='Euler',fmt='o')
	ax.errorbar(drift_vals, milstein_sigma_errors, yerr=milstein_sigma_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('2_drift_sigma.png')
	plt.show()
	
		
def compare_std():
	
	ds.compared_mu_vals_a = [] 
	ds.compared_mu_vals_e = [] 
	ds.compared_mu_vals_m = [] 
	ds.compared_sigma_vals_a = [] 
	ds.compared_sigma_vals_e = [] 
	ds.compared_sigma_vals_m = [] 
	
	std_vals = np.arange(0.02,0.52,0.02)
	num_threads = len(std_vals)*3
	threads_done = 0
	total_start_time = time.time()
	
	for std in std_vals:
		start_time = time.time()
		a = Analytical(sigma=std)
		e = Euler(sigma=std)
		m = Milstein(sigma=std)
		threads = [
					Thread(target=a.sim_for_comparison),
					Thread(target=e.sim_for_comparison),
					Thread(target=m.sim_for_comparison)
					]
		for t in threads:
			t.start()
			threads_done += 1
			print('Created thread for varying sigma: {}/{}'.format(threads_done,num_threads))
		for th in threads:
			th.join()
		print('Sigma = {}'.format(std))
		print('Time elapsed = {}s'.format(round(time.time()-start_time)))
		print('Total time elapsed = {}s'.format(round(time.time()-total_start_time)))
		print('Progress: {}%\n'.format(round(100*(threads_done/num_threads),2)))

		
	sorted_mu_a = sorted(ds.compared_mu_vals_a,key=lambda x: x['obj'].sigma)
	sorted_sigma_a = sorted(ds.compared_sigma_vals_a,key=lambda x: x['obj'].sigma)
	analytical_mu_errors = [i['rel_error'] for i in sorted_mu_a]	
	analytical_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_a]
	analytical_sigma_errors = [i['rel_error'] for i in sorted_sigma_a]	
	analytical_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_a]
	
	sorted_mu_e = sorted(ds.compared_mu_vals_e,key=lambda x: x['obj'].sigma)
	sorted_sigma_e = sorted(ds.compared_sigma_vals_e,key=lambda x: x['obj'].sigma)
	euler_mu_errors = [i['rel_error'] for i in sorted_mu_e]
	euler_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_e]
	euler_sigma_errors = [i['rel_error'] for i in sorted_sigma_e]	
	euler_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_e]
	
	sorted_mu_m = sorted(ds.compared_mu_vals_m,key=lambda x: x['obj'].sigma)
	sorted_sigma_m = sorted(ds.compared_sigma_vals_m,key=lambda x: x['obj'].sigma)
	milstein_mu_errors = [i['rel_error'] for i in sorted_mu_m]
	milstein_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_m]
	milstein_sigma_errors = [i['rel_error'] for i in sorted_sigma_m]	
	milstein_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_m]
	
	
	fig = plt.figure("Comparison of Numerical Methods, varied standard deviation, Geometric Brownian Motion - {} trials".format(a.number_of_sims))
	fig.suptitle("Shown: Relative error in \u03BC, \u03BC = {}, dt = {}, {} trials".format(a.mu,a.dt,a.number_of_sims))
	ax = plt.subplot(111)
	ax.set_xlabel('Sigma value used')
	ax.set_ylabel('Relative error in \u03BC (%)')
	ax.errorbar(std_vals, analytical_mu_errors, yerr=analytical_mu_error_bars,label='Analytical',fmt='o')
	ax.errorbar(std_vals, euler_mu_errors, yerr=euler_mu_error_bars,label='Euler',fmt='o')
	ax.errorbar(std_vals, milstein_mu_errors, yerr=milstein_mu_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('3_std_mu.png')
	plt.show()
	
	fig = plt.figure("Comparison of Numerical Methods, varied standard deviation, Geometric Brownian Motion - {} trials".format(a.number_of_sims))
	fig.suptitle("Shown: Relative error in \u03C3, \u03BC = {}, dt = {}, {} trials".format(a.mu,a.dt,a.number_of_sims))
	ax = plt.subplot(111)
	ax.set_xlabel('Sigma value used')
	ax.set_ylabel('Relative error in \u03C3 (%)')
	ax.errorbar(std_vals, analytical_sigma_errors, yerr=analytical_sigma_error_bars,label='Analytical',fmt='o')
	ax.errorbar(std_vals, euler_sigma_errors, yerr=euler_sigma_error_bars,label='Euler',fmt='o')
	ax.errorbar(std_vals, milstein_sigma_errors, yerr=milstein_sigma_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('4_std_sigma.png')
	plt.show()
	

def compare_time_step():
	
	ds.compared_mu_vals_a = [] 
	ds.compared_mu_vals_e = [] 
	ds.compared_mu_vals_m = [] 
	ds.compared_sigma_vals_a = [] 
	ds.compared_sigma_vals_e = [] 
	ds.compared_sigma_vals_m = [] 
	
	dt_vals = np.arange(0.002,0.052,0.002)
	num_threads = len(dt_vals)*3
	threads_done = 0
	total_start_time = time.time()
	
	for dt in dt_vals:
		start_time = time.time()
		a = Analytical(dt=dt,t_max=dt*500)
		e = Euler(dt=dt,t_max=dt*500)
		m = Milstein(dt=dt,t_max=dt*500)
		threads = [
					Thread(target=a.sim_for_comparison),
					Thread(target=e.sim_for_comparison),
					Thread(target=m.sim_for_comparison)
					]
		for t in threads:
			t.start()
			threads_done += 1
			print('Created thread for varying dt: {}/{}'.format(threads_done,num_threads))
		for th in threads:
			th.join()
		print('dt = {}'.format(dt))
		print('Time elapsed = {}s'.format(round(time.time()-start_time)))
		print('Total time elapsed = {}s'.format(round(time.time()-total_start_time)))
		print('Progress: {}%\n'.format(round(100*(threads_done/num_threads),2)))


		
	sorted_mu_a = sorted(ds.compared_mu_vals_a,key=lambda x: x['obj'].dt)
	sorted_sigma_a = sorted(ds.compared_sigma_vals_a,key=lambda x: x['obj'].dt)
	analytical_mu_errors = [i['rel_error'] for i in sorted_mu_a]	
	analytical_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_a]
	analytical_sigma_errors = [i['rel_error'] for i in sorted_sigma_a]	
	analytical_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_a]
	
	sorted_mu_e = sorted(ds.compared_mu_vals_e,key=lambda x: x['obj'].dt)
	sorted_sigma_e = sorted(ds.compared_sigma_vals_e,key=lambda x: x['obj'].dt)
	euler_mu_errors = [i['rel_error'] for i in sorted_mu_e]
	euler_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_e]
	euler_sigma_errors = [i['rel_error'] for i in sorted_sigma_e]	
	euler_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_e]
	
	sorted_mu_m = sorted(ds.compared_mu_vals_m,key=lambda x: x['obj'].dt)
	sorted_sigma_m = sorted(ds.compared_sigma_vals_m,key=lambda x: x['obj'].dt)
	milstein_mu_errors = [i['rel_error'] for i in sorted_mu_m]
	milstein_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_m]
	milstein_sigma_errors = [i['rel_error'] for i in sorted_sigma_m]	
	milstein_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_m]
	
	fig = plt.figure("Comparison of Numerical Methods, varied time step duration, Geometric Brownian Motion - {} trials".format(a.number_of_sims))
	fig.suptitle("Shown: Relative error in \u03BC, \u03BC = {}, \u03C3 = {}, {} trials".format(a.mu,a.sigma,a.number_of_sims))
	ax = plt.subplot(111)
	ax.set_xlabel('dt value used')
	ax.set_ylabel('Relative error in \u03BC (%)')
	ax.errorbar(dt_vals, analytical_mu_errors, yerr=analytical_mu_error_bars,label='Analytical',fmt='o')
	ax.errorbar(dt_vals, euler_mu_errors, yerr=euler_mu_error_bars,label='Euler',fmt='o')
	ax.errorbar(dt_vals, milstein_mu_errors, yerr=milstein_mu_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('5_dt_mu_png')
	plt.show()
	
	fig = plt.figure("Comparison of Numerical Methods, varied time step duration, Geometric Brownian Motion - {} trials".format(a.number_of_sims))
	fig.suptitle("Shown: Relative error in \u03C3, \u03BC = {}, \u03C3 = {}, {} trials".format(a.mu,a.sigma,a.number_of_sims))
	ax = plt.subplot(111)
	ax.set_xlabel('dt value used')
	ax.set_ylabel('Relative error in \u03C3 (%)')
	ax.errorbar(dt_vals, analytical_sigma_errors, yerr=analytical_sigma_error_bars,label='Analytical',fmt='o')
	ax.errorbar(dt_vals, euler_sigma_errors, yerr=euler_sigma_error_bars,label='Euler',fmt='o')
	ax.errorbar(dt_vals, milstein_sigma_errors, yerr=milstein_sigma_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('6_dt_sigma.png')
	plt.show()
	
	
def compare_num_trials():
	
	ds.compared_mu_vals_a = [] 
	ds.compared_mu_vals_e = [] 
	ds.compared_mu_vals_m = [] 
	ds.compared_sigma_vals_a = [] 
	ds.compared_sigma_vals_e = [] 
	ds.compared_sigma_vals_m = [] 
	
	trial_nums = list(np.arange(50,400,25)) + list(np.arange(400,3200,200))
	num_threads = len(trial_nums)*3
	threads_done = 0
	total_start_time = time.time()
	
	for num in trial_nums:
		start_time = time.time()
		a = Analytical(number_of_sims=num)
		e = Euler(number_of_sims=num)
		m = Milstein(number_of_sims=num)
		threads = [
					Thread(target=a.sim_for_comparison),
					Thread(target=e.sim_for_comparison),
					Thread(target=m.sim_for_comparison)
					]
		for t in threads:
			t.start()
			threads_done += 1
			print('Created thread for varying number of simulations: {}/{}'.format(threads_done,num_threads))
		for th in threads:
			th.join()
		print('Number of trials = {}'.format(num))
		print('Time elapsed = {}s'.format(round(time.time()-start_time)))
		print('Total time elapsed = {}s'.format(round(time.time()-total_start_time)))
		print('Progress: {}%\n'.format(round(100*(threads_done/num_threads),2)))

	
	sorted_mu_a = sorted(ds.compared_mu_vals_a,key=lambda x: x['obj'].mu)
	sorted_sigma_a = sorted(ds.compared_sigma_vals_a,key=lambda x: x['obj'].mu)
	analytical_mu_errors = [i['rel_error'] for i in sorted_mu_a]	
	analytical_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_a]
	analytical_sigma_errors = [i['rel_error'] for i in sorted_sigma_a]	
	analytical_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_a]
	
	sorted_mu_e = sorted(ds.compared_mu_vals_e,key=lambda x: x['obj'].mu)
	sorted_sigma_e = sorted(ds.compared_sigma_vals_e,key=lambda x: x['obj'].mu)
	euler_mu_errors = [i['rel_error'] for i in sorted_mu_e]
	euler_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_e]
	euler_sigma_errors = [i['rel_error'] for i in sorted_sigma_e]	
	euler_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_e]
	
	sorted_mu_m = sorted(ds.compared_mu_vals_m,key=lambda x: x['obj'].mu)
	sorted_sigma_m = sorted(ds.compared_sigma_vals_m,key=lambda x: x['obj'].mu)
	milstein_mu_errors = [i['rel_error'] for i in sorted_mu_m]
	milstein_mu_error_bars = [i['rel_mrg_err'] for i in sorted_mu_m]
	milstein_sigma_errors = [i['rel_error'] for i in sorted_sigma_m]	
	milstein_sigma_error_bars = [i['rel_mrg_err'] for i in sorted_sigma_m]
	
	fig = plt.figure("Comparison of Numerical Methods, varied number of trials, Geometric Brownian Motion")
	fig.suptitle("Shown: Relative error in \u03BC, \u03BC = {}, \u03C3 = {}, dt = {}".format(a.mu,a.sigma,a.dt))
	ax = plt.subplot(111)
	ax.set_xlabel('Number of simulations used')
	ax.set_ylabel('Relative error (%)')
	ax.errorbar(trial_nums, analytical_mu_errors, yerr=analytical_mu_error_bars,label='Analytical',fmt='o')
	ax.errorbar(trial_nums, euler_mu_errors, yerr=euler_mu_error_bars,label='Euler',fmt='o')
	ax.errorbar(trial_nums, milstein_mu_errors, yerr=milstein_mu_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('7_num_sims_mu.png')
	plt.show()
	
	fig = plt.figure("Comparison of Numerical Methods, varied numbe of trials, Geometric Brownian Motion")
	fig.suptitle("Shown: Relative error in \u03C3, \u03BC = {}, \u03C3 = {}, dt = {}".format(a.mu,a.sigma,a.dt))
	ax = plt.subplot(111)
	ax.set_xlabel('Number of simulations used')
	ax.set_ylabel('Relative error (%)')
	ax.errorbar(trial_nums, analytical_sigma_errors, yerr=analytical_sigma_error_bars,label='Analytical',fmt='o')
	ax.errorbar(trial_nums, euler_sigma_errors, yerr=euler_sigma_error_bars,label='Euler',fmt='o')
	ax.errorbar(trial_nums, milstein_sigma_errors, yerr=milstein_sigma_error_bars,label='Milstein',fmt='o')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('8_num_sims_sigma.png')
	plt.show()
	
	
def compare_models():
	ds.compared_mu_vals_a = [] 
	ds.compared_mu_vals_e = [] 
	ds.compared_mu_vals_m = [] 
	ds.compared_sigma_vals_a = [] 
	ds.compared_sigma_vals_e = [] 
	ds.compared_sigma_vals_m = [] 
	
	a = Analytical(s_initial=10,mu=0.1,sigma=0.08,dt=0.05,t_max=200,number_of_sims=1000)
	e = Euler(s_initial=10,mu=0.1,sigma=0.08,dt=0.05,t_max=200,number_of_sims=1000)
	m = Milstein(s_initial=10,mu=0.1,sigma=0.08,dt=0.05,t_max=200,number_of_sims=1000)
	threads = [
				Thread(target=a.sim_for_comparison),
				Thread(target=e.sim_for_comparison),
				Thread(target=m.sim_for_comparison)
				]
	for t in threads:
		t.start()
	for th in threads:
		th.join()
	
	sorted_mu_a = ds.compared_mu_vals_a
	sorted_sigma_a = ds.compared_sigma_vals_a
	analytical_mu_errors = sorted_mu_a[0]['computed_mu_rel_errors']
	analytical_sigma_errors = sorted_sigma_a[0]['computed_sigma_rel_errors']
	
	sorted_mu_e = ds.compared_mu_vals_e
	sorted_sigma_e = ds.compared_sigma_vals_e
	euler_mu_errors = sorted_mu_e[0]['computed_mu_rel_errors']
	euler_sigma_errors = sorted_sigma_e[0]['computed_sigma_rel_errors']	
	
	sorted_mu_m = ds.compared_mu_vals_m
	sorted_sigma_m = ds.compared_sigma_vals_m
	milstein_mu_errors = sorted_mu_m[0]['computed_mu_rel_errors']
	milstein_sigma_errors = sorted_sigma_m[0]['computed_sigma_rel_errors']
		
	time_vals = np.arange(a.dt*150,a.t_max,a.dt)
	
	fig = plt.figure("Comparison of Numerical Methods, Geometric Brownian Motion")
	fig.suptitle("Shown: Relative error in \u03BC, \u03BC = {}, \u03C3 = {}, dt = {}".format(a.mu,a.sigma,a.dt))
	ax = plt.subplot(111)
	ax.set_xlabel('Time elapsed')
	ax.set_ylabel('Relative error (%)')
	ax.plot(time_vals, analytical_mu_errors,label='Analytical')
	ax.plot(time_vals, euler_mu_errors,label='Euler')
	ax.plot(time_vals, milstein_mu_errors,label='Milstein')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('9_optimised_mu.png')
	plt.show()
	
	fig = plt.figure("Comparison of Numerical Methods, Geometric Brownian Motion")
	fig.suptitle("Shown: Relative error in \u03C3, \u03BC = {}, \u03C3 = {}, dt = {}".format(a.mu,a.sigma,a.dt))
	ax = plt.subplot(111)
	ax.set_xlabel('Time elapsed')
	ax.set_ylabel('Relative error (%)')
	ax.plot(time_vals, analytical_sigma_errors,label='Analytical')
	ax.plot(time_vals, euler_sigma_errors,label='Euler')
	ax.plot(time_vals, milstein_sigma_errors,label='Milstein')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
	plt.savefig('10_optimised_sigma.png')
	plt.show()	
	
	
	
if __name__ == '__main__':
	#compare_drift()
	#compare_std()
	#compare_time_step()
	compare_num_trials()
	#compare_models()
