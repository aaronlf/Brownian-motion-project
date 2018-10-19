#!/usr/bin/env python3

from randomwalk import RandomWalk

rw = RandomWalk(p=0.6,n=100)

print("Mean:",rw.mean)                           #Get the mean of the random walk object
print("Variance:",rw.variance)                   #Get the variance of the random walk object

rw.plot_distribution()                           #Plot the probability distribution
print("Confidence interval probability:",
	  rw.get_confidence_interval(-2,1))          #Get the probability that x_n is between -2 and 1

rw.plot_monte_carlo(number_of_trials=5000)       #Run a Monte-Carlo simulation and plot results
print("Monte Carlo mean:",rw.mc_mean)            #Get the mean of the Monte Carlo results
print("Monte Carlo variance:",rw.mc_variance)    #Get the variance of the Monte Carlo results

