#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from randomwalk import RandomWalk


# ~~~ CREATE RANDOM WALK INSTANCE ~~~ #
rw = RandomWalk(p=0.5,n=800) 					#Max n is ~800


# ~~~ PRINT RANDOM WALK STATS ~~~ #
print("Mean:",rw.mean)                         	#Print the mean of the random walk object
print("Variance:",rw.variance,'\n')            	#Print the variance of the random walk object
print("Mean distance:",rw.mean_distance)		#Print theoretical mean distance from starting position
print("Square root of n:",np.sqrt(rw.n),'\n')	#Print square root of n for comparison with mean distance


# ~~~  PLOT THEORETICAL PROBABILITY DISTRIBUTION ~~~ #
rw.plot_distribution(show=False)


# ~~~ GET PROBABILITY THAT GIVEN X_VALUE IS WITHIN CONFIDENCE INTERVAL ~~~ #
p_conf = rw.get_confidence_interval(-2,1)			#Get probability that x_n is between -2 and 1.
print("Confidence interval probability:",p_conf)	#Print this probability


# ~~~ RUN MONTE CARLO SIMULATION, PLOTTING RESULTS AND PRINTING STATS ~~~ #
rw.run_monte_carlo(number_of_trials=10000,show=False,plot=True,histogram=True)
print("Monte Carlo mean:",rw.mc_mean)            		#Print the mean of the M.C. results
print("Monte Carlo variance:",rw.mc_variance,'\n') 		#Print the variance of the M.C. results
print("Monte Carlo mean distance:",rw.mc_mean_distance) #Print the mean distance of the M.C. results
print("Square root of n:",np.sqrt(rw.n))				#Print the square root of n for comparison with mean distance


# ~~~  DRAW REAL RANDOM WALK SIMULATION (STATIC GRAPH OR LIVE ANIMATION) ~~~ #
rw.random_walk_draw(10,animated=False,show=False)		#Plot simulation of 10 random walks on the same figure


# ~~~ SHOW PLOTS ~~~ #
"""Notice how each plotting method has the parameter "show=False". This is so they can be shown at 
the end instead of individually.
"""
plt.show()

