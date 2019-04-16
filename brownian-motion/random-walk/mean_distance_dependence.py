#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from randomwalk import RandomWalk

n_vals,mean_distance_vals = [],[]
max_n = 2000
for n in range(100,max_n,50):
	rw = RandomWalk(p=0.5,n=n,stats=False)
	rw.plot_monte_carlo(number_of_trials=3000,show=False)

	mean_distance_vals.append(rw.mc_mean_distance)
	n_vals.append(np.sqrt(rw.n))
	
slope, intercept = np.polyfit(n_vals, mean_distance_vals, 1)
print("Constant of proportionality k =",round(slope,3))
print("Intercept c =",round(intercept,3))

fig = plt.figure("Mean distance as a dependence on \u221An, Monte Carlo simulated")
plt.scatter(n_vals,mean_distance_vals,s=4)
plt.suptitle("s vs \u221An")
plt.xlabel('Square root of n')
plt.ylabel('mean distance, s')
plt.show()
