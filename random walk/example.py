from randomwalk import RandomWalk

rw = RandomWalk(p=0.6,n=100)

print(rw.variance)                       #Get the variance of the random walk object
print(rw.mean)                           #Get the mean of the random walk object

rw.plot_distribution()                   #Plot the probability distribution
print(rw.get_confidence_interval(-2,1))  #Get the probability that x_n is between -2 and 1
rw.plot_monte_carlo(histogram=False)     #Run a Monte-Carlo simulation and plot results
