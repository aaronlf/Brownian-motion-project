#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from randomwalk2d import RandomWalk2D


# ~~~ CREATE 2D RANDOM WALK INSTANCE ~~~ #
rw = RandomWalk2D(n=200)


# ~~~  DRAW REAL RANDOM WALK SIMULATION (STATIC GRAPH OR LIVE ANIMATION) ~~~ ##
rw.random_walk_draw(50,animated=True,projection='2d')
rw.random_walk_draw(50,animated=True,projection='3d')



# Note: There is a bug preventing plt.show() to display 2 graphs if plt.show() is at the the end.
# It must be called individually in each random_walk_draw() call, as above.

# plt.show()
