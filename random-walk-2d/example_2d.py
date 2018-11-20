#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from randomwalk2d import RandomWalk2D


# ~~~ CREATE 2D RANDOM WALK INSTANCE ~~~ #
rw = RandomWalk2D(n=200)


# ~~~  DRAW REAL RANDOM WALK SIMULATION (STATIC GRAPH OR LIVE ANIMATION) ~~~ ##
topview = rw.random_walk_draw(50,animated=True,projection='2d')
view_3d = rw.random_walk_draw(50,animated=True,projection='3d')



plt.show()
