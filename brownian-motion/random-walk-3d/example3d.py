#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from randomwalk3d import RandomWalk3D


# ~~~ CREATE 3D RANDOM WALK INSTANCE ~~~ #
rw = RandomWalk3D(n=10000)


# ~~~  DRAW REAL RANDOM WALK SIMULATION (STATIC GRAPH OR LIVE ANIMATION) ~~~ ##
rw.random_walk_draw(50,animated=False)



plt.show()
