#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from randomwalk2d import RandomWalk2D


# ~~~ CREATE 2D RANDOM WALK INSTANCE ~~~ #
rw = RandomWalk2D(p=0.5,n=1000)


# ~~~  DRAW REAL RANDOM WALK SIMULATION (STATIC GRAPH OR LIVE ANIMATION) ~~~ ##
rw.random_walk_draw(5,animated=True)
