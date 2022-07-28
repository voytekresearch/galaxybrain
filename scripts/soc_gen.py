import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append('../')

"""
Simulate SOC sandpile
From https://rosettacode.org/wiki/Abelian_sandpile_model#Python
TODO: saving spiking indices more efficient
TODO (Theory):
- adding grains at time intervals and random points, like external or internal inputs
- allow sand to overflow (get continuous oscillations that way)
    - Vs allowing resonance on walls (what does this say about inputs/outputs?)
- Or more simply what about just having sources and sinks
    - Also random initial conditions as in BTW
- What about refractory states?

"""

class AbelianSandpile:
    def __init__(self, size, ic, snn = False):
        """
        Initialize initial grid size (width) and initial starting amount ic

        If snn is True, simulates sandpile as a spiking neural network, only saving a binary representation
        of the grid (where threshold is a spike)

        if ic is a list, initializes with random initial conditions with parameters:
        [%percent of grid initiated, number of grains at each burst, variability in drop size: if 1 all are same size]
        """
        self.size = size
        self.grid = np.zeros((size,size), int)

        if type(ic) in [list, tuple]: #still have to implement var in drop size
            pcnt_on = ic[0]
            num_rand = int(size*size*pcnt_on)
            self.rand_loc = self.gen_rand_coords(num_rand)
            for coord in self.rand_loc:
                x, y = coord[0], coord[1]
                self.grid[x][y] = ic[1]

        else:
            i1, i2 = int(size/2 - 1), int(size/2)
            self.grid[i1:i2, i1:i2] = ic #so it starts somewhere near the center

        self.raster = []
        self.snn = snn

    def gen_rand_coords(self, num_rand):
        """helper function to generate random coordinates"""
        rand_loc = []
        def recursive_gen():
            for i in np.arange(num_rand):
                rand_coord = list(np.random.randint(0,self.size,2))
                if len(rand_loc) == num_rand:
                    break
                if rand_coord not in rand_loc:
                    rand_loc.append(rand_coord)
                else:
                    recursive_gen()
        recursive_gen()
        return rand_loc

    def iterate(self, grid, save_evol):
        changed = False
        for i, row in enumerate(grid):
            for j, height in enumerate(row):
                if height > 3:
                    grid[i,j] -= 4

                    if i > 0:
                        grid[i - 1, j] += 1
                    if i < len(grid)-1:
                        grid[i + 1, j] += 1
                    if j > 0:
                        grid[i, j - 1] += 1
                    if j < len(grid)-1:
                        grid[i, j + 1] += 1

                    changed = True

        if self.snn: #maybe this should be in simulate
            copy = grid.copy()
            copy[copy < 4] = 0
            copy[copy >= 4] = 1

            self.raster.append(copy.flatten())

        elif save_evol and not self.snn:
            copy = grid.copy()
            self.raster.append(copy)
        return grid, changed

    def simulate(self, save_evol):
        grid = self.grid.copy()
        t = 0
        while True:
            grid, changed = self.iterate(grid, save_evol)
            if not changed:
                return grid

    def run(self, save_evol = False):
        """save_evol saves each time point.  if snn, does this automatically"""
        if self.snn:
            save_evol = True
        final_grid = self.simulate(save_evol)
        plt.figure(figsize=(12,8))
        plt.subplot(1,2,1)
        plt.gray()
        plt.imshow(self.grid)
        plt.subplot(1,2,2)
        plt.gray()
        plt.imshow(final_grid)
        plt.show()

        if save_evol or self.snn:
            return self.raster
        else:
            return final_grid

hold = True #prompt bool
while hold:
    amount = int(input('Enter amount: ')); percent = float(input('Enter % initialized: ')); burst_size = int(input('Enter burst_size: '));
    ans = input(str(str(int(amount*amount*percent*burst_size))+ ' grains. Run sandpile?'))
    if ans.lower() == 'yes':
        hold = False
        spile = AbelianSandpile(amount,ic = [percent, burst_size], snn = True)
        raster = spile.run()
    else:
        continue

np.save('../data/experiments/sim/SOC_spikes', raster)