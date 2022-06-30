import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output

def metro_ising(T, runtime, plot=False, N=None, grid=None):
    """
    Metropolis Monte-Carlo Markov Chain algorithm
        T: temperature
        grid: if you choose to initialize with previous image
        runtime: iterations
        plot 
    returns array of shape (runtime, N, N)
    based on : https://www.asc.ohio-state.edu/braaten.1/statphys/Ising_MatLab.pdf
    """
    # set constants to 1 so temperature is dimensionless
    J = 1 #strength of interaction (Joules)
    k = 1 # Joules per kelvin (ie entropy?)
    frames = []

    if T == 'critical':
        T = 2.26918531421

    if grid != None:
        N = grid.shape[0]
    else:
        grid = 2*np.random.randint(2, size = (N,N)) - 1 #random initial configuration with 1s and -1s
    
    for t in range(runtime):
        #sum of interactions at each spin site (vectorized!)
        interactions = sum([np.roll(grid, shift=(0,  1), axis=1),
                            np.roll(grid, shift=(0, -1), axis=1),
                            np.roll(grid, shift=(1,  0), axis=0), # have to change axis because unlike MATLAB's circshift, np.roll sees shifts (0,1) == (1,0)
                            np.roll(grid, shift=(-1, 0), axis=0)])
        # Hamiltonian: change in energy of flipping a spin
        DeltaE = 2 * J * (grid * interactions) # element wise multiplication
        # transition probabilities
        p_trans = np.exp(-DeltaE/(k * T)) # according to the Boltzmann distribution. kT is the "characteristic energy"
        # accepting or rejecting spin flips in one fell swoop!
        #    assigning uniformly distributed values to each site, then checking if they are less than transition prob or less than 0.1?
        transitions = (np.random.random((N,N)) < p_trans ) * (np.random.random((N,N)) < 0.1) * -2 + 1
        grid = grid * transitions
        #raster.append(grid.flatten())
        frames.append(grid)
        
        if plot == 'animate':
            plt.figure(figsize=(6,6)) 
            plt.imshow(grid,cmap='gray')
            plt.axis('off')
            clear_output(wait=True)
            plt.show()
    
    ## plot final frame
    if plot == 'static':
        plt.figure(figsize=(6,6)) 
        plt.imshow(grid,cmap='gray')
        plt.axis('off')
        plt.show()
        
    #turning current state into binary to resemble spiking neural network
    frames = np.array(frames)
    #raster_df = pd.DataFrame(data = np.array(raster))
    frames[frames==1] = 0 #play around looking if up or down spin affects this
    frames[frames==-1] = 1
    return frames