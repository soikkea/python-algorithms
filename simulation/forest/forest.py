import matplotlib.pyplot as plt
import numpy as np

from simulation.forest.lattice import nearest_neighbors, add_connector
from simulation.forest.plot import make_plot


def forest_fire(p, max_fire_size, n_fires, do_plots, parallel=False,
                c_distribution=False, mode=None, site=False, direction=None):
    if not parallel:
        fire_sizes = np.zeros((n_fires, 1))
        for k in range(n_fires):
            fire_sizes[k] = burn_forest(p, max_fire_size, direction,
                                        do_plots, mode=mode, site=site)
            print(k + 1)
    else:
        raise NotImplementedError()
    
    # Indexes for the cumulative distribution function
    fs = np.arange(1, max_fire_size + 1)
    # The cumulative distribution function
    P = np.zeros((max_fire_size))

    # Calculate the cumulative distribution function:
    for k in range(np.size(fs)):
        P[k] = np.sum(fire_sizes <= fs[k]) / float(n_fires)
    
    # Plot the cdf
    if c_distribution:
        fig, ax = plt.subplots()
        ax.plot(fs, P)
        ax.set_ylabel('$P(N_{burned})$')
        ax.set_xlabel('$N_{burned}$')
        ax.set_title('$p = {}$'.format(p))
    
    return P, fs


def burn_forest(p, max_size, direction, do_plot=None, mode=None, site=False):
    forest_size = 2 * (max_size + 1) + 1

    forest = np.zeros((forest_size, forest_size), dtype=np.int)

    # The fire starts form the middle
    burning_list = np.array([(0, 0),], dtype=np.int)

    # TODO: FIX THIS
    forest[max_size + 1, max_size + 1] = 1

    burned_list = np.zeros((0, 2), dtype=np.int)

    # TODO: FIX THIS
    if do_plot > 0:
        # Initialize figure
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

    while True:
        # Burning loop
        burning_list, burned_list, forest = burn_step(
            burning_list, burned_list, p, forest, direction, mode=mode,
            site=site)

        if (np.size(burning_list, 0) == 0 or 
                np.size(burned_list, 0) > max_size):
            # Break the loop when there are no fires, or 
            # the maximum number of fires is reached.
            break
    
    # TODO: FIX ME
    if do_plot >= 1:
        make_plot(ax, burning_list, burned_list, forest, mode)
    
    fire_size = np.size(burned_list, 0)

    return fire_size


def burn_step(burning_list, burned_list, p, forest, direction, mode=None, 
              site=False):
    new_burning_list = np.zeros((0, 2), dtype=np.int)

    # TODO: FIX THIS
    # Offset so that (0, 0) is at the middle of the forest
    offset = (np.size(forest, 0) - 1) // 2
    assert type(offset) is int

    for T in range(np.size(burning_list, 0)):
        # Loop over burning trees
        Tx = burning_list[T, 0]
        Ty = burning_list[T, 1]
        neighbors = nearest_neighbors(Tx, Ty, mode=mode, direction=direction)
        for N in range(np.size(neighbors, 1)):
            # Loop over neighboring trees
            Nx = neighbors[0, N]
            Ny = neighbors[1, N]
            Nx_ix = Nx + offset
            Ny_ix = Ny + offset
            # TODO: FIX FOREST INTEGER MEANINGS
            if (forest[Nx_ix, Ny_ix] == 0):
                # There is an unburned tree at (Nx, Ny)
                # TODO: Seed generator
                if (np.random.rand() < p):
                    # The fire spreads to the neighbor
                    new_burning_list = np.vstack((new_burning_list, [Nx, Ny]))
                    forest[Nx_ix, Ny_ix] = 1 + add_connector(Tx, Ty, Nx, Ny)
                elif site:
                    # The Neighbor will not be ignited again
                    forest[Nx_ix, Ny_ix] = -1
    
    new_burned_list = np.vstack((burned_list, burning_list))
    return new_burning_list, new_burned_list, forest
