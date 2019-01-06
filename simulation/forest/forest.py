"""Simulating forest fires."""

import matplotlib.pyplot as plt
import numpy as np

from simulation.forest.lattice import (nearest_neighbors, add_connector,
                                       UNBURNED_TREE, BURNED_TREE, SURVIVED_TREE, matrix_offset)
from simulation.forest.plot import make_plot

RANDOM_SEED = 1


def forest_fire(p, max_fire_size, n_fires, do_plots, parallel=False,
                c_distribution=False, mode=None, site=False, direction=None):
    """Simulate a series of forest fires.

    Arguments:
        p {float} -- Ignition probability.
        max_fire_size {int} -- Maximum size for a fire.
        n_fires {int} -- Number of firest to simulate.
        do_plots {bool} -- Whether to plot the fires.

    Keyword Arguments:
        parallel {bool} -- Run fires in parallel. (default: {False})
        c_distribution {bool} -- Plot cumulative distribution. (default: {False})
        mode {int} -- Lattice type. (default: {None})
        site {bool} -- Use site percolation (each tree can ignite only once.) (default: {False})
        direction {tuple} -- Wind direction (-1|0|1) (default: {None})

    Raises:
        NotImplementedError -- If parallel equals True.

    Returns:
        ndarray -- Cumulative distribution.
        ndarray -- Cumulative distribution indexes.
    """

    if not parallel:
        random_state = np.random.RandomState(RANDOM_SEED)
        fire_sizes = np.zeros((n_fires, 1))
        for k in range(n_fires):
            fire_sizes[k] = burn_forest(p, max_fire_size, direction,
                                        do_plots, mode=mode, site=site,
                                        random_state=random_state)
            print(k + 1)
    else:
        # TODO: Parallelism
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


def burn_forest(p, max_size, direction, do_plot=None, mode=None, site=False,
                random_state=None):
    """Simulate one forest fire.

    Arguments:
        p {float } -- Ignition probability.
        max_size {int} -- Maximum size for the fire.
        direction {tuple} -- Wind direction.

    Keyword Arguments:
        do_plot {bool} -- Whether to plot the fire. (default: {None})
        mode {int} -- Lattice type. (default: {None})
        site {bool} -- Site percolation. (default: {False})
        random_state {RandomState} -- Random state. (default: {None})

    Returns:
        int -- Size of the fire.
    """

    forest_size = 2 * (max_size + 1) + 1

    forest = np.zeros((forest_size, forest_size), dtype=np.int)

    # The fire starts form the middle
    burning_list = np.array([(0, 0),], dtype=np.int)

    forest[max_size + 1, max_size + 1] = BURNED_TREE

    burned_list = np.zeros((0, 2), dtype=np.int)

    if do_plot:
        # Initialize figure
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

    while True:
        # Burning loop
        burning_list, burned_list, forest = burn_step(
            burning_list, burned_list, p, forest, direction, mode=mode,
            site=site, random_state=random_state)

        if (np.size(burning_list, 0) == 0 or
                np.size(burned_list, 0) > max_size):
            # Break the loop when there are no fires, or
            # the maximum number of fires is reached.
            break

    if do_plot:
        make_plot(ax, burning_list, burned_list, forest, mode)

    fire_size = np.size(burned_list, 0)

    return fire_size


def burn_step(burning_list, burned_list, p, forest, direction, mode=None,
              site=False, random_state=None):
    """Simulate one step of forest burning.

    Arguments:
        burning_list {ndarray} -- List of burning trees.
        burned_list {ndarray} -- List of burned trees.
        p {float} -- Ignition probability.
        forest {ndarray} -- Forest matrix.
        direction {tuple} -- Wind direction.

    Keyword Arguments:
        mode {int} -- Lattice type. (default: {None})
        site {bool} -- Site percolation. (default: {False})
        random_state {RandomState} -- Random state. (default: {None})

    Returns:
        ndarray -- Burning trees.
        ndarray -- Burned trees.
        ndarray -- Forest matrix.
    """

    new_burning_list = np.zeros((0, 2), dtype=np.int)

    # Offset so that (0, 0) is at the middle of the forest
    # offset = (np.size(forest, 0) - 1) // 2
    offset = matrix_offset(forest)
    assert isinstance(offset, int)

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
            if (forest[Nx_ix, Ny_ix] == UNBURNED_TREE):
                # There is an unburned tree at (Nx, Ny)
                if random_state is None:
                    random_number = np.random.rand()
                else:
                    random_number = random_state.rand()
                if random_number < p:
                    # The fire spreads to the neighbor
                    new_burning_list = np.vstack((new_burning_list, [Nx, Ny]))
                    forest[Nx_ix, Ny_ix] = BURNED_TREE + add_connector(Tx, Ty, Nx, Ny)
                elif site:
                    # The Neighbor will not be ignited again
                    forest[Nx_ix, Ny_ix] = SURVIVED_TREE

    new_burned_list = np.vstack((burned_list, burning_list))
    return new_burning_list, new_burned_list, forest
