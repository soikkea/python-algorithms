import numpy as np
import simulation.forest.lattice as lattice


# Size of the markers in the plotted figures
MARKER_SIZE = 10

def make_plot(ax, burning_list, burned_list, forest, mode):
    plot_forest(burning_list, burned_list, ax, mode)
    plot_bonds(burning_list, burned_list, forest, ax, mode)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_xlim(xmin - 1, xmax + 1)


def plot_forest(burning_list, burned_list, ax, mode=lattice.LATTICE_SQUARE):
    burning = lattice.convert_lattice(burning_list, mode)
    burned = lattice.convert_lattice(burned_list, mode)
    ax.plot(burning[:, 0], burning[:, 1], 'r^', ms=MARKER_SIZE)
    ax.plot(burned[:, 0], burned[:, 1], 'b^', ms=MARKER_SIZE)
    return


def plot_bonds(burning_list, burned_list, forest, ax,
               mode=lattice.LATTICE_SQUARE):
    full_list = np.vstack((burning_list, burned_list))
    for i in range(np.size(full_list, 0)):
        x = full_list[i, 0]
        y = full_list[i, 1]
        xx, yy = lattice.get_connection(x, y,
            lattice.get_value_at_coords(forest, x, y))
        if xx:
            xys = lattice.convert_lattice(np.array([(x, y), (xx, yy)]), mode)
            ax.plot(xys[:, 0], xys[:, 1], 'g')
    return
