import numpy as np


LATTICE_SQUARE = 0
LATTICE_HEXAGONAL = 1
LATTICE_TRIANGULAR = 2

DIRECTION_NORTH = 1
DIRECTION_NORTH_EAST = 2
DIRECTION_EAST = 3
DIRECTION_SOUTH_EAST = 4
DIRECTION_SOUTH = 5
DIRECTION_SOUTH_WEST = 6
DIRECTION_WEST = 7
DIRECTION_NORTH_WEST = 8


def nearest_neighbors(x, y, mode=LATTICE_SQUARE, direction=None):
    if mode == LATTICE_SQUARE:
        neighbors = nn_square(x, y)
    elif mode == LATTICE_HEXAGONAL:
        neighbors = nn_hex(x, y)
    elif mode == LATTICE_TRIANGULAR:
        neighbors = nn_trig(x, y)
    else:
        raise NotImplementedError(mode)
    
    if direction is not None:
        neighbors = apply_direction(x, y, neighbors,
                                    direction[0], direction[1], mode)
    
    return neighbors


def nn_square(x, y):
    return np.array([(x - 1, x + 1, x, x), (y, y, y - 1, y + 1)])


def nn_hex(x, y):
    neighbors = np.array([(x - 1, x + 1, x), (y, y, 0)])

    if ((x % 2 == 0) and (y % 2 != 0)) or ((x % 2 != 0) and (y % 2 == 0)):
        neighbors[1, 2] = y - 1
    else:
        neighbors[1, 2] = y + 1
    return neighbors


def nn_trig(x, y):
    if (y % 2 == 0):
        return np.array([(x - 1, x, x, x - 1, x + 1, x - 1),
                         (y, y + 1, y - 1, y + 1, y, y - 1)])
    else:
        return np.array([(x - 1, x, x, x + 1, x + 1, x + 1),
                         (y, y + 1, y - 1, y + 1, y, y - 1)])


def apply_direction(x, y, neighbors, x_dir, y_dir, mode):
    neighbors_tmp = neighbors

    if mode == LATTICE_TRIANGULAR:
        new_xy = convert_lattice(np.array([(x, y)]), mode)
        x = new_xy[0, 0]
        y = new_xy[0, 1]
        neighbors_tmp = (
            convert_lattice(neighbors.transpose(), mode)).transpose()
        
    if x_dir == 1:
        x_ix = neighbors_tmp[0, :] >= x
    elif x_dir == -1:
        x_ix = neighbors_tmp[0, :] <= x
    else:
        x_ix = neighbors_tmp[0, :] == neighbors_tmp[0, :]

    if y_dir == 1:
        y_ix = neighbors_tmp[1, :] >= y
    elif y_dir == -1:
        y_ix = neighbors_tmp[1, :] <= y
    else:
        y_ix = neighbors_tmp[1, :] == neighbors_tmp[1, :]
    
    indexes = np.nonzero(x_ix & y_ix)

    neighbors = neighbors[:, indexes[0]]

    return neighbors


def convert_lattice(xy_list, mode=LATTICE_SQUARE):
    if mode == LATTICE_HEXAGONAL:
        return sq_to_hex(xy_list)
    elif mode == LATTICE_TRIANGULAR:
        return sq_to_trig(xy_list)
    else:
        return xy_list
    raise NotImplementedError(mode)


def sq_to_trig(xy_list):
    y_shift = np.sqrt(3/4)
    new_xy = xy_list.copy().astype(np.float)
    new_xy[:, 1] *= y_shift
    mask = xy_list[:, 1] % 2 != 0
    new_xy[mask, 0] += 0.5
    return new_xy


def sq_to_hex(xy_list):
    scale = np.sqrt(1.0 / 3.0)
    new_xy = xy_list.copy().astype(np.float)
    # mask_same = xy_list[:, 0] % 2 == xy_list[:, 1] % 2
    mask_diff = xy_list[:, 0] % 2 != xy_list[:, 1] % 2
    new_xy[:, 1] *= 3
    new_xy[mask_diff, 1] -= 1
    new_xy[:, 1] *= scale
    return new_xy


def matrix_offset(matrix):
    offset = (np.size(matrix, 0) - 1) // 2
    return offset


def get_value_at_coords(matrix, x, y):
    offset = matrix_offset(matrix)
    return matrix[x + offset, y + offset]


def add_connector(x, y, n_x, n_y):
    if x == n_x:
        if n_y > y:
            return DIRECTION_SOUTH
        else:
            return DIRECTION_NORTH
    elif y == n_y:
        if n_x > x:
            return DIRECTION_WEST
        else:
            return DIRECTION_EAST
    elif n_x > x:
        if n_y > y:
            return DIRECTION_SOUTH_WEST
        else:
            return DIRECTION_NORTH_WEST
    elif n_x < x:
        if n_y > y:
            return DIRECTION_SOUTH_EAST
        else:
            return DIRECTION_NORTH_EAST
    raise NotImplementedError()


def get_connection(x, y, t):
    # TODO: WHY?
    t -= 1
    if t <= 0:
        return (None, None)
    else:
        if t == DIRECTION_NORTH:
            return x, y + 1
        elif t == DIRECTION_NORTH_EAST:
            return x + 1, y + 1
        elif t == DIRECTION_EAST:
            return x + 1, y
        elif t == DIRECTION_SOUTH_EAST:
            return x + 1, y - 1
        elif t == DIRECTION_SOUTH:
            return x, y - 1
        elif t == DIRECTION_SOUTH_WEST:
            return x - 1, y - 1
        elif t == DIRECTION_WEST:
            return x - 1, y
        elif t == DIRECTION_NORTH_WEST:
            return x - 1, y + 1
        else:
            raise NotImplementedError(t)
