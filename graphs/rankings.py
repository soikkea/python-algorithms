"""Algorithms ranking nodes in a graph."""

import numpy as np
import pandas as pd


def hubs_and_authorities(matrix, limit=25, disp_iterations=False):
    matrix_values = matrix.values
    rows, columns = matrix_values.shape

    # TODO: Error handling
    assert rows == columns
    # Initialize values
    hubs = np.ones((rows, 1)) * (1.0 / np.sqrt(rows))
    authorities = np.ones((rows, 1)) * (1.0 / np.sqrt(rows))

    for counter in range(limit):

        h_old = hubs.copy()
        a_old = authorities.copy()

        for s in range(rows):
            a_mask = matrix_values[s, :] > 0
            h_mask = matrix_values[:, s] > 0
            hubs[s] = authorities[a_mask].sum()
            authorities[s] = hubs[h_mask].sum()
        
        hubs = hubs / np.linalg.norm(hubs)
        authorities = authorities / np.linalg.norm(authorities)

        if (counter == limit-1) or disp_iterations:
            print(f"Iteration: {counter+1}")
            print("Change: {}, {}".format(
                np.sqrt(np.sum((authorities - a_old) ** 2)),
                np.sqrt(np.sum((hubs - h_old) ** 2))
            ))
    
    df_out = pd.DataFrame(index=matrix.index)
    df_out['Hubs'] = hubs
    df_out['Authorities'] = authorities
    df_out['Initial'] = 1.0 / np.sqrt(rows)

    return df_out
