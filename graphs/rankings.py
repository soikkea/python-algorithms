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


def page_rank(matrix, epsilon=0.001, limit=50):
    matrix_values = matrix.values
    rows, columns = matrix_values.shape

    ranks = np.zeros((rows, columns))

    for s in range(rows):
        for t in range(columns):
            t_row_sum = matrix_values[t, :].sum()
            # If t is connected to s:
            if matrix_values[t, s] == 1:
                ranks[s, t] = epsilon / float(rows)
                if not np.isclose(t_row_sum, 0):
                    ranks[s, t] = (1.0 - epsilon) / t_row_sum
            elif np.isclose(t_row_sum, 0):
                ranks[s, t] = 1.0 / float(rows)
            else:
                ranks[s, t] = epsilon / float(rows)

    relevance = np.ones((rows, 1)) * (1.0 / float(rows))
    old_relevance = relevance.copy()

    final_ranks = np.zeros((rows, 1))

    for i in range(limit):
        for x in range(rows):
            row_count = np.sum(ranks[x, :] * old_relevance.ravel())
            relevance[x] = row_count
            if i == limit - 1:
                final_ranks[x] = relevance[x]
        old_relevance = relevance.copy()
    
    df_out = pd.DataFrame(final_ranks, index=matrix.index, columns=['Rank'])

    return df_out
