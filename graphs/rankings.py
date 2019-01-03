"""Algorithms ranking nodes in a graph."""

import numpy as np
import pandas as pd


def hubs_and_authorities(matrix, limit=25, disp_iterations=False):
    """Hubs and authorities algorithm.
    Also known as Hyperlink-Induced Topic Search.

    Gives each node two scores:
    Hub score is based on how many other nodes the node links to. It is
    calculated as a sum of authority scores of nodes the node points to.
    Authority score is based on how many other nodes link to the node. It is
    calculated as a sum of hub scores of nodes pointing to the node.

    After each iteration the scores are normalized.

    Full description at:
    https://en.wikipedia.org/wiki/HITS_algorithm

    Arguments:
        matrix {DataFrame} -- Pandas dataframe containing square adjacency matrix.

    Keyword Arguments:
        limit {int} -- Number of iterations. (default: {25})
        disp_iterations {bool} -- Print status every iteration. (default: {False})

    Returns:
        DataFrame -- Results.
    """

    matrix_values = matrix.values
    rows, columns = matrix_values.shape

    assert rows == columns, ("Number of rows and columns in matrix does not "
                             "match")

    # Initialize values
    init_value = 1.0 / np.sqrt(rows)
    hubs = np.ones((rows, 1)) * (init_value)
    authorities = np.ones((rows, 1)) * (init_value)

    # Main iteration loop
    for counter in range(limit):

        h_old = hubs.copy()
        a_old = authorities.copy()

        for node in range(rows):
            # Nodes the current node points to
            a_mask = matrix_values[node, :] > 0
            # Nodes pointing to the current node
            h_mask = matrix_values[:, node] > 0

            hubs[node] = authorities[a_mask].sum()
            authorities[node] = hubs[h_mask].sum()

        # Normalize scores
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
    df_out['Initial'] = init_value

    return df_out


def page_rank(matrix, epsilon=0.001, limit=50):
    """PageRank algorithm.

    Ranks nodes in a manner that gives high score to nodes which are linked to
    by other nodes with high scores.
    Final ranks are equivalent to probability distribution of random traveller
    arraiving to specific node.

    The algorithm is implemented as follows:
    First, the matrix F is calculated. Its elements are defined as:
    F_st = (1 - eps) / deg(t) + eps / N, if t links to s,
         = 1 / N                       , if t doesn't link anywhere,
         = eps / N                     , otherwise.
    Here, deg(t) is the number of links from t, N is the total number of nodes
    and eps is some small constant.
    Second, the final ranks r_n are computed iteratively r_i = F * r_i-1 until
    iteration limit i=n is reached.

    Full description at:
    https://en.wikipedia.org/wiki/PageRank

    Arguments:
        matrix {DataFrame} -- Pandas dataframe containing square adjacency matrix.

    Keyword Arguments:
        epsilon {float} -- Constant value. (default: {0.001})
        limit {int} -- Number of iterations. (default: {50})

    Returns:
        DataFrame -- Results.
    """

    matrix_values = matrix.values
    rows, columns = matrix_values.shape

    ranks = np.zeros((rows, columns))

    for s in range(rows):
        for t in range(columns):
            t_n_links = matrix_values[t, :].sum()
            # If t is connected to s:
            if matrix_values[t, s] == 1:
                ranks[s, t] = epsilon / float(rows)
                assert not np.isclose(t_n_links, 0)
                ranks[s, t] += (1.0 - epsilon) / t_n_links
            elif np.isclose(t_n_links, 0):
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
