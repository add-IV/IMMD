# Frederick Vandermoeten, October 2024

import numpy as np
import scipy.sparse as sp

# Exercise 2 & 3
# Since scipy sparse matrices do not offer great support for nan values, when we convert a numpy array to a sparse matrix,
# it will keep track of all nan values. This defeats the purpose of using a sparse matrix since all our entries are either non-zero or nan.
# Therefore, we will keep track of the nan values in a separate sparse matrix and replace them with 0 in the original matrix after centering.
# (Else a 0 rating after centering would be mistakenly considered as a non-rated item.)
# We also need to keep track of the means of the non-nan values in order to compute the rating of the items.
# This way, we can still use the benefits of sparse matrices while keeping track of the nan values.
# Since this uses two separate sparse matrices, it is only efficient when we have a large number of nan values.


def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None


def center_matrix_and_store_non_nan(matrix, axis=0):
    """Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    matrix_non_nan = np.isnan(matrix)
    return (
        sp.csr_matrix(np.nan_to_num(matrix_centered)),
        means,
        sp.csr_matrix(matrix_non_nan),
    )


def filter_coo_matrix(matrix: sp.coo_matrix, rows):
    """Filter the matrix to keep only the rows in 'rows'"""
    if not isinstance(matrix, sp.coo_matrix):
        raise ValueError("Input matrix must be of type scipy.sparse.coo_matrix")
    # Filter the data in the matrix
    mask = np.isin(matrix.row, rows)
    return sp.coo_matrix(
        (matrix.data[mask], (matrix.row[mask], matrix.col[mask])),
        shape=matrix.shape,
    )


def argsort_coo_matrix(matrix):
    """Sort the matrix in descending order, since this doesn't return a coo matrix but a list of tuples the name is a bit misleading"""
    if not isinstance(matrix, sp.coo_matrix):
        raise ValueError("Input matrix must be of type scipy.sparse.coo_matrix")
    # Sort the data in the matrix
    tuples = zip(matrix.data, matrix.row, matrix.col)
    # Sort the tuples by the data
    tuples = sorted(tuples, key=lambda x: (x[0], x[1]), reverse=True)
    # Reconstruct the matrix
    return tuples


# Implement the CF from the lecture 1
def rate_all_items(
    utility_matrix: sp.csr_matrix, user_index, neighborhood_size, means, matrix_non_nan
):
    print(
        f"\n>>> CF computation for UM w/ shape: "
        + f"{utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n"
    )

    # matrix should already be centered
    """ Compute the rating of all items not yet rated by the user"""
    user_col = utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_centered_cosine_sim(utility_matrix, user_col)
    # turn into coo matrix since filtering and sorting is easier
    similarities = similarities.tocoo()

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not matrix_non_nan[item_index, user_index]:
            return utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        condition = utility_matrix[item_index, :] != 0
        users_who_rated = sp.find(condition)[1]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        filtered_similarities = filter_coo_matrix(similarities, users_who_rated)
        best_among_who_rated = argsort_coo_matrix(filtered_similarities)
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[:neighborhood_size]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated_indices = np.array([user[1] for user in best_among_who_rated])
        # Retain only those indices where the similarity is not nan
        # already removed nans from similarities
        if len(best_among_who_rated_indices) > 0:
            # Compute the rating of the item
            rating_of_item = np.mean(
                utility_matrix[item_index, best_among_who_rated_indices]
                + means[best_among_who_rated_indices]
            )
        else:
            rating_of_item = np.nan
        print(
            f"item_idx: {item_index}, neighbors: {best_among_who_rated_indices}, rating: {rating_of_item}"
        )
        return rating_of_item

    num_items = utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings


def centered_cosine_sim(u: sp.csr_array, v: sp.csr_array) -> float:
    return u @ v / (sp.linalg.norm(u) * sp.linalg.norm(v))


def fast_centered_cosine_sim(
    um: sp.csr_matrix, vector: sp.csr_matrix, axis=0
) -> sp.csr_matrix:
    """Compute the cosine similarity between the matrix and the vector"""
    norms = sp.linalg.norm(um, axis=axis)
    um_normalized = um / norms

    dot = um_normalized.T @ vector
    scaled = dot / sp.linalg.norm(vector)
    return scaled


if __name__ == "__main__":
    um_lecture = [
        [1.0, np.nan, 3.0, np.nan, np.nan, 5.0],
        [np.nan, np.nan, 5.0, 4.0, np.nan, np.nan],
        [2.0, 4.0, np.nan, 1.0, 2.0, np.nan],
        [np.nan, 2.0, 4.0, np.nan, 5.0, np.nan],
        [np.nan, np.nan, 4.0, 3.0, 4.0, 2.0],
        [1.0, np.nan, 3.0, np.nan, 3.0, np.nan],
    ]

    matrix_lecture = np.asarray(um_lecture)
    um_matrix, means, matrix_non_nan = center_matrix_and_store_non_nan(matrix_lecture)

    rate_all_items(um_matrix, 4, 2, means, matrix_non_nan)