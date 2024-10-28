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

    if isinstance(matrix, sp.coo_matrix):
        matrix = matrix.todense()

    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    matrix_non_nan = ~np.isnan(matrix)
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
    utility_matrix: sp.csr_matrix,
    user_index,
    neighborhood_size,
    means,
    matrix_non_nan,
    rate_single_item=None,
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
    print(similarities)
    # turn into numpy matrix since filtering and sorting is easier
    similarities = similarities.tocoo()

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if matrix_non_nan[item_index, user_index]:
            return utility_matrix[item_index, user_index] + means[user_index]

        # Find the indices of users who rated the item
        condition = utility_matrix[item_index, :] != 0
        users_who_rated = sp.find(condition)[1]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        filtered_similarities = filter_coo_matrix(similarities, users_who_rated)
        best_among_who_rated = argsort_coo_matrix(filtered_similarities)
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[:neighborhood_size]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated_indices = np.array(
            [user[1] for user in best_among_who_rated]
        )
        similarities_for_best = np.array(
            [user[0] for user in best_among_who_rated]
        ).ravel()
        # Retain only those indices where the similarity is not nan
        # already removed nans from similarities
        if len(best_among_who_rated_indices) > 0:
            # Compute the rating of the item
            print(
                utility_matrix[item_index, best_among_who_rated_indices]
                + means[best_among_who_rated_indices]
            )
            print(similarities_for_best)
            rating_of_item = np.sum(
                similarities_for_best
                @ (
                    utility_matrix[item_index, best_among_who_rated_indices]
                    + means[best_among_who_rated_indices]
                ).reshape(-1, 1)
            ) / np.sum(np.abs(similarities_for_best))
        else:
            rating_of_item = np.nan
        print(
            f"item_idx: {item_index}, neighbors: {best_among_who_rated_indices}, rating: {rating_of_item}"
        )
        return rating_of_item

    num_items = utility_matrix.shape[0]

    if rate_single_item is not None:
        return rate_one_item(rate_single_item)

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings


def rate_one_item_sparse(user_id, movie_id, shelve_prefix):
    """Compute the rating of a single item"""

    import shelve

    with shelve.open(shelve_prefix + "_user_col", flag="r") as user_col:
        if str(user_id) in user_col:
            user_col = user_col[str(user_id)]
        else:
            raise ValueError(f"User {user_id} not found in the user_col shelve")

    relevant_movies = user_col.col

    # Load the movie matrix
    with shelve.open(shelve_prefix + "_rated_by", flag="r") as rated_by:
        if str(movie_id) in rated_by:
            movie_rated_by = rated_by[str(movie_id)]
        else:
            raise ValueError(f"Movie {movie_id} not found in the rated_by shelve")

    total_row, total_col, total_data = [], [], []

    # load relevant user columns
    with shelve.open(shelve_prefix + "_user_col", flag="r") as user_col:
        for user_id in movie_rated_by:
            if str(user_id) in user_col:
                curr_user_col = user_col[str(user_id)]
            else:
                raise ValueError(f"User {user_id} not found in the user_col shelve")
            curr_user_col, curr_user_row, curr_user_data = (
                curr_user_col.col,
                curr_user_col.row,
                curr_user_col.data,
            )
            for i in range(len(curr_user_col)):
                if curr_user_col[i] not in relevant_movies:
                    continue
                total_col.append(user_id)
                total_row.append(curr_user_col[i])
                total_data.append(curr_user_data[i])

    total_matrix = sp.coo_matrix((total_data, (total_row, total_col)))

    assert type(total_matrix) == sp.coo_matrix

    # Compute the rating

    um_matrix, means, matrix_non_nan = center_matrix_and_store_non_nan(total_matrix)

    if means.ndim == 2:
        means = means.T


    return rate_all_items(
        um_matrix, user_id, 4, means, matrix_non_nan, rate_single_item=movie_id
    )


def centered_cosine_sim(u: sp.csr_matrix, v: sp.csr_matrix) -> float:
    return u.dot(v.T) / (sp.linalg.norm(u) * sp.linalg.norm(v))


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
