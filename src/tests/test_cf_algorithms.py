import pytest
import numpy as np
import scipy.sparse as sp

from ps_1.cf_algorithms import fast_cosine_sim, cosine_sim, center_and_nan_to_zero
from ps_1.data_util import get_um_by_name
from ps_1.sparse_algorithms import (
    centered_cosine_sim,
    fast_centered_cosine_sim,
)


def test_setup():
    assert 1 == 1


class TestExercise2:
    def test_fast_cosine_sim(self):
        # Test the fast_cosine_sim function
        utility_matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        vector = np.array([1, 0, 1])
        result = fast_cosine_sim(utility_matrix, vector)
        assert np.allclose(result, [1, 0.5, 0.5])

    def test_fast_cosine_sym_with_normal_cosine_sim(self):
        # Test the fast_cosine_sim function with normal cosine_sim function
        utility_matrix = np.random.rand(5, 5)
        vector = np.random.rand(5)
        result = fast_cosine_sim(utility_matrix, vector)
        result2 = np.array(
            [
                cosine_sim(vector, utility_matrix[:, i])
                for i in range(utility_matrix.shape[1])
            ]
        )
        assert np.allclose(result, result2)

    def test_centered_cosine_sim(self):
        k = 100
        vector_x = np.arange(1, k + 1, dtype=float)
        vector_x = center_and_nan_to_zero(vector_x)

        vector_y = vector_x[::-1]

        for i in range(k):
            j = k - 1 - i
            assert vector_x[i] == vector_y[j]
        sparse_x = sp.csr_matrix(vector_x)
        sparse_y = sp.csr_matrix(vector_y)
        cosine_sim = centered_cosine_sim(sparse_x, sparse_y)
        numpy_cosine_sim = np.dot(vector_x, vector_y) / (
            np.linalg.norm(vector_x) * np.linalg.norm(vector_y)
        )
        assert np.allclose(cosine_sim, numpy_cosine_sim)

    def test_centered_cosine_sim_sparse(self):
        k = 100
        vector_x = np.arange(1, k + 1, dtype=float)
        c = [2, 3, 4, 5, 6]
        for i, x in enumerate(vector_x):
            if x % 10 in c:
                vector_x[i] = np.nan
        vector_x = center_and_nan_to_zero(vector_x)
        vector_y = vector_x[::-1]
        for i in range(k):
            j = k - 1 - i
            if np.isnan(vector_x[i]):
                assert np.isnan(vector_y[j])
            else:
                assert vector_x[i] == vector_y[j]

        sparse_x = sp.csr_matrix(vector_x)
        sparse_y = sp.csr_matrix(vector_y)

        assert sparse_x.nnz == 50
        assert sparse_y.nnz == 50

        cosine_sim = centered_cosine_sim(sparse_x, sparse_y)
        numpy_cosine_sim = np.dot(vector_x, vector_y) / (
            np.linalg.norm(vector_x) * np.linalg.norm(vector_y)
        )
        assert np.allclose(cosine_sim, numpy_cosine_sim)

    def test_centered_fast_cosine_sim(self):
        um_lecture = get_um_by_name(None, "lecture_1")
        clean_matrix = center_and_nan_to_zero(um_lecture)
        user_col = clean_matrix[:, 4]

        similarities_numpy = fast_cosine_sim(clean_matrix, user_col)

        clean_matrix = sp.csr_matrix(clean_matrix)
        user_col = sp.csr_matrix(user_col).T

        assert clean_matrix.nnz == 19
        assert user_col.nnz == 4

        similarities = fast_centered_cosine_sim(clean_matrix, user_col)

        similarities = similarities.todense().reshape(-1)

        assert np.allclose(similarities, similarities_numpy)
