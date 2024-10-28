import pytest
import numpy as np
import scipy.sparse as sp
import logging

from ps_1.cf_algorithms import fast_cosine_sim, cosine_sim, center_and_nan_to_zero
from ps_1.cf_algorithms import rate_all_items as rate_all_items_non_sparse
from ps_1.data_util import get_um_by_name as get_um_by_name_data_util
from ps_1.sparse_data_util import get_um_by_name, get_size_in_mb, create_shelves_for_dataset
from ps_1.sparse_algorithms import (
    centered_cosine_sim,
    fast_centered_cosine_sim,
    rate_all_items,
    rate_one_item_sparse
)

LOGGER = logging.getLogger(__name__)


def test_setup():
    assert 1 == 1

import dataclasses
# Configurations for collaborative filtering
@dataclasses.dataclass
class ConfigSparseCf:
    max_rows: int = int(2.5e6)
    download_dir: str = "./movielens/"
    unzipped_dir: str = download_dir + "ml-25m/"
    dowload_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    file_path: str = download_dir + "ml-25m/ratings.csv"
    shelve_path: str = download_dir + "ml-25m/shelve"
    batch_size: int = 1000000


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

        sparse_x = sp.csr_matrix(np.asmatrix(vector_x))
        sparse_y = sp.csr_matrix(np.asmatrix(vector_y))
        cosine_sim = centered_cosine_sim(sparse_x, sparse_y).todense()
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

        sparse_x = sp.csr_matrix(np.asmatrix(vector_x))
        sparse_y = sp.csr_matrix(np.asmatrix(vector_y))

        assert sparse_x.nnz == 50
        assert sparse_y.nnz == 50

        cosine_sim = centered_cosine_sim(sparse_x, sparse_y).todense()
        numpy_cosine_sim = np.dot(vector_x, vector_y) / (
            np.linalg.norm(vector_x) * np.linalg.norm(vector_y)
        )
        assert np.allclose(cosine_sim, numpy_cosine_sim)

    def test_centered_fast_cosine_sim(self):
        um_lecture = get_um_by_name_data_util(None, "lecture_1")
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


class TestExercise3:
    def test_shrinking_um(self):
        um_lecture, means, matrix_non_nan = get_um_by_name(None, "lecture_1")
        assert um_lecture.shape == (6, 6)
        assert means.shape == (6,)
        assert matrix_non_nan.shape == (6, 6)
        assert um_lecture.nnz == 19
        assert matrix_non_nan.nnz == 19

    def test_big_matrix_size_improvements(self):
        # test works but is slow, sparse version has less than 1 MB
        # return
        import dataclasses

        @dataclasses.dataclass
        class config:
            max_rows: int = int(1e5)
            dowload_url: str = (
                "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
            )
            download_dir: str = "./movielens"
            unzipped_dir: str = download_dir + "/ml-25m/"
            file_path: str = download_dir + "/ml-25m/ratings.csv"

        um_traditional = get_um_by_name_data_util(config, "movielens")
        um_lecture, means, matrix_non_nan = get_um_by_name(config, "movielens")

        size_lecture = get_size_in_mb(um_lecture)
        size_means = get_size_in_mb(means)
        size_matrix_non_nan = get_size_in_mb(matrix_non_nan)

        size_traditional = get_size_in_mb(um_traditional)

        LOGGER.info("Size Non Sparse: %.2f MB", size_traditional)
        LOGGER.info("Size Sparse: %.2f MB", size_lecture + size_means + size_matrix_non_nan)

        assert size_lecture < size_traditional
        assert size_lecture + size_means + size_matrix_non_nan < size_traditional

    def test_sparse_ratings(self):
        um_lecture, means, matrix_non_nan = get_um_by_name(None, "lecture_1")
        user_index = 4
        neighborhood_size = 2
        ratings = rate_all_items(
            um_lecture, user_index, neighborhood_size, means, matrix_non_nan
        )

        assert len(ratings) == um_lecture.shape[0]

        non_sparse_matrix = get_um_by_name_data_util(None, "lecture_1")


        ratings2 = rate_all_items_non_sparse(non_sparse_matrix, user_index, neighborhood_size)

        assert np.allclose(ratings, ratings2)


class TestExercise5:
    def test_first_user(self):
        user_id = 828
        movie_id = 11

        config = ConfigSparseCf()

        rated_by_shelve_path, user_col_shelve_path = create_shelves_for_dataset(
            config, "movielens"
        )

        rating = rate_one_item_sparse(user_id, movie_id, config.shelve_path)

        assert  rating == 4.0
