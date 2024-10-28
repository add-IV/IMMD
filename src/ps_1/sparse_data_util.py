from .sparse_algorithms import center_matrix_and_store_non_nan
import pandas as pd
import numpy as np
import polars as pl
import scipy.sparse as sp
from tqdm import tqdm

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


def get_size_in_mb(obj):
    if isinstance(obj, pl.DataFrame):
        return obj.estimated_size(unit="mb")
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    elif isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / (1024 * 1024)
    elif isinstance(obj, sp.csr_matrix):
        return (
            obj.data.nbytes / (1024 * 1024)
            + obj.indices.nbytes / (1024 * 1024)
            + obj.indptr.nbytes / (1024 * 1024)
        )
    else:
        return None


def print_df_stats(prefix_msg, df):
    print(f"{prefix_msg}, df shape: {df.shape}, size in MB: {get_size_in_mb(df)} ")


def load_and_unzip_dataset(url, path_to_save, unzip_path, force_download=False):
    import requests
    import zipfile
    import io, os

    if not force_download and os.path.exists(unzip_path):
        print(f"Dir '{unzip_path}' already exists, skipping download")
        return

    print(f"Downloading '{url}' to '{path_to_save}'")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path_to_save)
    print(f"Loaded and unzipped '{url}' to '{unzip_path}'")


def read_movielens_file_and_convert_to_um(file_path, max_rows=None):
    """Read a local MovieLens file and return the utility matrix as a pivot table where
    columns=userid, rows=movieid (relabeled) and contents are movie rating (NaN = no rating given).
    Original file has columns: userId,movieId,rating,timestamp
    See https://files.grouplens.org/datasets/movielens/ml-25m-README.html
    """
    print(f"\n### Start reading data from '{file_path}'")
    df = pl.read_csv(
        file_path,
        has_header=True,
        columns=[0, 1, 2],
        new_columns=["userID", "movieID", "rating"],
        n_rows=max_rows,
        schema_overrides={
            "userID": pl.UInt32,
            "movieID": pl.UInt32,
            "rating": pl.Float32(),
        },
    )
    print_df_stats(f"Loaded data from '{file_path}'", df)

    # Convert from long to wide format and then drop column 'movieID'
    print("Pivoting the data")
    util_mat_pl = df.pivot(index="movieID", on="userID", values="rating").drop(
        "movieID"
    )
    print_df_stats(f"Utility matrix", util_mat_pl)

    util_mat_np = np.array(util_mat_pl).astype(np.float32)
    print_df_stats(f"Final utility matrix (numpy array as np.float32)", util_mat_np)
    return util_mat_np


def read_movielens_file_and_convert_to_shelves(
    file_path, shelve_prefix, batch_size, max_rows=None
):
    """Read a local MovieLens file and return the rated_by and user_col as shelves
    Uses streams to avoid loading the entire file into memory
    columns=userid, rows=movieid (relabeled) and contents are movie rating (NaN = no rating given).
    Original file has columns: userId,movieId,rating,timestamp
    See https://files.grouplens.org/datasets/movielens/ml-25m-README.html
    """

    # exploration of the data returns the max_movie_id, as well as that user_id is always increasing
    max_movie_id = 209171
    curr_user_id = -1
    import shelve

    print(f"creating shelves at {shelve_prefix}_*")

    rated_by_shelve_path = f"{shelve_prefix}_rated_by"
    user_col_shelve_path = f"{shelve_prefix}_user_col"

    try:
        with shelve.open(rated_by_shelve_path, flag="r") as user_col:
            pass
        print("Shelve already exists, skipping")
        return rated_by_shelve_path, user_col_shelve_path
    except:
        print("Shelve does not exist, creating new one")

    print(f"\n### Start reading data from '{file_path}'")

    with shelve.open(rated_by_shelve_path, flag="n") as user_col:
        user_col.clear()
    with shelve.open(user_col_shelve_path, flag="n") as user_col:
        user_col.clear()

    total_lines = 25000095
    if max_rows is not None:
        total_lines = min(total_lines, max_rows)

    pbar = tqdm(total=total_lines)

    batch = 0
    load_size = batch_size

    while batch * batch_size < total_lines:
        if batch * batch_size + batch_size > total_lines:
            load_size = total_lines - batch * batch_size
        df = pl.read_csv(
            file_path,
            skip_rows=batch * batch_size,
            n_rows=load_size,
            has_header=True,
            new_columns=["userID", "movieID", "rating"],
            schema_overrides={
                "userID": pl.UInt32,
                "movieID": pl.UInt32,
                "rating": pl.Float32(),
            },
        )
        batch += 1

        user_movie_id_lists = df.group_by("userID", maintain_order=True).agg(
            pl.col("movieID")
        )
        user_rating_lists = df.group_by("userID", maintain_order=True).agg(
            pl.col("rating")
        )

        with shelve.open(user_col_shelve_path, flag="c") as user_col:
            for row in range(user_movie_id_lists.shape[0]):
                user_id = user_movie_id_lists["userID"][row]
                user_movie_id_list = user_movie_id_lists.row(row)[1]
                user_rating_list = user_rating_lists.row(row)[1]
                if str(user_id) in user_col:
                    user_current = user_col[str(user_id)]
                    user_data, user_col, user_row = (
                        user_current.data,
                        user_current.col,
                        user_current.row,
                    )
                    user_data = np.concatenate((user_data, user_rating_list))
                    user_col = np.concatenate((user_col, user_movie_id_list))
                    user_row = np.zeros_like(user_col)
                    user_matrix = sp.coo_matrix(
                        (user_data, (user_row, user_col)), shape=(1, max_movie_id + 1)
                    )
                    user_col[str(user_id)] = user_matrix
                else:
                    user_matrix = sp.coo_matrix(
                        (
                            user_rating_list,
                            (np.zeros_like(user_movie_id_list), user_movie_id_list),
                        ),
                        shape=(1, max_movie_id + 1),
                    )
                    user_col[str(user_id)] = user_matrix

        movie_id_rated_by = df.group_by("movieID", maintain_order=True).agg(
            pl.col("userID")
        )

        with shelve.open(rated_by_shelve_path, flag="c") as user_col:
            for row in movie_id_rated_by.iter_rows():
                print(row)
                movie_id = row[0]
                user_ids = row[1]
                if str(movie_id) in user_col:
                    movie_current = user_col[str(movie_id)]
                    movie_list = movie_current + list(user_ids)
                    user_col[str(movie_id)] = movie_list
                else:
                    user_col[str(movie_id)] = list(user_ids)

        pbar.update(load_size)

    pbar.close()

    return rated_by_shelve_path, user_col_shelve_path


def get_um_by_name(config, dataset_name):
    if dataset_name == "movielens":
        # Load (part of) the MovieLens 25M data
        # See https://grouplens.org/datasets/movielens/25m/
        load_and_unzip_dataset(
            config.dowload_url, config.download_dir, config.unzipped_dir
        )
        um = read_movielens_file_and_convert_to_um(
            config.file_path, max_rows=config.max_rows
        )
    elif dataset_name == "lecture_1":
        um_lecture = [
            [1.0, np.nan, 3.0, np.nan, np.nan, 5.0],
            [np.nan, np.nan, 5.0, 4.0, np.nan, np.nan],
            [2.0, 4.0, np.nan, 1.0, 2.0, np.nan],
            [np.nan, 2.0, 4.0, np.nan, 5.0, np.nan],
            [np.nan, np.nan, 4.0, 3.0, 4.0, 2.0],
            [1.0, np.nan, 3.0, np.nan, 3.0, np.nan],
        ]
        um = np.asarray(um_lecture)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return center_matrix_and_store_non_nan(um)


def create_shelves_for_dataset(config, dataset_name):
    if dataset_name == "movielens":
        # Load (part of) the MovieLens 25M data
        # See https://grouplens.org/datasets/movielens/25m/
        load_and_unzip_dataset(
            config.dowload_url, config.download_dir, config.unzipped_dir
        )
        rated_by_shelve_path, user_col_shelve_path = (
            read_movielens_file_and_convert_to_shelves(
                config.file_path,
                config.shelve_path,
                config.batch_size,
                max_rows=config.max_rows,
            )
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return rated_by_shelve_path, user_col_shelve_path


if __name__ == "__main__":
    rated_by_shelve_path, user_col_shelve_path = create_shelves_for_dataset(
        ConfigSparseCf, "movielens"
    )

    import shelve

    with shelve.open(rated_by_shelve_path, flag="r") as rated_by:
        print(type(rated_by["1"]), len(rated_by["1"]))
    with shelve.open(user_col_shelve_path, flag="r") as user_col:
        print(type(user_col["1"]), user_col["1"].shape)

# end main
