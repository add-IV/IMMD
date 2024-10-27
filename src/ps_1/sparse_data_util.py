from .data_util import get_um_by_name as get_um_by_name_data_util
from .sparse_algorithms import center_matrix_and_store_non_nan
import pandas as pd
import numpy as np
import polars as pl
import scipy.sparse as sp



def get_size_in_mb(obj):
    if isinstance(obj, pl.DataFrame):
        return obj.estimated_size(unit='mb')
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    elif isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / (1024 * 1024)
    elif isinstance(obj, sp.csr_matrix):
        return obj.data.nbytes / (1024 * 1024) + obj.indices.nbytes / (1024 * 1024) + obj.indptr.nbytes / (1024 * 1024)
    else:
        return None
    
def get_um_by_name(config, dataset_name):
    um = get_um_by_name_data_util(config, dataset_name)
    return center_matrix_and_store_non_nan(um)


if __name__ == "__main__":
    um_lecture, means, matrix_non_nan = get_um_by_name(None, "lecture_1")
# end main