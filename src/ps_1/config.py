import dataclasses


# Configurations for collaborative filtering
@dataclasses.dataclass
class ConfigCf:
    max_rows: int = int(2.5e6)
    download_dir: str = "/scratch/core/artur/movielens/"
    unzipped_dir: str = download_dir + "ml-25m/"
    dowload_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    file_path: str = download_dir + "ml-25m/ratings.csv"


# Configurations for latent factor models
@dataclasses.dataclass
class ConfigLf:
    # DS details here: https://www.tensorflow.org/datasets/catalog/movielens
    # This dataset has only split 'train'. Good explanation of the split arg:
    # https://stackabuse.com/split-train-test-and-validation-sets-with-tensorflow-datasets-tfds/

    # dataset_base_name: str = 'movielens/25m'
    # dataset_split = 'train[:1%]'   # Use only a part the data for debugging
    dataset_base_name: str = 'movielens/100k'
    dataset_split = 'train'
    split_ratios = (0.8, 0.1, 0.1)  # train, test, validation

    shuffle_files: bool = True
    data_dir: str = '/scratch/core/artur/movielens/'

    # Configurations for matrix factorization
    rng_seed: int = 42
    num_factors = 1
    num_epochs: int = 2
    learning_rate: float = 0.01
    reg_param: float = 0.1
    batch_size: int = 1
    batch_size_predict_and_cmp: int = 1
    num_predictions_to_show: int = 10
