from dataclasses import dataclass, field


@dataclass(init=False, frozen=True)
class PathsConfig:
    raw_data_path: str = "data/trx/"  # Raw upstream data.
    preprocessing_data_path: str = "data/intermediate/preprocessing/"
    features_data_path: str = "data/intermediate/features/"

    mean_amount_all_categories_file_name = "_mean_amount_all_categories.pkl"
    std_amount_all_categories_file_name = "std_amount_all_categories.pkl"
    mean_amount_per_category_file_name = "mean_amount_per_category.pkl"
    std_amount_per_category_file_name = "std_amount_per_category.pkl"
    indexer_per_column_file_name_prefix = "indexer_per_column_"
    encoder_per_column_file_name_prefix = "encoder_per_column_"
    column_path_map_file_name = "column_path_map.pkl"


@dataclass(frozen=True)
class _RunConfig:
    """
    This class should not be used directly without instantiation. Always use the run_config in this module.
    """

    spark_master_url: str = "local[2]"
    # The fraction of the entire raw data to be selected at random.
    # For the sake of speed in development, set this to smaller number. In production, always expected to be
    # set to 1.0.
    sample_rate: float = 1.0
    num_folds: int = 5
    threshold_in_eur_to_consider_large_transactions: float = 123.45
    mlflow_tracking_uri: str = "http://mlflow:8080"
    experiment_name: str = "Alpha"
    run_name: str = "Run_234"
    categorical_cols: list = field(default_factory=lambda: ["category", "payment_type"])


@dataclass(
    init=False,
)
class ModelConfig:
    model_name: str = "alpha"
    max_depth: int = 100
    class_weight: str = "balanced"


run_config = _RunConfig()
