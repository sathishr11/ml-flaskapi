from dataclasses import dataclass
from pathlib import Path
from pydoc import ModuleScanner
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    data_source: str
    data_ingestion_dir: Path
    raw_file_dir: Path
    raw_file_name: Path


@dataclass(frozen=True)
class TrainTestSplitConfig:
    raw_file_path: Path
    train_data_dir: Path
    train_file: str
    test_file: str
    test_size: float
    random_state: int


@dataclass(frozen=True)
class DataTransformationConfig:
    transformed_data_dir: Path
    integer_features: List
    float_features: List
    map_function: dict


@dataclass(frozen=True)
class TrainingConfig:
    train_path: Path
    test_path: Path
    regression_output: str
    classification_output: str


@dataclass(frozen=True)
class TrainingParams:
    models: dict
    optuna_trials: int
    mlflow_experiment_name: str
    mlflow_artifacts_path: str
    mlflow_model_registry_name: str
