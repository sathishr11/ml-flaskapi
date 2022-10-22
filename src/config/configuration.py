from src.utils import logger
from src.utils import Helper
from src.entity import DataIngestionConfig, TrainingParams
from src.entity import TrainTestSplitConfig
from src.entity import DataTransformationConfig
from src.entity import TrainingConfig
from src.utils import Singleton
from pathlib import Path
import os


class ConfigurationManager(Singleton):
    def __init__(self):
        self.helper = Helper()
        self.helper.load_args()
        self.config = self.helper.read_yaml(self.helper.config_path)
        self.helper.create_directories([self.config.artifacts_root], logger)
        self.params = self.helper.read_yaml(self.helper.params_path)
        self.data_ingestion_config = None
        self.train_test_split_config = None
        self.data_transformation_config = None
        self.training_config = None
        self.regression_params = None
        self.classification_params = None

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        self.data_ingestion_config = DataIngestionConfig(
            data_source=config.data_source,
            data_ingestion_dir=Path(
                os.path.join(self.config.artifacts_root, config.data_ingestion_dir)
            ),
            raw_file_dir=Path(
                os.path.join(
                    self.config.artifacts_root,
                    config.data_ingestion_dir,
                    config.raw_file_dir,
                )
            ),
            raw_file_name=config.raw_file_name,
        )

        return self.data_ingestion_config

    def get_train_split_config(self) -> TrainTestSplitConfig:
        data_ingestion_config = self.get_data_ingestion_config()
        train_test_split_config = self.config.train_test_split

        self.train_test_split_config = TrainTestSplitConfig(
            raw_file_path=Path(
                os.path.join(
                    data_ingestion_config.raw_file_dir,
                    data_ingestion_config.raw_file_name,
                )
            ),
            train_data_dir=Path(
                os.path.join(
                    self.config.artifacts_root, train_test_split_config.train_data_dir
                )
            ),
            train_file=train_test_split_config.train_file,
            test_file=train_test_split_config.test_file,
            test_size=self.params.split_data.test_size,
            random_state=self.params.split_data.random_state,
        )

        return self.train_test_split_config

    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation

        self.data_transformation_config = DataTransformationConfig(
            transformed_data_dir=Path(
                os.path.join(self.config.artifacts_root, config.data_transformation_dir)
            ),
            integer_features=config.integer_features,
            float_features=config.float_features,
            map_function=config.map_function,
        )

        return self.data_transformation_config

    def get_training_config(self) -> TrainingConfig:

        config = self.config.training

        self.training_config = TrainingConfig(
            train_path=Path(
                os.path.join(
                    self.config.artifacts_root,
                    self.config.data_transformation.data_transformation_dir,
                    self.config.train_test_split.train_file,
                )
            ),
            test_path=Path(
                os.path.join(
                    self.config.artifacts_root,
                    self.config.data_transformation.data_transformation_dir,
                    self.config.train_test_split.test_file,
                )
            ),
            regression_output=config.regression_output,
            classification_output=config.classification_output,
        )
        return self.training_config

    def get_regression_params(self) -> TrainingParams:
        self.regression_params = TrainingParams(
            models=self.params.regression_models,
            optuna_trials=self.params.optuna.no_of_trails,
            mlflow_experiment_name=self.params.ml_flow_reg.experiment_name,
            mlflow_artifacts_path=self.params.ml_flow_reg.artifacts_path,
            mlflow_model_registry_name=self.params.ml_flow_reg.model_registry_name,
        )
        return self.regression_params

    def get_classification_params(self) -> TrainingParams:
        self.classification_params = TrainingParams(
            models=self.params.classification_models,
            optuna_trials=self.params.optuna.no_of_trails,
            mlflow_experiment_name=self.params.ml_flow_cls.experiment_name,
            mlflow_artifacts_path=self.params.ml_flow_cls.artifacts_path,
            mlflow_model_registry_name=self.params.ml_flow_cls.model_registry_name,
        )
        return self.classification_params
