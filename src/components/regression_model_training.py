import logging
import pandas as pd
from src.utils import Helper
import importlib
import os
import sys
from src.utils.logger import logger
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.entity import TrainingConfig
from src.entity import TrainingParams
from src.exception import CustomException
from dotenv import load_dotenv
import mlflow
from mlflow.tracking.client import MlflowClient
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus


class TrainModel():
    def __init__(self, config: TrainingConfig, params: TrainingParams, logger: logging.Logger=logger):
        self.logger = logger
        self.helper = Helper()
        self.config = config
        self.params = params
        self.output_column = self.config.regression_output
        self.max_score = 0
        self.best_model = None
        load_dotenv()
        mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
        self.mlflow_experiment_name = self.params.mlflow_experiment_name
        self.mlflow_artifacts_path = self.params.mlflow_artifacts_path
        self.mlflow_model_registry_name = self.params.mlflow_model_registry_name
        self.optimize_direction = 'minimize'
        self.optimize_scoring ='neg_root_mean_squared_error'

    def _load_data(self):
        try:
            self.train_data = pd.read_csv(self.config.train_path)
            self.test_data = pd.read_csv(self.config.test_path)
        except Exception as e:
            self.logger.error(e)
            raise CustomException(e, sys) from e

    def _wait_model_transition(self, model_version: str, stage: str = "None"):
        client = MlflowClient()
        for _ in range(10):
            model_version_details = client.get_model_version(name=self.mlflow_model_registry_name,
                                                         version=model_version,
                                                         )
            status = ModelVersionStatus.from_string(model_version_details.status)
            print("Model status: %s" % ModelVersionStatus.to_string(status))
            if status == ModelVersionStatus.READY:
                client.transition_model_version_stage(
                name=self.mlflow_model_registry_name,
                version=model_version,
                stage=stage,
                )
                break   
            time.sleep(1)

    @staticmethod
    def _evaluate(model, X_test, y_test):

        y_pred = model.predict(X_test)
        n = X_test.shape[0]
        features = X_test.shape[1]
         
        adj_r2 = (1 - ((1-r2_score(y_test, y_pred))*(n-1)/(n-features-1)))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        return {
            "Test adjusted r score": adj_r2, 
            "Test r2 score": r2,
            "Test MAE: ": mae, 
            "Test MSE: ": mse, 
            "Test RMSE: ": rmse
        }
    
    def train_model(self):
        self._load_data()
        try:


            X_train = self.train_data.drop(columns=[self.output_column])
            y_train = self.train_data[self.output_column]

            X_test = self.test_data.drop(columns=[self.output_column])
            y_test = self.test_data[self.output_column]

            # Set up mlflow experiment
            mlflow.set_experiment(self.params.mlflow_experiment_name)

            client = MlflowClient()
            artifact_path = self.mlflow_experiment_name

            for model in self.params.models:

                with mlflow.start_run() as run:
                    run_num = run.info.run_id
                    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_num, artifact_path=artifact_path)
                    model_params = (self.params.models[model])
                    self.logger.info('------------')
                    self.logger.info(model)
                    self.logger.info('------------')
                    study = optuna.create_study(direction=self.optimize_direction,sampler=TPESampler())
                    study.optimize(
                        lambda trial:TrainModel.objective(
                            trial, model, model_params, X_train, y_train, scoring=self.optimize_scoring
                        ),
                        n_trials= self.params.optuna_trials,
                        show_progress_bar = True

                    )
                    self.logger.info(f' Best parameters for {model} is : {str(study.best_params)}')
                    best_params = study.best_params

                    ml_model = TrainModel._prepare_model(
                        module_name=self.params.models[model].module,
                        class_name=model,
                        params=best_params
                    )
                    # ml_model = eval(model)(**best_params)

                    ml_model.fit(X_train, y_train)

                    evaluation_metrics = self._evaluate(
                        ml_model,
                        X_test,
                        y_test
                    )
                    metrics_dict = {"Train score": study.best_value, **evaluation_metrics}
                    # print(metrics_dict)
                    mlflow.log_params(study.best_params)
                    print('------')
                    for key, val in metrics_dict.items():
                        print({key:val})
                        mlflow.log_metric(
                            key=key, value=val
                        )

                    mlflow.sklearn.log_model(
                        ml_model,
                        artifact_path,
                        registered_model_name=self.mlflow_model_registry_name
                    )

                # Grab this latest model version
                model_version_infos = client.search_model_versions("name = '%s'" % self.mlflow_model_registry_name)
                new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
                print(type(new_model_version))
                self._wait_model_transition(new_model_version)

                # Add a description
                client.update_model_version(
                name=self.mlflow_model_registry_name,
                version=new_model_version,
                description=model
                )            
        except Exception as e:
            self.logger.error(e)
            raise CustomException(e, sys) from e

    @staticmethod
    def _prepare_model(module_name, class_name, params):
        module = importlib.import_module(module_name)
        class_ref = getattr(module, class_name)
        model = class_ref(**params)
        return model
    

    @staticmethod
    def objective(trial, model_name, model_params, X_train, y_train, scoring):
        params = {}
        for suggest_type in model_params.params:
            if suggest_type == 'int':
                for arg in model_params.params[suggest_type]:
                    params[arg] = trial.suggest_int(arg, *model_params.params[suggest_type][arg])
            elif suggest_type == 'float':
                for arg in model_params.params[suggest_type]:
                    params[arg] = trial.suggest_float(arg, *model_params.params[suggest_type][arg])
            elif suggest_type == 'float_log':
                for arg in model_params.params[suggest_type]:
                    params[arg] = trial.suggest_float(arg, *model_params.params[suggest_type][arg], log=True)
            elif suggest_type == 'categorical':
                for arg in model_params.params[suggest_type]:
                    params[arg] = trial.suggest_categorical(arg, model_params.params[suggest_type][arg])

        model = TrainModel._prepare_model(
            module_name=model_params.module,
            class_name=model_name,
            params=params
        )
        metrics = -np.mean(cross_val_score(model,X_train,y_train, cv = 4, n_jobs =-1,scoring=scoring))
        return metrics
