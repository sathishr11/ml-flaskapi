import logging
from src.utils import Helper
from src.utils import logger
import os
import pandas as pd
import optuna
# from optuna import Trial, visualization
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, classification_report
from src.components.regression_model_training import TrainModel
from src.entity import TrainingConfig
from src.entity import TrainingParams
from src.exception import CustomException
import mlflow
from mlflow.tracking.client import MlflowClient
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import sys


class TrainClassificationModel(TrainModel):
    def __init__(self, config: TrainingConfig, params: TrainingParams, logger: logging.Logger=logger):
        super().__init__(config, params, logger)
        self.output_column = self.config.classification_output
        self.optimize_direction = 'maximize'
        self.optimize_scoring = 'recall'
    
    @staticmethod
    def _evaluate(model, X_test, y_test):

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precesion = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred)

        return {
            "Test accuracy": accuracy, 
            "Test recall score":recall,
            "Test precesion score: ": precesion, 
            "Test f1 score: ": f1,
            # "Test classification report": class_rep
        }


