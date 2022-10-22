import logging
from src.utils import logger
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    precision_score,
    f1_score,
    # classification_report,
)
from src.components.regression_model_training import TrainModel
from src.entity import TrainingConfig
from src.entity import TrainingParams


class TrainClassificationModel(TrainModel):
    def __init__(
        self,
        config: TrainingConfig,
        params: TrainingParams,
        logger: logging.Logger = logger,
    ):
        super().__init__(config, params, logger)
        self.output_column = self.config.classification_output
        self.optimize_direction = "maximize"
        self.optimize_scoring = "recall"

    @staticmethod
    def _evaluate(model, X_test, y_test):

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precesion = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # class_rep = classification_report(y_test, y_pred)

        return {
            "Test accuracy": accuracy,
            "Test recall score": recall,
            "Test precesion score: ": precesion,
            "Test f1 score: ": f1,
            # "Test classification report": class_rep
        }
