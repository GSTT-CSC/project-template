"""
mlflow PythonModel wrapper class for themodels. 
This class is a custom wrapper that uses mlflow's PythonModel class for serving the models.
"""

import mlflow
import pandas as pd
from torch import topk
from project_NLP.NLPDataset import NLPDataset
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer):
        self.model = model
        with open("label_columns.data", "rb") as filehandle:
            # read the data as binary data stream
            label_columns = pickle.load(filehandle)
        self.dataset = NLPDataset(tokenizer=tokenizer, label_columns=label_columns)

    def predict(self, context, model_input):
        logger.info(f"Running prediction service: {model_input}")
        encoding = self.dataset.encoder(model_input.TextString.tolist())

        logger.info(f"Running inference")
        _, test_prediction = self.model(
            encoding["input_ids"], encoding["attention_mask"]
        )
        res = topk(test_prediction, 1).indices.tolist()

        logger.info(f"inference results: {test_prediction} -- res {res}")
        confidences = pd.DataFrame(
            test_prediction.tolist(), columns=self.dataset.label_columns
        )
        prediction = pd.DataFrame(
            {"Prediction": [self.dataset.label_columns[x[0]] for x in res]}
        )

        results = pd.concat([prediction, confidences], axis=1)
        logger.info(f"Inference complete, results: {results}")
        return results


# https://www.alexanderjunge.net/blog/mlflow-sagemaker-deploy/
# https://docs.databricks.com/_static/notebooks/mlflow/mlflow-end-to-end-example.html
