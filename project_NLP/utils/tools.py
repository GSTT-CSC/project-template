import mlflow
from project_NLP.utils.Wrapper import Wrapper
import pandas as pd


def wrap_and_log(model, tokenizer):
    """
    This function wraps and logs the trained model using MLflow.

    1. freezes the model to ensure the model weights aren't updated anymore.
    2. wraps the model to simplify the model's API for easier inference.
    3. A test DataFrame is prepared to infer the model's signature which defines the schema of the model's inputs and outputs.
    4. the model artifact along with its metadata (including the model signature, any associated code,
    and pip requirements) is logged to the current MLflow run.

    Args:
        model (nn.Module): The trained model to be logged.
        tokenizer (Tokenizer): The tokenizer used during the model training.
    Returns:
        None
    """

    model.eval()
    model.freeze()
    wrappedModel = Wrapper(model, tokenizer)
    test_df = pd.DataFrame(["Test string 1", "Test string 2"], columns=["TextString"])
    signature = mlflow.models.signature.infer_signature(
        test_df, wrappedModel.predict(None, test_df)
    )
    mlflow.pyfunc.log_model(
        "project_nlp",
        python_model=wrappedModel,
        signature=signature,
        code_path=["project_NLP/", "config/"],
        pip_requirements="requirements.txt",
    )
