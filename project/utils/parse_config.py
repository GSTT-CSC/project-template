import configparser
import os
import json # Import json for parsing dictionary strings from config

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression


def load_config(config_path):
    """
    Loads the configuration from the specified INI-style config file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        configparser.ConfigParser: The loaded ConfigParser object.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def get_model_and_hyperparams(config):
    """
    Retrieves the model class and its hyperparameters from the configuration.

    Args:
        config (configparser.ConfigParser): The loaded configuration object.

    Returns:
        tuple: A tuple containing the model class and a dictionary of its hyperparameters.

    Raises:
        ValueError: If an unsupported model name is specified or linear regression
                    is applied to a non-regression task.
    """
    model_name = config["model"]["model_name"].strip() # Ensure no leading/trailing whitespace
    task = config["model"]["task"].strip()

    if model_name == "random_forest":
        model_class = RandomForestClassifier if task == "classification" else RandomForestRegressor
        hyperparams = dict(config["hyperparameters.random_forest"])
    elif model_name == "xgboost":
        model_class = XGBClassifier if task == "classification" else XGBRegressor
        hyperparams = dict(config["hyperparameters.xgboost"])
    elif model_name == "linear_regression":
        if task != "regression":
            raise ValueError("Linear Regression is only applicable for regression tasks.")
        model_class = LinearRegression
        hyperparams = dict(config["hyperparameters.linear_regression"])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model_class, hyperparams

def get_data_config(config):
    """
    Retrieves data-related configurations from the ConfigParser object.

    Args:
        config (configparser.ConfigParser): The loaded configuration object.

    Returns:
        dict: A dictionary containing data configuration settings.
    """
    data_config = {
        "data_path": config["data"]["data_path"],
        "target_column": config["data"]["target_column"],
        "visualise_data": config["data"].getboolean("visualise_data"),
        "check_imbalance": config["data"].getboolean("check_imbalance"),
        "test_size": config["data"].getfloat("test_size"),
        "random_state": config["data"].getint("random_state"),
        "stratify": config["data"].getboolean("stratify"),
        "columns_to_drop": [col.strip() for col in config["data"].get("columns_to_drop", "").split(',') if col.strip()]
    }

    # Parse JSON strings for categorical and numerical column settings
    if "categorical_columns_settings" in config["data"]:
        try:
            data_config["categorical_columns_settings"] = json.loads(config["data"]["categorical_columns_settings"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing categorical_columns_settings JSON: {e}. Please ensure it's a valid JSON string.")
    else:
        data_config["categorical_columns_settings"] = {}

    if "numerical_columns_settings" in config["data"]:
        try:
            data_config["numerical_columns_settings"] = json.loads(config["data"]["numerical_columns_settings"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing numerical_columns_settings JSON: {e}. Please ensure it's a valid JSON string.")
    else:
        data_config["numerical_columns_settings"] = {}

    return data_config


def get_grid_search_config(config):
    """
    Retrieves grid search (or hyperparameter tuning) configurations.
    This function specifically extracts settings for Optuna or GridSearchCV.
    """
    return {
        "use_optuna": config["training"].getboolean("use_optuna"),
        "n_trials": config["training"].getint("n_trials"),
        "timeout": config["training"].getint("timeout"),
        "optuna_direction": config["training"]["optuna_direction"],
        "use_kfold_cv": config["training"].getboolean("use_kfold_cv"), # Added this for consistency with training section
        "n_splits": config["training"].getint("n_splits"), # Added this for consistency with training section
        "shuffle_cv": config["training"].getboolean("shuffle_cv"), # Added this for consistency with training section
        "random_state_cv": config["training"].getint("random_state_cv") # Added this for consistency with training section
    }


def get_grid_search_params(config, model_name):
    """
    Retrieves the parameter grid for GridSearchCV for a specific model.

    Args:
        config (configparser.ConfigParser): The loaded configuration object.
        model_name (str): The name of the model to retrieve parameters for.

    Returns:
        dict: A dictionary representing the parameter grid.

    Raises:
        ValueError: If JSON parsing for the grid parameters fails.
    """
    section_name = f"grid_params.{model_name.strip()}" # Ensure no whitespace
    if section_name in config:
        params_string = config[section_name]["params"]
        try:
            # Using eval() can be risky, but common for parameter grids where input is controlled.
            # Alternatively, if your grid is strict JSON, use json.loads.
            # For Python dicts with None/tuples, eval is often necessary.
            return eval(params_string)
        except Exception as e:
            raise ValueError(
                f"Error parsing param_grid for {model_name} from section '{section_name}': {e}\\n"
                f"Ensure 'params' value is a valid Python dictionary string. String: {params_string}"
            )
    else:
        # Return empty dict if no grid params are defined for this model,
        # train.py should handle if use_grid_search is True but params are empty.
        return {}


def get_logging_config(config):
    """
    Retrieves logging-related configurations from the ConfigParser object.

    Args:
        config (configparser.ConfigParser): The loaded configuration object.

    Returns:
        dict: A dictionary containing logging configuration settings.
    """
    logging_config = {
        "save_model": config["logging"].getboolean("save_model"),
        "model_output_path": config["logging"]["model_output_path"],
        "mlflow_tracking_uri": config["logging"]["mlflow_tracking_uri"]
    }
    return logging_config

def get_training_config(config):
    """
    Retrieves training-related configurations from the ConfigParser object.

    Args:
        config (configparser.ConfigParser): The loaded configuration object.

    Returns:
        dict: A dictionary containing training configuration settings.
    """
    training_config = {
        "use_kfold_cv": config["training"].getboolean("use_kfold_cv"),
        "n_splits": config["training"].getint("n_splits"),
        "shuffle_cv": config["training"].getboolean("shuffle_cv"),
        "random_state_cv": config["training"].getint("random_state_cv"),
        "use_optuna": config["training"].getboolean("use_optuna"),
        "n_trials": config["training"].getint("n_trials"),
        "timeout": config["training"].getint("timeout"),
        "optuna_direction": config["training"]["optuna_direction"]
    }
    return training_config