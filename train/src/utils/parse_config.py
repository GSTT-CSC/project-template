import configparser
import os
import json
from typing import Dict, Any, Tuple, Union, List

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def load_config(config_path: str) -> configparser.ConfigParser:
    """Load configuration from .cfg file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def get_model_and_hyperparams(config: configparser.ConfigParser) -> Tuple[Any, Dict[str, Any]]:
    """
    Retrieves the model class and its hyperparameters from the configuration.

    Args:
        config: The loaded configuration object.

    Returns:
        tuple: Model class and dictionary of hyperparameters.

    Raises:
        ValueError: If unsupported model or invalid task combination.
    """
    model_name = config["model"]["model_name"].strip().lower()
    task = config["model"]["task"].strip().lower()

    # Model mapping
    model_mapping = {
        "random_forest": {
            "classification": RandomForestClassifier,
            "regression": RandomForestRegressor
        },
        "xgboost": {
            "classification": XGBClassifier,
            "regression": XGBRegressor
        },
        "linear_regression": {
            "regression": LinearRegression
        },
        "logistic_regression": {
            "classification": LogisticRegression
        }
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if task not in model_mapping[model_name]:
        raise ValueError(f"Model {model_name} does not support task: {task}")

    model_class = model_mapping[model_name][task]
    
    # Get hyperparameters
    hyperparams_section = f"hyperparameters.{model_name}"
    if hyperparams_section not in config:
        return model_class, {}
    
    hyperparams = {}
    for key, value in config[hyperparams_section].items():
        # Convert string values to appropriate types
        hyperparams[key] = _convert_config_value(value)
    
    return model_class, hyperparams


def get_data_config(config: configparser.ConfigParser) -> Dict[str, Any]:
    """
    Retrieves data-related configurations.

    Args:
        config: The loaded configuration object.

    Returns:
        dict: Data configuration settings.
    """
    data_section = config["data"]
    
    data_config = {
        "data_path": data_section["data_path"],
        "target_column": data_section["target_column"],
        "visualise_data": data_section.getboolean("visualise_data"),
        "check_imbalance": data_section.getboolean("check_imbalance"),
        "test_size": data_section.getfloat("test_size"),
        "random_state": data_section.getint("random_state"),
        "stratify": data_section.getboolean("stratify"),
        "columns_to_drop": _parse_comma_separated(data_section.get("columns_to_drop", ""))
    }
    
    # Parse preprocessing settings
    preprocessing_settings = _parse_preprocessing_config(config)
    data_config["preprocessor_settings"] = preprocessing_settings
    
    return data_config


def _parse_preprocessing_config(config: configparser.ConfigParser) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Parse preprocessing configuration from config sections."""
    preprocessing_settings = {"categorical": {}, "numerical": {}}
    
    # Parse categorical preprocessing
    if "preprocessing.categorical" in config:
        for key in config["preprocessing.categorical"]:
            column_name, setting_type = key.split(".", 1)
            
            if column_name not in preprocessing_settings["categorical"]:
                preprocessing_settings["categorical"][column_name] = {}
            
            value = config["preprocessing.categorical"][key]
            if setting_type == "encoder_options":
                try:
                    preprocessing_settings["categorical"][column_name][setting_type] = json.loads(value)
                except json.JSONDecodeError:
                    preprocessing_settings["categorical"][column_name][setting_type] = {}
            else:
                preprocessing_settings["categorical"][column_name][setting_type] = value
    
    # Parse numerical preprocessing
    if "preprocessing.numerical" in config:
        for key in config["preprocessing.numerical"]:
            column_name, setting_type = key.split(".", 1)
            
            if column_name not in preprocessing_settings["numerical"]:
                preprocessing_settings["numerical"][column_name] = {}
            
            value = config["preprocessing.numerical"][key]
            preprocessing_settings["numerical"][column_name][setting_type] = value
    
    return preprocessing_settings


def get_logging_config(config: configparser.ConfigParser) -> Dict[str, Any]:
    """
    Retrieves logging-related configurations.

    Args:
        config: The loaded configuration object.

    Returns:
        dict: Logging configuration settings.
    """
    logging_section = config["logging"]
    
    return {
        "save_model": logging_section.getboolean("save_model"),
        "model_output_path": logging_section["model_output_path"],
        "mlflow_tracking_uri": logging_section["mlflow_tracking_uri"]
    }


def get_training_config(config: configparser.ConfigParser) -> Dict[str, Any]:
    """
    Retrieves training-related configurations.

    Args:
        config: The loaded configuration object.

    Returns:
        dict: Training configuration settings.
    """
    training_section = config["training"]
    
    return {
        "use_kfold_cv": training_section.getboolean("use_kfold_cv"),
        "n_splits": training_section.getint("n_splits"),
        "shuffle_cv": training_section.getboolean("shuffle_cv"),
        "random_state_cv": training_section.getint("random_state_cv"),
        "use_optuna": training_section.getboolean("use_optuna"),
        "n_trials": training_section.getint("n_trials"),
        "timeout": training_section.getint("timeout"),
        "optuna_direction": training_section["optuna_direction"]
    }


def _convert_config_value(value: str) -> Union[str, int, float, bool]:
    """Convert string configuration value to appropriate Python type."""
    value = value.strip()
    
    # Boolean conversion
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Integer conversion
    try:
        if '.' not in value:
            return int(value)
    except ValueError:
        pass
    
    # Float conversion
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def _parse_comma_separated(value: str) -> List[str]:
    """Parse comma-separated string into list of stripped strings."""
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def validate_config(config: configparser.ConfigParser) -> None:
    """
    Validate configuration settings.
    
    Args:
        config: The loaded configuration object.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    required_sections = ["data", "model", "logging", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required section '{section}' missing from config")
    
    # Validate data path exists
    data_path = config["data"]["data_path"]
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found: {data_path}")
    
    # Validate model and task combination
    try:
        get_model_and_hyperparams(config)
    except ValueError as e:
        raise ValueError(f"Invalid model configuration: {e}")
    
    print("Configuration validation passed.")