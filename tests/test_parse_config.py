import pytest
import os
import configparser
import json # Used for comparison with json.loads output

# Adjust the import path to correctly find parse_config.py
# Assuming tests/ is at the same level as project/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project.utils.parse_config import (
    load_config,
    get_model_and_hyperparams,
    get_data_config,
    get_grid_search_config,
    get_grid_search_params,
    get_logging_config,
    get_training_config
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression

# Define a temporary config file content for testing
TEMP_CONFIG_CONTENT = """
[model]
model_name = xgboost
task = classification

[hyperparameters.xgboost]
n_estimators = 100
max_depth = 5
learning_rate = 0.1

[data]
data_path = test_data.csv
target_column = TargetColumn
visualise_data = yes
check_imbalance = no
test_size = 0.25
random_state = 42
stratify = yes
columns_to_drop = colA, colB
categorical_columns_settings = {"Gender": {"imputer": "cat_imputer", "encoder": "onehot", "encoder_options": {"sparse_output": false}}}
numerical_columns_settings = {"Age": {"imputer": "num_imputer", "scaler": "standard"}}

[training]
use_kfold_cv = yes
n_splits = 5
shuffle_cv = yes
random_state_cv = 123
use_optuna = no
n_trials = 50
timeout = 300
optuna_direction = maximize

[logging]
save_model = yes
model_output_path = models/
mlflow_tracking_uri = file:///tmp/mlruns

[grid_params.xgboost]
params = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
"""

# Pytest fixture to create a temporary config file for each test
@pytest.fixture
def temp_config_file(tmp_path):
    config_file = tmp_path / "test_config.cfg"
    config_file.write_text(TEMP_CONFIG_CONTENT)
    return str(config_file)

# --- Tests for parse_config.py functions ---

def test_load_config(temp_config_file):
    config = load_config(temp_config_file)
    assert isinstance(config, configparser.ConfigParser)
    assert config['model']['model_name'] == 'xgboost'
    assert config['data']['target_column'] == 'TargetColumn'

def test_get_model_and_hyperparams(temp_config_file):
    config = load_config(temp_config_file)
    model_class, hyperparams = get_model_and_hyperparams(config)

    assert model_class == XGBClassifier
    assert isinstance(hyperparams, dict)
    assert hyperparams['n_estimators'] == '100' # Note: configparser reads as strings
    assert hyperparams['max_depth'] == '5'

def test_get_model_and_hyperparams_linear_regression(tmp_path):
    # Test linear regression specific case
    config_file = tmp_path / "test_lin_reg_config.cfg"
    config_file.write_text("""
    [model]
    model_name = linear_regression
    task = regression
    [hyperparameters.linear_regression]
    fit_intercept = true
    """)
    config = load_config(str(config_file))
    model_class, hyperparams = get_model_and_hyperparams(config)
    assert model_class == LinearRegression
    assert hyperparams['fit_intercept'] == 'true'

def test_get_model_and_hyperparams_unsupported_model(tmp_path):
    config_file = tmp_path / "test_unsupported_config.cfg"
    config_file.write_text("""
    [model]
    model_name = unsupported_model
    task = classification
    """)
    config = load_config(str(config_file))
    with pytest.raises(ValueError, match="Unsupported model name: unsupported_model"):
        get_model_and_hyperparams(config)

def test_get_model_and_hyperparams_linear_regression_classification_task(tmp_path):
    config_file = tmp_path / "test_lin_reg_classification_config.cfg"
    config_file.write_text("""
    [model]
    model_name = linear_regression
    task = classification
    """)
    config = load_config(str(config_file))
    with pytest.raises(ValueError, match="Linear Regression is only applicable for regression tasks."):
        get_model_and_hyperparams(config)


def test_get_data_config(temp_config_file):
    config = load_config(temp_config_file)
    data_config = get_data_config(config)

    assert isinstance(data_config, dict)
    assert data_config['data_path'] == 'test_data.csv'
    assert data_config['target_column'] == 'TargetColumn'
    assert data_config['visualise_data'] is True
    assert data_config['check_imbalance'] is False
    assert data_config['test_size'] == 0.25
    assert data_config['random_state'] == 42
    assert data_config['stratify'] is True
    assert data_config['columns_to_drop'] == ['colA', 'colB']

    # Test JSON parsing for dict settings
    assert isinstance(data_config['categorical_columns_settings'], dict)
    assert data_config['categorical_columns_settings'] == {'Gender': {'imputer': 'cat_imputer', 'encoder': 'onehot', 'encoder_options': {'sparse_output': False}}}
    
    assert isinstance(data_config['numerical_columns_settings'], dict)
    assert data_config['numerical_columns_settings'] == {'Age': {'imputer': 'num_imputer', 'scaler': 'standard'}}

def test_get_data_config_no_json_settings(tmp_path):
    config_file = tmp_path / "test_no_json_config.cfg"
    config_file.write_text("""
    [data]
    data_path = path.csv
    target_column = Y
    visualise_data = no
    check_imbalance = no
    test_size = 0.2
    random_state = 0
    stratify = no
    """)
    config = load_config(str(config_file))
    data_config = get_data_config(config)
    assert data_config['categorical_columns_settings'] == {}
    assert data_config['numerical_columns_settings'] == {}

def test_get_data_config_invalid_json(tmp_path):
    config_file = tmp_path / "test_invalid_json_config.cfg"
    config_file.write_text("""
    [data]
    data_path = path.csv
    target_column = Y
    visualise_data = no
    check_imbalance = no
    test_size = 0.2
    random_state = 0
    stratify = no
    categorical_columns_settings = {"Gender": "invalid_json
    """) # Malformed JSON
    config = load_config(str(config_file))
    with pytest.raises(ValueError, match="Error parsing categorical_columns_settings JSON"):
        get_data_config(config)

def test_get_grid_search_config(temp_config_file):
    config = load_config(temp_config_file)
    grid_search_config = get_grid_search_config(config)

    assert isinstance(grid_search_config, dict)
    assert grid_search_config['use_optuna'] is False
    assert grid_search_config['n_trials'] == 50
    assert grid_search_config['timeout'] == 300
    assert grid_search_config['optuna_direction'] == 'maximize'
    assert grid_search_config['use_kfold_cv'] is True
    assert grid_search_config['n_splits'] == 5
    assert grid_search_config['shuffle_cv'] is True
    assert grid_search_config['random_state_cv'] == 123

def test_get_grid_search_params(temp_config_file):
    config = load_config(temp_config_file)
    params = get_grid_search_params(config, 'xgboost')
    assert isinstance(params, dict)
    assert params == {'n_estimators': [50, 100], 'max_depth': [3, 5]}

def test_get_grid_search_params_no_section(temp_config_file):
    config = load_config(temp_config_file)
    params = get_grid_search_params(config, 'non_existent_model')
    assert params == {}

def test_get_grid_search_params_invalid_eval_string(tmp_path):
    config_file = tmp_path / "test_invalid_eval_config.cfg"
    config_file.write_text("""
    [grid_params.test_model]
    params = {'n_estimators': [50, 100], 'max_depth': [3, 5],
    """) # Malformed Python dict string
    config = load_config(str(config_file))
    with pytest.raises(ValueError, match="Error parsing param_grid for test_model"):
        get_grid_search_params(config, 'test_model')


def test_get_logging_config(temp_config_file):
    config = load_config(temp_config_file)
    logging_config = get_logging_config(config)
    assert isinstance(logging_config, dict)
    assert logging_config['save_model'] is True
    assert logging_config['model_output_path'] == 'models/'
    assert logging_config['mlflow_tracking_uri'] == 'file:///tmp/mlruns'

def test_get_training_config(temp_config_file):
    config = load_config(temp_config_file)
    training_config = get_training_config(config)
    assert isinstance(training_config, dict)
    assert training_config['use_kfold_cv'] is True
    assert training_config['n_splits'] == 5
    assert training_config['shuffle_cv'] is True
    assert training_config['random_state_cv'] == 123
    assert training_config['use_optuna'] is False
    assert training_config['n_trials'] == 50
    assert training_config['timeout'] == 300
    assert training_config['optuna_direction'] == 'maximize'