import pytest
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

from project.DataModule import DataModule 

@pytest.fixture(scope="module")
def dummy_data_path(tmp_path_factory):
    """Creates a dummy CSV file for testing and returns its path."""
    data = {
        'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'] * 3, 
        'Age': list(range(20, 50)), 
        'Height': [1.70, 1.85, 1.65, 1.75, 1.72, 1.80, 1.68, 1.90, 1.60, 1.78] * 3, 
        'Weight': [70.5, 90.2, 60.0, 80.1, 75.3, 85.0, 68.9, 95.5, 55.1, 82.7] * 3, 
        'Favorite Food': ['Pizza', 'Burger', 'Salad', 'Pasta', 'Sushi', 'Taco', 'Steak', 'Curry', 'Ramen', 'Pho'] * 3, 
        'TargetColumn': ['ClassA', 'ClassB', 'ClassC'] * 10 
    }
    df = pd.DataFrame(data)

    data_dir = tmp_path_factory.mktemp("data")
    fn = data_dir / "dummy_data.csv"

    df.to_csv(fn, index=False)
    return fn

@pytest.fixture(scope="module")
def basic_data_module(dummy_data_path):
    """Provides a basic DataModule instance for testing."""
    return DataModule(data_path=dummy_data_path, target_column='TargetColumn')

@pytest.fixture(scope="module")
def configured_data_module(dummy_data_path):
    """Provides a DataModule instance with specific configurations."""
    preprocessor_settings = {
        'numerical': {
            'Age': {'scaler': 'standard'},    
            'Height': {'scaler': 'standard'}, 
            'Weight': {'scaler': 'standard'}  
        },
        'categorical': {
            'Gender': {'encoder': 'onehot', 'encoder_options': {'sparse_output': False}} 
        }
    }
    return DataModule(
        data_path=dummy_data_path,
        target_column='TargetColumn',
        columns_to_drop=['Favorite Food'],
        preprocessor_settings=preprocessor_settings,
        visualise=False,
        check_imbalance=True,
        stratify=True,
        test_size=0.2
    )

def test_initialization(basic_data_module):
    """Test DataModule initialization."""
    dm = basic_data_module
    assert dm.data_path is not None
    assert dm.target_column == 'TargetColumn'
    assert dm.columns_to_drop == []
    assert dm.preprocessor_settings == {}
    assert dm.data is None
    assert dm.preprocessor is None
    assert dm.visualise is False
    assert dm.check_imbalance is False
    assert dm.test_size == 0.2
    assert dm.stratify is False
    assert dm.random_state == 42

def test_load_data(dummy_data_path):
    """Test data loading functionality."""
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn')
    X, y = dm.load_and_prepare()
    assert dm.data is not None
    assert not dm.data.empty
    assert 'TargetColumn' not in X.columns
    assert 'TargetColumn' in y.name
    assert X.shape[0] == y.shape[0]

def test_load_and_prepare(dummy_data_path, capsys):
    """Test load_and_prepare method, including column dropping and inference."""
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn', columns_to_drop=['Weight'])
    X, y = dm.load_and_prepare()
    assert 'Weight' not in X.columns
    assert 'Weight' not in dm.data.columns
    assert dm.numerical_features or dm.categorical_features 
    assert 'Age' in dm.numerical_features
    assert 'Gender' in dm.categorical_features
    
    captured = capsys.readouterr()
    assert f"Data loaded successfully from {dummy_data_path}. Shape: (30, 6)" in captured.out
    assert "Dropped columns: Weight" in captured.out


def test_infer_column_types(dummy_data_path):
    """Test column type inference."""
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn')
    X, _ = dm.load_and_prepare()
    assert 'Age' in dm.numerical_features
    assert 'Gender' in dm.categorical_features
    assert 'Height' in dm.numerical_features
    assert 'Weight' in dm.numerical_features
    assert 'Favorite Food' in dm.categorical_features
    assert 'TargetColumn' not in dm.numerical_features and 'TargetColumn' not in dm.categorical_features

def test_setup_preprocessor_basic_inference(basic_data_module, capsys):
    """
    Test setup_preprocessor with basic inference and no specific settings.
    Should result in a ColumnTransformer with no active transformers.
    """
    dm = basic_data_module
    X, _ = dm.load_and_prepare() 
    
    assert isinstance(dm.preprocessor, ColumnTransformer)
    assert len(dm.preprocessor.transformers) == 0 
    assert dm.preprocessor.remainder == 'passthrough' 
    
    captured = capsys.readouterr()
    assert "No active transformers configured for any columns. Creating a passthrough preprocessor." in captured.out


def test_setup_preprocessor_with_custom_settings(configured_data_module):
    """
    Test setup_preprocessor with custom settings (StandardScaler, OneHotEncoder).
    Should result in a ColumnTransformer.
    """
    dm = configured_data_module
    X, y = dm.load_and_prepare()
    
    assert isinstance(dm.preprocessor, ColumnTransformer)
    
    transformer_names = [name for name, _, _ in dm.preprocessor.transformers]
    assert 'num_pipeline' in transformer_names
    assert 'cat_pipeline' in transformer_names

    for name, pipeline, _ in dm.preprocessor.transformers:
        if name == 'num_pipeline':
            assert isinstance(pipeline, Pipeline)
            assert any(isinstance(step[1], StandardScaler) for step in pipeline.steps)
        if name == 'cat_pipeline':
            assert isinstance(pipeline, Pipeline)
            assert any(isinstance(step[1], OneHotEncoder) for step in pipeline.steps)


def test_create_and_fit_preprocessor(configured_data_module):
    """Test creating and fitting the preprocessor."""
    dm = configured_data_module
    X, _ = dm.load_and_prepare() 
    
    fitted_preprocessor = dm.create_and_fit_preprocessor(X)
    
    assert isinstance(fitted_preprocessor, ColumnTransformer)
    assert hasattr(fitted_preprocessor, 'transform') 

def test_transform_data(configured_data_module):
    """Test data transformation."""
    dm = configured_data_module
    X, y = dm.load_and_prepare() 
    dm.preprocessor = dm.create_and_fit_preprocessor(X)
    
    X_transformed = dm.transform_data(X)
    
    assert isinstance(X_transformed, np.ndarray) 
    assert X_transformed.shape[0] == X.shape[0]

    assert X_transformed.shape[1] == 5


def test_perform_train_test_split_no_stratify(basic_data_module):
    """Test non-stratified train-test split."""
    dm = basic_data_module
    dm.stratify = False 
    X, y = dm.load_and_prepare()
    
    X_train, X_test, y_train, y_test = dm.perform_train_test_split(X, y)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

def test_perform_train_test_split_stratified(configured_data_module):
    """Test stratified train-test split."""
    dm = configured_data_module 
    X, y = dm.load_and_prepare()
    
    assert dm.stratify is True
    X_train, X_test, y_train, y_test = dm.perform_train_test_split(X, y)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    train_counts = y_train.value_counts(normalize=True)
    test_counts = y_test.value_counts(normalize=True)
    overall_counts = y.value_counts(normalize=True)

    pd.testing.assert_series_equal(train_counts, overall_counts, check_exact=False, rtol=0.1, atol=0.1, check_index=False)
    pd.testing.assert_series_equal(test_counts, overall_counts, check_exact=False, rtol=0.1, atol=0.1, check_index=False)


def test_visualize_column_distributions(basic_data_module, capsys):
    """Test column distribution visualization."""
    dm = basic_data_module
    dm.visualise = True 
    dm.load_and_prepare()
    dm.visualize_column_distributions()
    captured = capsys.readouterr()
    
    assert "--- Generating Column Distribution Visualizations ---" in captured.out
    assert "Column distribution visualizations complete." in captured.out
    
    assert "Displaying distributions for numerical columns..." in captured.out
    assert "Displaying distributions for categorical columns..." in captured.out


def test_check_imbalance_runs(configured_data_module, capsys):
    """Test that imbalance check runs and prints output when enabled."""
    dm = configured_data_module 
    dm.load_and_prepare() 
    captured = capsys.readouterr()
    
    assert "Checking target class imbalance for 'TargetColumn'..." in captured.out
    assert "Class distribution for 'TargetColumn':" in captured.out


def test_get_feature_names_after_preprocessing_passthrough(basic_data_module):
    """Test getting feature names when preprocessor is a passthrough ColumnTransformer."""
    dm = basic_data_module
    X_raw, _ = dm.load_and_prepare()

    initial_feature_names = X_raw.columns.tolist()
    
    feature_names = dm.get_feature_names_after_preprocessing(dm.preprocessor, initial_feature_names)
    
    expected_names = ['Gender', 'Age', 'Height', 'Weight', 'Favorite Food']
    assert sorted(feature_names) == sorted(expected_names)


def test_get_feature_names_after_preprocessing_with_transformers(configured_data_module):
    """Test getting feature names after preprocessing with actual transformers."""
    dm = configured_data_module
    X_raw, _ = dm.load_and_prepare() 
    dm.preprocessor = dm.create_and_fit_preprocessor(X_raw)

    initial_feature_names = X_raw.columns.tolist()

    fitted_preprocessor = dm.create_and_fit_preprocessor(X_raw) 
    feature_names = dm.get_feature_names_after_preprocessing(fitted_preprocessor, initial_feature_names)

    expected_names_parts = [
        'num_pipeline__Age',
        'num_pipeline__Height',
        'num_pipeline__Weight',
        'cat_pipeline__Gender_Female', 
        'cat_pipeline__Gender_Male'  
    ] 
    
    
    for expected_part in expected_names_parts:
        assert any(expected_part in name for name in feature_names), f"'{expected_part}' not found in feature names: {feature_names}"
    
   
    assert len(feature_names) == 5

def test_empty_columns_to_drop(dummy_data_path):
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn', columns_to_drop=[])
    X, _ = dm.load_and_prepare()
    assert 'Favorite Food' in X.columns

def test_non_existent_columns_to_drop(dummy_data_path, capsys):
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn', columns_to_drop=['NonExistentColumn'])
    X, _ = dm.load_and_prepare()
    captured = capsys.readouterr()
    assert "No specified columns were dropped (they might not exist)." in captured.out
    assert 'NonExistentColumn' not in X.columns 

def test_no_numerical_or_categorical_features(tmp_path_factory):
 
    data_file = tmp_path_factory.mktemp("data_minimal") / "minimal_data.csv"
    pd.DataFrame({'A': [1,2,3], 'Target': [0,1,0]}).to_csv(data_file, index=False)
    dm = DataModule(data_path=str(data_file), target_column='Target')
    X, y = dm.load_and_prepare()
    assert 'A' in dm.numerical_features
    assert not dm.categorical_features # No categorical features
    assert isinstance(dm.preprocessor, ColumnTransformer)
    assert len(dm.preprocessor.transformers) == 0 
    
    fitted_preprocessor = dm.create_and_fit_preprocessor(X) 
    X_transformed = dm.transform_data(X)
    assert X_transformed.shape[1] == 1 

    data_file_empty = tmp_path_factory.mktemp("data_empty") / "empty_data.csv"
    pd.DataFrame({'A': [1,2,3], 'B': ['x','y','z'], 'Target': [0,1,0]}).to_csv(data_file_empty, index=False)
    
    dm_empty_settings = DataModule(
        data_path=str(data_file_empty),
        target_column='Target',
        preprocessor_settings={'numerical': {}, 'categorical': {}}
    )
    X_empty, _ = dm_empty_settings.load_and_prepare()
    
    assert isinstance(dm_empty_settings.preprocessor, ColumnTransformer)
    assert len(dm_empty_settings.preprocessor.transformers) == 0 
    assert dm_empty_settings.preprocessor.remainder == 'passthrough'
    
    dm_empty_settings.create_and_fit_preprocessor(X_empty) 
    
    transformed_empty = dm_empty_settings.transform_data(X_empty)

    assert transformed_empty.shape[1] == 2
 


