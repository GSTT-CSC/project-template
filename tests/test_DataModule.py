import pytest
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

# Assuming your DataModule is in a file named 'DataModule.py' inside a 'project' directory
from project.DataModule import DataModule # Adjust this import based on your project structure

@pytest.fixture(scope="module")
def dummy_data_path(tmp_path_factory):
    """Creates a dummy CSV file for testing and returns its path."""
    data = {
        'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'] * 3, # 30 samples
        'Age': list(range(20, 50)), # 30 samples
        'Height': [1.70, 1.85, 1.65, 1.75, 1.72, 1.80, 1.68, 1.90, 1.60, 1.78] * 3, # 30 samples
        'Weight': [70.5, 90.2, 60.0, 80.1, 75.3, 85.0, 68.9, 95.5, 55.1, 82.7] * 3, # 30 samples
        'Favorite Food': ['Pizza', 'Burger', 'Salad', 'Pasta', 'Sushi', 'Taco', 'Steak', 'Curry', 'Ramen', 'Pho'] * 3, # 30 samples
        'TargetColumn': ['ClassA', 'ClassB', 'ClassC'] * 10 # 30 samples, 10 per class
    }
    df = pd.DataFrame(data)

    # Use the / operator for path joining with Pathlib objects
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
    # MODIFIED: Updated preprocessor_settings structure to match DataModule.py's parsing logic
    preprocessor_settings = {
        'numerical': {
            'Age': {'scaler': 'standard'},    # Apply StandardScaler to Age
            'Height': {'scaler': 'standard'}, # Apply StandardScaler to Height
            'Weight': {'scaler': 'standard'}  # Apply StandardScaler to Weight
        },
        'categorical': {
            'Gender': {'encoder': 'onehot', 'encoder_options': {'sparse_output': False}} # Apply OneHotEncoder to Gender
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
    # MODIFIED: Expect None at initialization
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
    assert dm.numerical_features or dm.categorical_features # Ensure types are inferred
    assert 'Age' in dm.numerical_features
    assert 'Gender' in dm.categorical_features
    
    captured = capsys.readouterr()
    assert f"Data loaded successfully from {dummy_data_path}. Shape: (30, 6)" in captured.out
    assert "Dropped columns: Weight" in captured.out


def test_infer_column_types(dummy_data_path):
    """Test column type inference."""
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn')
    X, _ = dm.load_and_prepare()
    # infer_column_types is called within load_and_prepare, no need to call explicitly here
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
    # MODIFIED: load_and_prepare now calls setup_preprocessor internally
    X, _ = dm.load_and_prepare() 
    
    # MODIFIED: Assert that it's a ColumnTransformer with no actual transformers
    assert isinstance(dm.preprocessor, ColumnTransformer)
    assert len(dm.preprocessor.transformers) == 0 # Should be empty for basic passthrough
    assert dm.preprocessor.remainder == 'passthrough' # Check remainder strategy
    
    captured = capsys.readouterr()
    # MODIFIED: Check the correct print statement from DataModule
    assert "No active transformers configured for any columns. Creating a passthrough preprocessor." in captured.out


def test_setup_preprocessor_with_custom_settings(configured_data_module):
    """
    Test setup_preprocessor with custom settings (StandardScaler, OneHotEncoder).
    Should result in a ColumnTransformer.
    """
    dm = configured_data_module
    # load_and_prepare now calls setup_preprocessor internally
    X, y = dm.load_and_prepare()
    
    assert isinstance(dm.preprocessor, ColumnTransformer)
    
    # Check if correct transformers are present in the ColumnTransformer
    transformer_names = [name for name, _, _ in dm.preprocessor.transformers]
    assert 'num_pipeline' in transformer_names
    assert 'cat_pipeline' in transformer_names

    # Check the actual transformers within the pipelines
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
    X, _ = dm.load_and_prepare() # Calls setup_preprocessor internally
    
    fitted_preprocessor = dm.create_and_fit_preprocessor(X)
    
    # It should be a ColumnTransformer now based on configured_data_module's settings
    assert isinstance(fitted_preprocessor, ColumnTransformer)
    assert hasattr(fitted_preprocessor, 'transform') # Check if it's a fitted transformer

def test_transform_data(configured_data_module):
    """Test data transformation."""
    dm = configured_data_module
    X, y = dm.load_and_prepare() # Calls setup_preprocessor and potentially create_and_fit_preprocessor internally
    dm.preprocessor = dm.create_and_fit_preprocessor(X) # Ensure preprocessor is fitted for this test
    
    X_transformed = dm.transform_data(X)
    
    assert isinstance(X_transformed, np.ndarray) # Transformed data from ColumnTransformer is typically a numpy array
    assert X_transformed.shape[0] == X.shape[0]
    # The number of columns will change due to one-hot encoding, etc.
    # Original columns from X (after dropping Favorite Food): Gender, Age, Height, Weight -> 4 features
    # Gender ('Male', 'Female') -> 2 columns after OneHotEncoder
    # Age, Height, Weight (numerical) -> 3 columns after StandardScaler
    # Total expected columns = 2 (Gender OHE) + 3 (Numerical Scaled) = 5
    assert X_transformed.shape[1] == 5


def test_perform_train_test_split_no_stratify(basic_data_module):
    """Test non-stratified train-test split."""
    dm = basic_data_module
    dm.stratify = False # Explicitly set for this test
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
    dm = configured_data_module # This fixture has stratify=True and larger data
    X, y = dm.load_and_prepare()
    
    # Ensure stratify is True for this specific test case, as the fixture sets it
    assert dm.stratify is True
    X_train, X_test, y_train, y_test = dm.perform_train_test_split(X, y)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    # Check for stratification (approximate proportion)
    train_counts = y_train.value_counts(normalize=True)
    test_counts = y_test.value_counts(normalize=True)
    overall_counts = y.value_counts(normalize=True)

    # Allow for small floating point differences and ignore index order
    pd.testing.assert_series_equal(train_counts, overall_counts, check_exact=False, rtol=0.1, atol=0.1, check_index=False)
    pd.testing.assert_series_equal(test_counts, overall_counts, check_exact=False, rtol=0.1, atol=0.1, check_index=False)


def test_visualize_column_distributions(basic_data_module, capsys):
    """Test column distribution visualization."""
    dm = basic_data_module
    dm.visualise = True # Ensure visualization is enabled
    dm.load_and_prepare() # Load data first
    
    # This will attempt to open matplotlib figures. In CI/CD, you might need a headless backend (e.g., Agg)
    # import matplotlib.pyplot as plt # Make sure this import is at the top of your test file if not already
    # plt.switch_backend('Agg') # Uncomment for headless environments
    
    dm.visualize_column_distributions()
    captured = capsys.readouterr()
    
    assert "--- Generating Column Distribution Visualizations ---" in captured.out
    assert "Column distribution visualizations complete." in captured.out
    
    assert "Displaying distributions for numerical columns..." in captured.out
    assert "Displaying distributions for categorical columns..." in captured.out


def test_check_imbalance_runs(configured_data_module, capsys):
    """Test that imbalance check runs and prints output when enabled."""
    dm = configured_data_module # This fixture has check_imbalance=True and proper data
    dm.load_and_prepare() # This method will trigger the imbalance check
    captured = capsys.readouterr()
    
    assert "Checking target class imbalance for 'TargetColumn'..." in captured.out
    assert "Class distribution for 'TargetColumn':" in captured.out


def test_get_feature_names_after_preprocessing_passthrough(basic_data_module):
    """Test getting feature names when preprocessor is a passthrough ColumnTransformer."""
    dm = basic_data_module
    X_raw, _ = dm.load_and_prepare() # This calls setup_preprocessor internally

    initial_feature_names = X_raw.columns.tolist()
    
    # MODIFIED: Pass the actual dm.preprocessor object, which will be a passthrough ColumnTransformer
    feature_names = dm.get_feature_names_after_preprocessing(dm.preprocessor, initial_feature_names)
    
    # Expected names are the initial features, as no columns are dropped in basic_data_module fixture's setup
    # and the preprocessor is a passthrough ColumnTransformer.
    # The dummy data has: 'Gender', 'Age', 'Height', 'Weight', 'Favorite Food', 'TargetColumn'
    # X_raw will contain all columns except 'TargetColumn'.
    expected_names = ['Gender', 'Age', 'Height', 'Weight', 'Favorite Food']
    assert sorted(feature_names) == sorted(expected_names)


def test_get_feature_names_after_preprocessing_with_transformers(configured_data_module):
    """Test getting feature names after preprocessing with actual transformers."""
    dm = configured_data_module
    X_raw, _ = dm.load_and_prepare() # Calls setup_preprocessor internally
    dm.preprocessor = dm.create_and_fit_preprocessor(X_raw) # Ensure preprocessor is fitted

    initial_feature_names = X_raw.columns.tolist() # e.g., ['Gender', 'Age', 'Height', 'Weight']
    
    # Note: ColumnTransformer.get_feature_names_out() returns names like 'num_pipeline__Age', 'cat_pipeline__Gender_Female', etc.
    # For one-hot encoded 'Gender', it will create 'Gender_Female' and 'Gender_Male'.

    fitted_preprocessor = dm.create_and_fit_preprocessor(X_raw) 
    feature_names = dm.get_feature_names_after_preprocessing(fitted_preprocessor, initial_feature_names)

    # Expected feature names based on the configured_data_module's preprocessor_settings
    # Numerical: Age, Height, Weight (scaled, names remain the same but prefixed by pipeline)
    # Categorical: Gender (one-hot encoded, e.g., Gender_Female, Gender_Male)
    # The ColumnTransformer names them as <pipeline_name>__<original_feature_name> or <pipeline_name>__<original_feature_name>_<category>
    
    # Example expected names after preprocessing with StandardScaler and OneHotEncoder on dummy data
    # (assuming Gender categories 'Female', 'Male' in that order after fitting)
    # The exact order might depend on internal sklearn sorting, so check presence.
    expected_names_parts = [
        'num_pipeline__Age',
        'num_pipeline__Height',
        'num_pipeline__Weight',
        'cat_pipeline__Gender_Female', # Or Gender_Male, order depends on internal OHE category ordering
        'cat_pipeline__Gender_Male'    # Ensure both are present
    ]
    
    # Check if all expected names (or parts) are present in the returned list
    # Use a loop or all() for robustness against exact order
    for expected_part in expected_names_parts:
        assert any(expected_part in name for name in feature_names), f"'{expected_part}' not found in feature names: {feature_names}"
    
    # Also assert the total count for completeness
    assert len(feature_names) == 5

# Additional tests for robustness and edge cases
def test_empty_columns_to_drop(dummy_data_path):
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn', columns_to_drop=[])
    X, _ = dm.load_and_prepare()
    assert 'Favorite Food' in X.columns # Should not be dropped if not specified

def test_non_existent_columns_to_drop(dummy_data_path, capsys):
    dm = DataModule(data_path=dummy_data_path, target_column='TargetColumn', columns_to_drop=['NonExistentColumn'])
    X, _ = dm.load_and_prepare()
    captured = capsys.readouterr()
    assert "No specified columns were dropped (they might not exist)." in captured.out
    assert 'NonExistentColumn' not in X.columns # Should not cause error if not in data

def test_no_numerical_or_categorical_features(tmp_path_factory):
    # Create data with only one numerical and target, no categoricals to ensure passthrough works fine
    data_file = tmp_path_factory.mktemp("data_minimal") / "minimal_data.csv"
    pd.DataFrame({'A': [1,2,3], 'Target': [0,1,0]}).to_csv(data_file, index=False)
    dm = DataModule(data_path=str(data_file), target_column='Target')
    X, y = dm.load_and_prepare()
    assert 'A' in dm.numerical_features
    assert not dm.categorical_features # No categorical features
    assert isinstance(dm.preprocessor, ColumnTransformer)
    # The number of transformers can be 0 if no specific numerical/categorical settings are applied
    # and only 'passthrough' is active based on inferred types.
    # If StandardScaler was specified, len would be 1. With empty numerical settings, it's 0.
    assert len(dm.preprocessor.transformers) == 0 
    
    fitted_preprocessor = dm.create_and_fit_preprocessor(X) # Ensure fitting occurs
    X_transformed = dm.transform_data(X)
    assert X_transformed.shape[1] == 1 # Should still have 1 column 'A'

    # Test case for no inferred numerical/categorical if config explicitly empty
    data_file_empty = tmp_path_factory.mktemp("data_empty") / "empty_data.csv"
    pd.DataFrame({'A': [1,2,3], 'B': ['x','y','z'], 'Target': [0,1,0]}).to_csv(data_file_empty, index=False)
    
    dm_empty_settings = DataModule(
        data_path=str(data_file_empty),
        target_column='Target',
        preprocessor_settings={'numerical': {}, 'categorical': {}} # Explicitly empty settings
    )
    X_empty, _ = dm_empty_settings.load_and_prepare()
    
    assert isinstance(dm_empty_settings.preprocessor, ColumnTransformer)
    assert len(dm_empty_settings.preprocessor.transformers) == 0 # Should be empty transformers
    assert dm_empty_settings.preprocessor.remainder == 'passthrough'
    
    # ADD THIS LINE: Fit the preprocessor before transforming
    dm_empty_settings.create_and_fit_preprocessor(X_empty) 
    
    transformed_empty = dm_empty_settings.transform_data(X_empty)

    # With remainder='passthrough' and no active transformers,
    # all columns from X_empty (A and B) should pass through.
    assert transformed_empty.shape[1] == 2
    # You might also want to check the content, e.g., if it's a numpy array.
    # For A: [1,2,3], For B: ['x','y','z'] (will be encoded if it was categorical)
    # Since there are no categorical settings, 'B' should pass as a string column.
    # transformed_empty might be a numpy array, so direct comparison to df is tricky.
    # You could convert back to DataFrame if needed for content checks.
    
    # Example for checking content if transformed_empty is a numpy array:
    # assert np.array_equal(transformed_empty[:, 0], np.array([1, 2, 3]))
    # assert np.array_equal(transformed_empty[:, 1], np.array([b'x', b'y', b'z']) if isinstance(transformed_empty, np.ndarray) else np.array(['x', 'y', 'z']))
    # For consistency, it's often easier to check shape and dtypes if applicable.


# --- Other tests (e.g., test_get_feature_names_after_preprocessing_passthrough) ---
# Ensure test_get_feature_names_after_preprocessing_passthrough does NOT call create_and_fit_preprocessor
# as the static method now handles the stateless case directly.


