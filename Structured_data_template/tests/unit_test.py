"""
Comprehensive unit tests for the ML pipeline components.
Tests cover DataLoader, DataModule, data utilities, and configuration parsing.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import configparser
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the modules to test
from train.src.DataLoader import DataLoader
from train.src.DataModule import DataModule
from train.src.utils.data_utils import FeatureEngineer, FeatureSelector, DataTransformer, DataProfiler
from train.src.utils.parse_config import load_config, get_model_and_hyperparams, get_data_config
from train.src.utils.visualise import DataVisualizer


class TestDataLoader(unittest.TestCase):
    """Test suite for DataLoader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'numerical_col1': [1, 2, 3, 4, 5, 100],  # Contains outlier
            'numerical_col2': [10.5, 20.1, 30.2, 40.3, 50.4, 60.5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A'],
            'target': ['class1', 'class2', 'class1', 'class3', 'class2', 'class1']
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        # Initialize DataLoader
        self.data_loader = DataLoader(
            data_path=self.temp_file.name,
            target_column='target',
            visualise=False,  # Disable for testing
            feature_engineering=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_data_loading(self):
        """Test basic data loading functionality."""
        self.data_loader._load_data()
        
        self.assertIsNotNone(self.data_loader.data)
        self.assertEqual(self.data_loader.data.shape, (6, 4))
        self.assertTrue('target' in self.data_loader.data.columns)
    
    def test_column_type_inference(self):
        """Test automatic column type inference."""
        self.data_loader._load_data()
        X, y = self.data_loader._split_features_target()
        self.data_loader._infer_column_types(X)
        
        expected_numerical = ['numerical_col1', 'numerical_col2']
        expected_categorical = ['categorical_col']
        
        self.assertEqual(set(self.data_loader.numerical_features), set(expected_numerical))
        self.assertEqual(set(self.data_loader.categorical_features), set(expected_categorical))
    
    def test_data_cleaning_removes_duplicates(self):
        """Test that data cleaning removes duplicate rows."""
        # Add duplicate row
        duplicate_data = pd.concat([self.sample_data, self.sample_data.iloc[[0]]], ignore_index=True)
        duplicate_data.to_csv(self.temp_file.name, index=False)
        
        self.data_loader._load_data()
        initial_shape = self.data_loader.data.shape
        self.data_loader._clean_data()
        final_shape = self.data_loader.data.shape
        
        self.assertLess(final_shape[0], initial_shape[0])
    
    def test_columns_to_drop(self):
        """Test column dropping functionality."""
        self.data_loader.columns_to_drop = ['numerical_col1']
        self.data_loader._load_data()
        self.data_loader._drop_columns()
        
        self.assertNotIn('numerical_col1', self.data_loader.data.columns)
        self.assertIn('numerical_col2', self.data_loader.data.columns)
    
    def test_target_validation(self):
        """Test target column validation."""
        self.data_loader.target_column = 'nonexistent_column'
        self.data_loader._load_data()
        
        with self.assertRaises(ValueError):
            self.data_loader._validate_target_column()
    
    def test_complete_pipeline(self):
        """Test the complete data loading and preparation pipeline."""
        X, y = self.data_loader.load_and_prepare_data()
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertNotIn(self.data_loader.target_column, X.columns)


class TestDataModule(unittest.TestCase):
    """Test suite for DataModule functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'num_col': [1, 2, 3, 4, 5],
            'cat_col': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        # Setup preprocessing settings
        self.preprocessor_settings = {
            'numerical': {
                'num_col': {'imputer': 'mean', 'scaler': 'standard'}
            },
            'categorical': {
                'cat_col': {'imputer': 'mode', 'encoder': 'onehot'}
            }
        }
        
        self.data_module = DataModule(
            data_path=self.temp_file.name,
            target_column='target',
            preprocessor_settings=self.preprocessor_settings,
            visualise=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_load_and_prepare(self):
        """Test the complete load and prepare pipeline."""
        X, y = self.data_module.load_and_prepare()
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertIsNotNone(self.data_module.preprocessor)
    
    def test_preprocessor_creation(self):
        """Test preprocessing pipeline creation."""
        X, y = self.data_module.load_and_prepare()
        
        # Check that preprocessor was created
        self.assertIsNotNone(self.data_module.preprocessor)
        
        # Check that it has the expected transformers
        transformer_names = [name for name, _, _ in self.data_module.preprocessor.transformers_]
        self.assertIn('num', transformer_names)
        self.assertIn('cat', transformer_names)
    
    def test_train_test_split(self):
        """Test train-test splitting functionality."""
        X, y = self.data_module.load_and_prepare()
        X_train, X_test, y_train, y_test = self.data_module.perform_train_test_split(X, y)
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))
        
        # Check that split maintains data integrity
        self.assertEqual(X_train.shape[1], X_test.shape[1])
    
    def test_preprocessor_fit_transform(self):
        """Test preprocessor fitting and transformation."""
        X, y = self.data_module.load_and_prepare()
        
        # Fit preprocessor
        fitted_preprocessor = self.data_module.create_and_fit_preprocessor(X)
        
        # Transform data
        X_transformed = self.data_module.transform_data(X)
        
        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertEqual(X_transformed.shape[0], len(X))


class TestFeatureEngineer(unittest.TestCase):
    """Test suite for FeatureEngineer utility class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer(random_state=42)
        self.sample_data = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', 'C', 'B']
        })
    
    def test_interaction_features(self):
        """Test interaction feature creation."""
        result = self.feature_engineer.create_interaction_features(
            self.sample_data, 
            ['num1', 'num2'], 
            max_combinations=5
        )
        
        # Check that interaction feature was created
        interaction_cols = [col for col in result.columns if '_x_' in col]
        self.assertTrue(len(interaction_cols) > 0)
        self.assertIn('num1_x_num2', result.columns)
        
        # Check interaction values are correct
        expected_interaction = self.sample_data['num1'] * self.sample_data['num2']
        pd.testing.assert_series_equal(
            result['num1_x_num2'], 
            expected_interaction, 
            check_names=False
        )
    
    def test_polynomial_features(self):
        """Test polynomial feature creation."""
        result = self.feature_engineer.create_polynomial_features(
            self.sample_data,
            ['num1'],
            degree=2
        )
        
        # Check that polynomial features were created
        poly_cols = [col for col in result.columns if 'num1^2' in col or 'num1 num1' in col]
        self.assertTrue(len(poly_cols) > 0)
    
    def test_binning_features(self):
        """Test binning feature creation."""
        result = self.feature_engineer.create_binning_features(
            self.sample_data,
            'num1',
            n_bins=3
        )
        
        # Check that binning features were created
        bin_cols = [col for col in result.columns if 'bin' in col]
        self.assertTrue(len(bin_cols) > 0)
    
    def test_feature_creation_log(self):
        """Test that feature creation is properly logged."""
        initial_log_length = len(self.feature_engineer.feature_creation_log)
        
        self.feature_engineer.create_interaction_features(
            self.sample_data, 
            ['num1', 'num2']
        )
        
        final_log_length = len(self.feature_engineer.feature_creation_log)
        self.assertGreater(final_log_length, initial_log_length)


class TestDataTransformer(unittest.TestCase):
    """Test suite for DataTransformer utility class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_transformer = DataTransformer(random_state=42)
        self.sample_data = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'outlier_col': [1, 2, 3, 4, 100],  # Contains outlier
            'skewed_col': [1, 2, 4, 8, 16]     # Positively skewed
        })
    
    def test_outlier_handling_iqr(self):
        """Test IQR-based outlier handling."""
        result = self.data_transformer.handle_outliers(
            self.sample_data,
            ['outlier_col'],
            method='iqr'
        )
        
        # The outlier (100) should be clipped
        self.assertLess(result['outlier_col'].max(), 100)
        self.assertEqual(len(result), len(self.sample_data))
    
    def test_log_transformation(self):
        """Test log transformation."""
        result = self.data_transformer.apply_log_transform(
            self.sample_data,
            ['skewed_col']
        )
        
        # Check that log column was created
        self.assertIn('skewed_col_log', result.columns)
        
        # Check log values are correct
        expected_log = np.log(self.sample_data['skewed_col'])
        pd.testing.assert_series_equal(
            result['skewed_col_log'], 
            expected_log, 
            check_names=False
        )
    
    def test_log_transformation_non_positive(self):
        """Test log transformation with non-positive values."""
        data_with_zeros = self.sample_data.copy()
        data_with_zeros.loc[0, 'skewed_col'] = 0
        
        result = self.data_transformer.apply_log_transform(
            data_with_zeros,
            ['skewed_col']
        )
        
        # Should use log1p for non-positive values
        self.assertIn('skewed_col_log', result.columns)


class TestDataProfiler(unittest.TestCase):
    """Test suite for DataProfiler utility class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_profiler = DataProfiler()
        self.sample_data = pd.DataFrame({
            'numerical': [1, 2, 3, 4, 5, np.nan],
            'categorical': ['A', 'B', 'A', 'C', 'B', 'A'],
            'target': [0, 1, 0, 1, 0, 1]
        })
    
    def test_data_profile_generation(self):
        """Test comprehensive data profile generation."""
        profile = self.data_profiler.generate_data_profile(
            self.sample_data, 
            target_col='target'
        )
        
        # Check that all major sections are present
        expected_sections = ['overview', 'columns', 'missing_data', 'data_quality']
        for section in expected_sections:
            self.assertIn(section, profile)
        
        # Check overview section
        self.assertEqual(profile['overview']['shape'], (6, 3))
        self.assertIn('numerical_columns', profile['overview'])
        self.assertIn('categorical_columns', profile['overview'])
    
    def test_column_profiling(self):
        """Test individual column profiling."""
        profile = self.data_profiler.generate_data_profile(self.sample_data)
        
        # Check numerical column profile
        num_profile = profile['columns']['numerical']
        self.assertEqual(num_profile['missing_count'], 1)
        self.assertIn('mean', num_profile)
        self.assertIn('std', num_profile)
        
        # Check categorical column profile
        cat_profile = profile['columns']['categorical']
        self.assertEqual(cat_profile['missing_count'], 0)
        self.assertIn('most_frequent', cat_profile)
        self.assertIn('unique_count', cat_profile)
    
    def test_missing_data_analysis(self):
        """Test missing data analysis."""
        profile = self.data_profiler.generate_data_profile(self.sample_data)
        
        missing_data = profile['missing_data']
        self.assertEqual(missing_data['total_missing_cells'], 1)
        self.assertIn('numerical', missing_data['columns_with_missing'])
    
    def test_json_serialization(self):
        """Test that profile can be serialized to JSON."""
        profile = self.data_profiler.generate_data_profile(self.sample_data)
        
        # This should not raise an exception
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            try:
                output_path = self.data_profiler.save_profile_report(profile, temp_file.name)
                self.assertTrue(os.path.exists(output_path))
                
                # Verify the file contains valid JSON
                with open(output_path, 'r') as f:
                    loaded_profile = json.load(f)
                    self.assertIsInstance(loaded_profile, dict)
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)


class TestConfigParsing(unittest.TestCase):
    """Test suite for configuration parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_content = """
[project]
name = test_project

[data]
data_path = /path/to/data.csv
target_column = target
visualise_data = true
feature_engineering = false
test_size = 0.2
random_state = 42

[model]
model_name = logistic_regression
task = classification

[hyperparameters.logistic_regression]
C = 1.0
penalty = l2
max_iter = 1000

[preprocessing.numerical]
feature1.imputer = mean
feature1.scaler = standard

[preprocessing.categorical]
feature2.encoder = onehot
feature2.encoder_options = {"sparse_output": false}
"""
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
        self.temp_config.write(self.config_content)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config.name)
    
    def test_config_loading(self):
        """Test basic configuration loading."""
        config = load_config(self.temp_config.name)
        
        self.assertIsInstance(config, configparser.ConfigParser)
        self.assertEqual(config['project']['name'], 'test_project')
        self.assertEqual(config['data']['target_column'], 'target')
    
    def test_model_and_hyperparams_extraction(self):
        """Test model and hyperparameter extraction."""
        config = load_config(self.temp_config.name)
        model_class, hyperparams = get_model_and_hyperparams(config)
        
        # Check that we get the right model class
        from sklearn.linear_model import LogisticRegression
        self.assertEqual(model_class, LogisticRegression)
        
        # Check hyperparameters
        self.assertIn('C', hyperparams)
        self.assertEqual(hyperparams['C'], 1.0)
        self.assertEqual(hyperparams['penalty'], 'l2')
    
    def test_data_config_extraction(self):
        """Test data configuration extraction."""
        config = load_config(self.temp_config.name)
        data_config = get_data_config(config)
        
        # Check basic data settings
        self.assertEqual(data_config['data_path'], '/path/to/data.csv')
        self.assertEqual(data_config['target_column'], 'target')
        self.assertTrue(data_config['visualise_data'])
        self.assertFalse(data_config['feature_engineering'])
        
        # Check preprocessing settings
        self.assertIn('preprocessor_settings', data_config)
        preprocessing = data_config['preprocessor_settings']
        
        self.assertIn('numerical', preprocessing)
        self.assertIn('categorical', preprocessing)
        
        # Check specific preprocessing settings
        self.assertEqual(preprocessing['numerical']['feature1']['imputer'], 'mean')
        self.assertEqual(preprocessing['categorical']['feature2']['encoder'], 'onehot')


class TestVisualizerRobustness(unittest.TestCase):
    """Test suite for visualizer error handling and robustness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = DataVisualizer(output_dir="test_plots")
        self.sample_data = pd.DataFrame({
            'numerical': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'mixed_target': ['class1', 'class2', 'class1', 'class2', 'class1']
        })
    
    def tearDown(self):
        """Clean up test plots directory."""
        import shutil
        if os.path.exists("test_plots"):
            shutil.rmtree("test_plots")
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_categorical_target_correlation_handling(self, mock_savefig, mock_show):
        """Test that categorical targets don't break correlation analysis."""
        # This should not raise an exception
        self.visualizer._plot_target_distribution(
            self.sample_data, 
            'mixed_target', 
            'test_target_plot'
        )
        
        # Verify that savefig was called (plot was created)
        mock_savefig.assert_called()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_empty_data_handling(self, mock_savefig, mock_show):
        """Test handling of edge cases like empty data."""
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            self.visualizer.plot_distributions(empty_data)
        except Exception as e:
            # Should not crash with unhandled exceptions
            self.fail(f"Visualizer should handle empty data gracefully, but raised: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a more comprehensive dataset
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 100),
            'target': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Create temporary CSV
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        # Create config
        self.config_content = f"""
[data]
data_path = {self.temp_file.name}
target_column = target
visualise_data = false
feature_engineering = false
test_size = 0.2
random_state = 42

[preprocessing.numerical]
age.imputer = median
age.scaler = standard
income.imputer = mean
income.scaler = standard

[preprocessing.categorical]
education.encoder = onehot
city.encoder = onehot
"""
        
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
        self.temp_config.write(self.config_content)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        os.unlink(self.temp_file.name)
        os.unlink(self.temp_config.name)
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        # Load configuration
        config = load_config(self.temp_config.name)
        data_config = get_data_config(config)
        
        # Initialize DataModule
        data_module = DataModule(
            data_path=data_config['data_path'],
            target_column=data_config['target_column'],
            preprocessor_settings=data_config['preprocessor_settings'],
            test_size=data_config['test_size'],
            random_state=data_config['random_state'],
            visualise=False
        )
        
        # Load and prepare data
        X, y = data_module.load_and_prepare()
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = data_module.perform_train_test_split(X, y)
        
        # Fit preprocessor and transform data
        fitted_preprocessor = data_module.create_and_fit_preprocessor(X_train)
        X_train_transformed = data_module.transform_data(X_train)
        X_test_transformed = data_module.transform_data(X_test)
        
        # Verify final results
        self.assertIsInstance(X_train_transformed, np.ndarray)
        self.assertIsInstance(X_test_transformed, np.ndarray)
        self.assertEqual(X_train_transformed.shape[1], X_test_transformed.shape[1])
        self.assertGreater(X_train_transformed.shape[1], 2)  # Should have expanded due to encoding


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_classes = [
        TestDataLoader,
        TestDataModule,
        TestFeatureEngineer,
        TestDataTransformer,
        TestDataProfiler,
        TestConfigParsing,
        TestVisualizerRobustness,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    # Run specific test class or all tests
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test class
        test_class_name = sys.argv[1]
        if test_class_name in globals():
            unittest.main(argv=[''], test_class=globals()[test_class_name], verbosity=2)
        else:
            print(f"Test class {test_class_name} not found")
    else:
        # Run all tests
        result = run_all_tests()
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"TESTS RUN: {result.testsRun}")
        print(f"FAILURES: {len(result.failures)}")
        print(f"ERRORS: {len(result.errors)}")
        print(f"SUCCESS RATE: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        print(f"{'='*50}")