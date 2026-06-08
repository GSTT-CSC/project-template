import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

# Import DataLoader
from .DataLoader import DataLoader


class DataModule:
    """
    Streamlined class focused on preprocessing pipeline creation and data splitting.
    Works in conjunction with DataLoader for complete ML data preparation.
    """
    
    def __init__(self, data_path: str, target_column: str, columns_to_drop: Optional[List[str]] = None,
                 preprocessor_settings: Optional[Dict[str, Any]] = None, visualise: bool = False,
                 check_imbalance: bool = False, test_size: float = 0.2, stratify: bool = False,
                 random_state: int = 42, feature_engineering: bool = False):
        """
        Initialize DataModule.
        
        Args:
            data_path: Path to the CSV data file
            target_column: Name of the target column
            columns_to_drop: List of columns to drop
            preprocessor_settings: Settings for preprocessing pipelines
            visualise: Whether to generate visualizations
            check_imbalance: Whether to check class imbalance
            test_size: Proportion of data for testing
            stratify: Whether to use stratified splitting
            random_state: Random state for reproducibility
            feature_engineering: Whether to perform automated feature engineering
        """
        self.data_path = data_path
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop or []
        self.preprocessor_settings = preprocessor_settings or {}
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        
        # Initialize DataLoader
        self.data_loader = DataLoader(
            data_path=data_path,
            target_column=target_column,
            columns_to_drop=columns_to_drop,
            visualise=visualise,
            check_imbalance=check_imbalance,
            random_state=random_state,
            feature_engineering=feature_engineering
        )
        
        # Internal attributes
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data using DataLoader and prepare preprocessing pipeline.
        
        Returns:
            tuple: Features (X) and target (y) DataFrames
        """
        print("\n=== DATAMODULE: PREPROCESSING PIPELINE SETUP ===")
        
        # Use DataLoader to get clean, prepared data
        self.X, self.y = self.data_loader.load_and_prepare_data()
        
        # Get feature types from DataLoader
        self.numerical_features = self.data_loader.numerical_features.copy()
        self.categorical_features = self.data_loader.categorical_features.copy()
        
        # Update feature types after any transformations
        self._update_feature_types()
        
        # Setup preprocessor
        self._setup_preprocessor()
        
        return self.X, self.y

    def _update_feature_types(self) -> None:
        """Update feature types after data loading and transformations."""
        # Re-infer column types in case DataLoader created new features
        current_numerical = self.X.select_dtypes(include=[np.number]).columns.tolist()
        current_categorical = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Update only if there are significant changes
        if set(current_numerical) != set(self.numerical_features) or set(current_categorical) != set(self.categorical_features):
            print(f"Updating feature types after data loading:")
            print(f"  Numerical: {len(self.numerical_features)} -> {len(current_numerical)}")
            print(f"  Categorical: {len(self.categorical_features)} -> {len(current_categorical)}")
            
            self.numerical_features = current_numerical
            self.categorical_features = current_categorical

    def _setup_preprocessor(self) -> None:
        """Setup the preprocessing pipeline using ColumnTransformer."""
        print("\nSetting up preprocessing pipeline...")
        
        transformers = []
        
        # Numerical preprocessing
        if self.numerical_features:
            num_pipeline = self._create_numerical_pipeline()
            if num_pipeline.steps:  
                transformers.append(('num', num_pipeline, self.numerical_features))
        
        # Categorical preprocessing
        if self.categorical_features:
            cat_pipeline = self._create_categorical_pipeline()
            if cat_pipeline.steps:  
                transformers.append(('cat', cat_pipeline, self.categorical_features))
        
        # Create ColumnTransformer
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                sparse_threshold=0
            )
            print(f"Preprocessor created with {len(transformers)} transformer(s)")
        else:
            # Fallback: passthrough preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[],
                remainder='passthrough',
                sparse_threshold=0
            )
            print("No active transformers configured, using passthrough")

    def _create_numerical_pipeline(self) -> Pipeline:
        """Create numerical preprocessing pipeline."""
        steps = []
        
        # Get numerical settings from config
        num_settings = self.preprocessor_settings.get('numerical', {})
        
        # Collect strategies across all columns
        imputation_methods = set()
        scaling_methods = set()
        
        for col_name, settings in num_settings.items():
            if col_name in self.numerical_features:
                if 'imputer' in settings:
                    strategy = 'median' if settings['imputer'] == 'median' else 'mean'
                    imputation_methods.add(strategy)
                
                if 'scaler' in settings:
                    scaling_methods.add(settings['scaler'])
        
        # Add imputer if any numerical columns need imputation or if there are missing values
        missing_values_exist = self.X[self.numerical_features].isnull().any().any()
        if imputation_methods or missing_values_exist:
            strategy = list(imputation_methods)[0] if imputation_methods else 'median'
            steps.append(('imputer', SimpleImputer(strategy=strategy)))
            print(f"  Added numerical imputer: {strategy}")
        
        # Add scaler if specified or as default
        if scaling_methods or num_settings:
            scaler_type = list(scaling_methods)[0] if scaling_methods else 'standard'
            if scaler_type == 'standard':
                steps.append(('scaler', StandardScaler()))
            elif scaler_type == 'minmax':
                steps.append(('scaler', MinMaxScaler()))
            print(f"  Added numerical scaler: {scaler_type}")
        
        return Pipeline(steps)

    def _create_categorical_pipeline(self) -> Pipeline:
        """Create categorical preprocessing pipeline."""
        steps = []
        
        # Get categorical settings from config
        cat_settings = self.preprocessor_settings.get('categorical', {})
        
        # Add imputation step if needed
        missing_values_exist = self.X[self.categorical_features].isnull().any().any()
        imputation_needed = any('imputer' in settings for settings in cat_settings.values())
        
        if imputation_needed or missing_values_exist:
            steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            print("  Added categorical imputer: most_frequent")
        
        # Add encoding step
        encoding_methods = set()
        encoder_options = {}
        
        for col_name, settings in cat_settings.items():
            if col_name in self.categorical_features and 'encoder' in settings:
                encoder_type = settings['encoder']
                encoding_methods.add(encoder_type)
                if 'encoder_options' in settings:
                    encoder_options[encoder_type] = settings['encoder_options']
        
        # Use specified encoder or default to OneHot
        if encoding_methods or self.categorical_features:
            encoder_type = list(encoding_methods)[0] if encoding_methods else 'onehot'
            options = encoder_options.get(encoder_type, {})
            
            if encoder_type == 'onehot':
                # Set default options for OneHotEncoder
                default_options = {'handle_unknown': 'ignore', 'sparse_output': False}
                default_options.update(options)
                steps.append(('encoder', OneHotEncoder(**default_options)))
                print(f"  Added categorical encoder: OneHot with options {default_options}")
            elif encoder_type == 'ordinal':
                default_options = {'handle_unknown': 'use_encoded_value', 'unknown_value': -1}
                default_options.update(options)
                steps.append(('encoder', OrdinalEncoder(**default_options)))
                print(f"  Added categorical encoder: Ordinal with options {default_options}")
        
        return Pipeline(steps)

    def create_and_fit_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Fit the preprocessor on the provided data.
        
        Args:
            X: Feature DataFrame to fit on
            
        Returns:
            Fitted preprocessor
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not set up. Call load_and_prepare() first.")
        
        print("Fitting preprocessor on training data...")
        self.preprocessor.fit(X)
        print("Preprocessor fitting completed")
        return self.preprocessor

    def transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data array
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not set up. Call load_and_prepare() first.")
        
        return self.preprocessor.transform(X)

    def perform_train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform train-test split.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nPerforming train-test split (test_size={self.test_size})...")
        
        stratify_param = y if self.stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        split_type = "stratified" if self.stratify else "random"
        print(f"  {split_type.title()} split completed: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_feature_names_after_preprocessing(fitted_preprocessor: ColumnTransformer, 
                                            initial_feature_names: List[str]) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Args:
            fitted_preprocessor: Fitted ColumnTransformer
            initial_feature_names: Original feature names
            
        Returns:
            List of feature names after preprocessing
        """
        try:
            if hasattr(fitted_preprocessor, 'get_feature_names_out'):
                return fitted_preprocessor.get_feature_names_out(initial_feature_names).tolist()
            else:
                print("WARNING: Cannot determine feature names after preprocessing")
                # Estimate number of features
                dummy_df = pd.DataFrame(columns=initial_feature_names)
                n_features = fitted_preprocessor.transform(dummy_df).shape[1] if len(dummy_df.columns) > 0 else 0
                return [f'feature_{i}' for i in range(n_features)]
        except Exception as e:
            print(f"WARNING: Error getting feature names: {e}")
            return []

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary including DataLoader insights.
        
        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'datamodule_info': {
                'numerical_features': self.numerical_features,
                'categorical_features': self.categorical_features,
                'preprocessor_configured': self.preprocessor is not None,
                'test_size': self.test_size,
                'stratify': self.stratify
            }
        }
        
        # Add DataLoader summary if available
        if self.data_loader:
            loader_summary = self.data_loader.get_data_summary()
            summary.update(loader_summary)
        
        return summary

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing pipeline.
        
        Returns:
            Dictionary with preprocessing pipeline details
        """
        if self.preprocessor is None:
            return {'status': 'not_configured'}
        
        info = {
            'status': 'configured',
            'transformers': [],
            'numerical_features_count': len(self.numerical_features),
            'categorical_features_count': len(self.categorical_features)
        }
        
        # Extract transformer information
        for name, transformer, columns in self.preprocessor.transformers_:
            transformer_info = {
                'name': name,
                'type': type(transformer).__name__,
                'columns': columns if isinstance(columns, list) else list(columns),
                'steps': []
            }
            
            if hasattr(transformer, 'steps'):
                for step_name, step_transformer in transformer.steps:
                    transformer_info['steps'].append({
                        'name': step_name,
                        'type': type(step_transformer).__name__
                    })
            
            info['transformers'].append(transformer_info)
        
        return info