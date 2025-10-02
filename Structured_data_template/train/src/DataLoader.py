import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility modules
from .utils.data_utils import FeatureEngineer, FeatureSelector, DataTransformer, DataProfiler
from .utils.visualise import DataVisualizer


class DataLoader:
    """
    Enhanced class to load, clean, visualize and prepare data for machine learning.
    Handles initial data preparation before preprocessing pipeline creation.
    """
    
    def __init__(self, data_path: str, target_column: str, columns_to_drop: Optional[List[str]] = None,
                 visualise: bool = False, check_imbalance: bool = False, 
                 random_state: int = 42, feature_engineering: bool = False):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV data file
            target_column: Name of the target column
            columns_to_drop: List of columns to drop
            visualise: Whether to generate visualizations
            check_imbalance: Whether to check class imbalance
            random_state: Random state for reproducibility
            feature_engineering: Whether to perform automated feature engineering
        """
        self.data_path = data_path
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop or []
        self.visualise = visualise
        self.check_imbalance = check_imbalance
        self.random_state = random_state
        self.feature_engineering = feature_engineering
        
        # Internal attributes
        self.data: Optional[pd.DataFrame] = None
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        
        # Utility classes
        self.feature_engineer = FeatureEngineer(random_state=random_state) if feature_engineering else None
        self.feature_selector = FeatureSelector(random_state=random_state)
        self.data_transformer = DataTransformer(random_state=random_state)
        self.data_profiler = DataProfiler()
        self.visualizer = DataVisualizer() if visualise else None
        
        print(f"DataLoader initialized for target: {target_column}")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main method to load and prepare data for ML pipeline.
        
        Returns:
            tuple: Prepared features (X) and target (y)
        """
        print("\n=== DATA LOADING AND PREPARATION ===")
        
        # Load raw data
        self._load_data()
        
        # Generate data profile if requested
        if self.visualise:
            self._generate_data_profile()
        
        # Data cleaning and preparation
        self._clean_data()
        
        # Drop specified columns
        self._drop_columns()
        
        # Validate target column
        self._validate_target_column()
        
        # Split features and target
        X, y = self._split_features_target()
        
        # Infer column types
        self._infer_column_types(X)
        
        # Data quality improvements
        X = self._improve_data_quality(X)
        
        # Feature engineering (if enabled)
        if self.feature_engineering and self.feature_engineer:
            X = self._perform_feature_engineering(X)
        
        # Visualization
        if self.visualise and self.visualizer:
            self._create_visualizations(X, y)
        
        # Check target imbalance
        if self.check_imbalance:
            self._check_target_imbalance(y)
        
        print(f"Data preparation completed. Final shape: {X.shape}")
        return X, y

    def _load_data(self) -> None:
        """Load data from CSV file."""
        try:
            print(f"Loading data from: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise IOError(f"Error loading data: {e}")

    def _generate_data_profile(self) -> None:
        """Generate comprehensive data profile."""
        print("\nGenerating comprehensive data profile")
        profile = self.data_profiler.generate_data_profile(self.data, self.target_column)
        
        # Save profile report
        profile_path = self.data_profiler.save_profile_report(profile, "data_profile_report.json")
        
        # Print key insights
        overview = profile.get('overview', {})
        print(f"  Dataset shape: {overview.get('shape', 'Unknown')}")
        print(f"  Memory usage: {overview.get('memory_usage_mb', 0):.2f} MB")
        print(f"  Data quality score: {profile.get('data_quality', {}).get('overall_quality', 0):.2f}")

    def _clean_data(self) -> None:
        """Perform basic data cleaning operations."""
        print("\nPerforming data cleaning")
        
        initial_shape = self.data.shape
        
        # Remove completely empty rows and columns.
        self.data = self.data.dropna(how='all').dropna(axis=1, how='all')
        
        # Remove duplicate rows
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        duplicates_removed = initial_rows - len(self.data)
        
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed} duplicate rows")
        
        # Basic data type optimization
        self._optimize_data_types()
        
        final_shape = self.data.shape
        if initial_shape != final_shape:
            print(f"  Shape after cleaning: {initial_shape} -> {final_shape}")

    def _optimize_data_types(self) -> None:
        """Optimize data types to reduce memory usage."""
        for col in self.data.columns:
            col_type = self.data[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if few unique values
                if self.data[col].nunique() / len(self.data) < 0.5:
                    self.data[col] = self.data[col].astype('category')
            
            elif col_type in ['int64', 'float64']:
                # Downcast numeric types
                if col_type == 'int64':
                    if self.data[col].min() >= 0:
                        if self.data[col].max() <= 255:
                            self.data[col] = self.data[col].astype('uint8')
                        elif self.data[col].max() <= 65535:
                            self.data[col] = self.data[col].astype('uint16')
                        elif self.data[col].max() <= 4294967295:
                            self.data[col] = self.data[col].astype('uint32')
                    else:
                        if self.data[col].min() >= -128 and self.data[col].max() <= 127:
                            self.data[col] = self.data[col].astype('int8')
                        elif self.data[col].min() >= -32768 and self.data[col].max() <= 32767:
                            self.data[col] = self.data[col].astype('int16')
                        elif self.data[col].min() >= -2147483648 and self.data[col].max() <= 2147483647:
                            self.data[col] = self.data[col].astype('int32')

    def _drop_columns(self) -> None:
        """Drop specified columns from the dataset."""
        if not self.columns_to_drop:
            return
            
        initial_columns = set(self.data.columns)
        
        # Handle case where columns_to_drop might be a single string
        if isinstance(self.columns_to_drop, str):
            self.columns_to_drop = [col.strip() for col in self.columns_to_drop.split(',')]
            
        self.data = self.data.drop(columns=self.columns_to_drop, errors='ignore')
        dropped_actual = initial_columns - set(self.data.columns)
        
        if dropped_actual:
            print(f"Dropped columns: {', '.join(sorted(dropped_actual))}")

    def _validate_target_column(self) -> None:
        """Validate that target column exists in data."""
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")

    def _split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target."""
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return X, y

    def _infer_column_types(self, X: pd.DataFrame) -> None:
        """Infer numerical and categorical column types."""
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Inferred column types:")
        print(f"  Numerical features ({len(self.numerical_features)}): {self.numerical_features}")
        print(f"  Categorical features ({len(self.categorical_features)}): {self.categorical_features}")

    def _improve_data_quality(self, X: pd.DataFrame) -> pd.DataFrame:
        """Improve data quality through various techniques."""
        print("\nImproving data quality")
        
        # Handle outliers in numerical columns
        if self.numerical_features:
            X = self.data_transformer.handle_outliers(
                X, self.numerical_features, method='iqr', factor=1.5
            )
        
        # Apply transformations if needed
        skewed_features = []
        for col in self.numerical_features:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                skewness = X[col].skew()
                if abs(skewness) > 2:  # Highly skewed
                    skewed_features.append(col)
        
        if skewed_features:
            print(f"Applying log transformation to skewed features: {skewed_features}")
            X = self.data_transformer.apply_log_transform(X, skewed_features)
        
        return X

    def _perform_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform automated feature engineering."""
        print("\nPerforming feature engineering")
        
        # Create interaction features for top numerical features
        if len(self.numerical_features) >= 2:
            top_numerical = self.numerical_features[:min(5, len(self.numerical_features))]
            X = self.feature_engineer.create_interaction_features(X, top_numerical, max_combinations=5)
        
        # Create polynomial features for selected numerical columns
        if self.numerical_features:
            # Select columns with reasonable ranges for polynomial features
            suitable_cols = []
            for col in self.numerical_features[:3]:  # Limit to first 3 to avoid explosion
                if col in X.columns:
                    col_range = X[col].max() - X[col].min()
                    if col_range > 0 and col_range < 1000:  # Reasonable range
                        suitable_cols.append(col)
            
            if suitable_cols:
                X = self.feature_engineer.create_polynomial_features(
                    X, suitable_cols, degree=2, include_bias=False
                )
        
        # Create binning features for numerical columns with high cardinality
        for col in self.numerical_features:
            if col in X.columns and X[col].nunique() > 20:
                X = self.feature_engineer.create_binning_features(X, col, n_bins=5, strategy='quantile')
        
        # Print summary
        if self.feature_engineer:
            summary = self.feature_engineer.get_feature_creation_summary()
            print(f"Feature engineering summary: {summary['total_features_created']} new features created")
        
        return X

    def _create_visualizations(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Create comprehensive visualizations."""
        print("\nCreating visualizations")
        
        # Combine features and target for visualization
        viz_data = X.copy()
        viz_data[self.target_column] = y
        
        # Basic distributions
        self.visualizer.plot_distributions(
            viz_data, 
            target_column=self.target_column,
            save_name="feature_distributions"
        )
        
        # Correlation matrix for numerical features
        if self.numerical_features:
            self.visualizer.plot_correlation_matrix(
                viz_data,
                save_name="correlation_matrix"
            )
        
        # Missing data patterns
        self.visualizer.plot_missing_data_pattern(
            viz_data,
            save_name="missing_data_patterns"
        )
        
        # Outlier analysis
        self.visualizer.plot_outliers_analysis(
            viz_data,
            save_name="outlier_analysis"
        )

    def _check_target_imbalance(self, y: pd.Series) -> None:
        """Check and report target class imbalance."""
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        
        print(f"\nTarget variable '{self.target_column}' distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Check for significant imbalance
        min_percentage = (class_counts.min() / total_samples) * 100
        if min_percentage < 10:
            print(f"WARNING: Class imbalance detected! Smallest class: {min_percentage:.1f}%")
            print("  Consider using stratified sampling or class balancing techniques.")

    @staticmethod
    def impute_data(df: pd.DataFrame, categorical_columns: List[str], numerical_columns: List[str]) -> pd.DataFrame:
        """
        Basic imputation method (kept for backward compatibility).
        
        Args:
            df: DataFrame to impute
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        for col in categorical_columns:
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else 'Unknown')
        
        for col in numerical_columns:
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
        
        return df_imputed

    @staticmethod
    def remove_outliers(df: pd.DataFrame, numerical_columns: List[str], threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method (kept for backward compatibility).
        
        Args:
            df: DataFrame to process
            numerical_columns: List of numerical column names
            threshold: IQR threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        for col in numerical_columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean

    @staticmethod
    def normalize_data(df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """
        Normalize numerical data (kept for backward compatibility).
        
        Args:
            df: DataFrame to normalize
            numerical_columns: List of numerical column names
            
        Returns:
            DataFrame with normalized values
        """
        df_normalized = df.copy()
        
        for col in numerical_columns:
            if col in df_normalized.columns:
                df_normalized[col] = (df_normalized[col] - df_normalized[col].mean()) / df_normalized[col].std()
        
        return df_normalized

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Returns:
            Dictionary containing data summary statistics
        """
        if self.data is None:
            return {}
        
        summary = {
            'original_shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.astype(str).to_dict()
        }
        
        if self.target_column in self.data.columns:
            summary['target_distribution'] = self.data[self.target_column].value_counts().to_dict()
        
        # Add feature engineering summary if available
        if self.feature_engineer:
            summary['feature_engineering'] = self.feature_engineer.get_feature_creation_summary()
        
        return summary