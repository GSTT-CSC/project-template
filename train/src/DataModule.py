import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



class DataModule:
    """
    A comprehensive class to handle data loading, preprocessing, splitting, and visualization
    for machine learning projects.
    """
    
    def __init__(self, data_path: str, target_column: str, columns_to_drop: Optional[List[str]] = None,
                 preprocessor_settings: Optional[Dict[str, Any]] = None, visualise: bool = False,
                 check_imbalance: bool = False, test_size: float = 0.2, stratify: bool = False,
                 random_state: int = 42):
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
        """
        self.data_path = data_path
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop or []
        self.preprocessor_settings = preprocessor_settings or {}
        self.visualise = visualise
        self.check_imbalance = check_imbalance
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        
        # Internal attributes
        self.data: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data, prepare features, and set up preprocessing pipeline.
        
        Returns:
            tuple: Features (X) and target (y) DataFrames
        """
        # Load data
        self._load_data()
        
        # Drop specified columns
        self._drop_columns()
        
        # Validate target column
        self._validate_target_column()
        
        # Split features and target
        X, y = self._split_features_target()
        
        # Infer column types
        self._infer_column_types(X)
        
        # Setup preprocessor
        self._setup_preprocessor()
        
        # Optional operations
        if self.visualise:
            self.visualize_column_distributions()
        
        if self.check_imbalance:
            self._check_target_imbalance(y)
        
        return X, y

    def _load_data(self) -> None:
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise IOError(f"Error loading data: {e}")

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
            print(f" Dropped columns: {', '.join(sorted(dropped_actual))}")

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
        
        print(f" Numerical features ({len(self.numerical_features)}): {self.numerical_features}")
        print(f" Categorical features ({len(self.categorical_features)}): {self.categorical_features}")

    def _setup_preprocessor(self) -> None:
        """Setup the preprocessing pipeline using ColumnTransformer."""
        transformers = []
        
        # Numerical preprocessing
        if self.numerical_features:
            num_pipeline = self._create_numerical_pipeline()
            if num_pipeline.steps:  # Only add if pipeline has steps
                transformers.append(('num', num_pipeline, self.numerical_features))
        
        # Categorical preprocessing
        if self.categorical_features:
            cat_pipeline = self._create_categorical_pipeline()
            if cat_pipeline.steps:  # Only add if pipeline has steps
                transformers.append(('cat', cat_pipeline, self.categorical_features))
        
        # Create ColumnTransformer
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                sparse_threshold=0
            )
            print("Preprocessor created with active transformers")
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
        
        # Add imputation step if specified
        imputation_methods = []
        scaling_methods = []
        
        for col_name, settings in num_settings.items():
            if col_name in self.numerical_features:
                if 'imputer' in settings:
                    strategy = 'median' if settings['imputer'] == 'median' else 'mean'
                    if strategy not in imputation_methods:
                        imputation_methods.append(strategy)
                
                if 'scaler' in settings:
                    scaler_type = settings['scaler']
                    if scaler_type not in scaling_methods:
                        scaling_methods.append(scaler_type)
        
        # Add imputer (use median as default if any numerical columns need imputation)
        if imputation_methods or any('imputer' in settings for settings in num_settings.values()):
            strategy = imputation_methods[0] if imputation_methods else 'median'
            steps.append(('imputer', SimpleImputer(strategy=strategy)))
        
        # Add scaler (use StandardScaler as default if any numerical columns need scaling)
        if scaling_methods or any('scaler' in settings for settings in num_settings.values()):
            scaler_type = scaling_methods[0] if scaling_methods else 'standard'
            if scaler_type == 'standard':
                steps.append(('scaler', StandardScaler()))
            elif scaler_type == 'minmax':
                steps.append(('scaler', MinMaxScaler()))
        
        return Pipeline(steps)

    def _create_categorical_pipeline(self) -> Pipeline:
        """Create categorical preprocessing pipeline."""
        steps = []
        
        # Get categorical settings from config
        cat_settings = self.preprocessor_settings.get('categorical', {})
        
        # Add imputation step if specified
        imputation_needed = any('imputer' in settings for settings in cat_settings.values())
        if imputation_needed:
            steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        # Add encoding step
        encoding_methods = []
        encoder_options = {}
        
        for col_name, settings in cat_settings.items():
            if col_name in self.categorical_features and 'encoder' in settings:
                encoder_type = settings['encoder']
                if encoder_type not in encoding_methods:
                    encoding_methods.append(encoder_type)
                    if 'encoder_options' in settings:
                        encoder_options[encoder_type] = settings['encoder_options']
        
        # Use OneHot as default if any categorical columns need encoding
        if encoding_methods or self.categorical_features:
            encoder_type = encoding_methods[0] if encoding_methods else 'onehot'
            options = encoder_options.get(encoder_type, {})
            
            if encoder_type == 'onehot':
                # Set default options for OneHotEncoder
                default_options = {'handle_unknown': 'ignore', 'sparse_output': False}
                default_options.update(options)
                steps.append(('encoder', OneHotEncoder(**default_options)))
            elif encoder_type == 'ordinal':
                default_options = {'handle_unknown': 'use_encoded_value', 'unknown_value': -1}
                default_options.update(options)
                steps.append(('encoder', OrdinalEncoder(**default_options)))
        
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
        
        print("Fitting preprocessor")
        self.preprocessor.fit(X)
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
        stratify_param = y if self.stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        split_type = "stratified" if self.stratify else "random"
        print(f"{split_type.title()} train-test split completed")
        print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test

    def visualize_column_distributions(self) -> None:
        """Generate distribution plots for numerical and categorical columns."""
        if self.data is None:
            print("⚠ Data not loaded. Cannot generate visualizations.")
            return
        
        print("\n Generating distribution visualizations")
        
        # Set matplotlib backend and style
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend first, then switch if needed
        try:
            matplotlib.use('TkAgg')  # Try interactive backend
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                print("⚠ Using non-interactive matplotlib backend. Plots will be saved instead of displayed.")
        
        plt.style.use('default')
        
        # Create output directory for plots if using non-interactive backend
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        
        # Numerical distributions
        if self.numerical_features:
            print(f" Creating plots for {len(self.numerical_features)} numerical features")
            
            n_cols = min(3, len(self.numerical_features))
            n_rows = (len(self.numerical_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows * n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(self.numerical_features):
                try:
                    # Create histogram with KDE
                    sns.histplot(data=self.data, x=col, kde=True, ax=axes[i], stat='density')
                    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Density', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics text
                    mean_val = self.data[col].mean()
                    std_val = self.data[col].std()
                    axes[i].text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error plotting {col}:\n{str(e)}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Error: {col}')
            
            # Hide empty subplots
            for i in range(len(self.numerical_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save and/or show plot
            numerical_plot_path = plot_dir / "numerical_distributions.png"
            plt.savefig(numerical_plot_path, dpi=300, bbox_inches='tight')
            print(f"  Numerical distributions saved to: {numerical_plot_path}")
            
            try:
                plt.show(block=False)
                print("  Numerical distributions displayed")
            except Exception:
                print(" Numerical distributions saved (display not available)")
            
            plt.close()
        
        # Categorical distributions
        if self.categorical_features:
            print(f" Creating plots for {len(self.categorical_features)} categorical features")
            
            n_cols = min(2, len(self.categorical_features))
            n_rows = (len(self.categorical_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            if n_rows * n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(self.categorical_features):
                try:
                    # Get value counts for better plotting
                    value_counts = self.data[col].value_counts().head(20)  # Show top 20 categories
                    
                    # Create count plot
                    sns.countplot(data=self.data[self.data[col].isin(value_counts.index)], 
                                x=col, ax=axes[i], order=value_counts.index)
                    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Count', fontsize=10)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3, axis='y')
                    
                    # Add count annotations on bars
                    for p in axes[i].patches:
                        height = p.get_height()
                        if height > 0:
                            axes[i].annotate(f'{int(height)}', 
                                           (p.get_x() + p.get_width()/2., height),
                                           ha='center', va='bottom', fontsize=8)
                    
                    # Add unique count info
                    unique_count = self.data[col].nunique()
                    axes[i].text(0.02, 0.98, f'Unique: {unique_count}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error plotting {col}:\n{str(e)}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Error: {col}')
            
            # Hide empty subplots
            for i in range(len(self.categorical_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save and/or show plot
            categorical_plot_path = plot_dir / "categorical_distributions.png"
            plt.savefig(categorical_plot_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Categorical distributions saved to: {categorical_plot_path}")
            
            try:
                plt.show(block=False)
                print("  👁️  Categorical distributions displayed")
            except Exception:
                print("  📊 Categorical distributions saved (display not available)")
            
            plt.close()
        
        # Target distribution
        if self.data is not None and self.target_column in self.data.columns:
            print(f"  🎯 Creating target variable plot...")
            
            plt.figure(figsize=(10, 6))
            
            target_data = self.data[self.target_column]
            
            if target_data.dtype in ['object', 'category'] or target_data.nunique() <= 20:
                # Categorical target
                sns.countplot(data=self.data, x=self.target_column)
                plt.title(f'Distribution of Target: {self.target_column}', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45)
                
                # Add percentage labels
                total = len(target_data)
                for p in plt.gca().patches:
                    percentage = f'{100 * p.get_height() / total:.1f}%'
                    plt.gca().annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                                     ha='center', va='bottom')
            else:
                # Numerical target
                sns.histplot(data=self.data, x=self.target_column, kde=True)
                plt.title(f'Distribution of Target: {self.target_column}', fontsize=14, fontweight='bold')
                
                # Add statistics
                mean_val = target_data.mean()
                std_val = target_data.std()
                plt.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save and/or show plot
            target_plot_path = plot_dir / "target_distribution.png"
            plt.savefig(target_plot_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Target distribution saved to: {target_plot_path}")
            
            try:
                plt.show(block=False)
                print("  👁️  Target distribution displayed")
            except Exception:
                print("  📊 Target distribution saved (display not available)")
            
            plt.close()
        
        print(f"✅ Visualization complete! Plots saved in '{plot_dir}' directory")
        
        # Keep plots open for a moment if interactive
        try:
            import time
            time.sleep(1)  # Brief pause to ensure plots are rendered
        except:
            pass

    def _check_target_imbalance(self, y: pd.Series) -> None:
        """Check and report target class imbalance."""
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        
        print(f"\n Target variable '{self.target_column}' distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Check for significant imbalance
        min_percentage = (class_counts.min() / total_samples) * 100
        if min_percentage < 10:
            print(f" Warning: Class imbalance detected! Smallest class: {min_percentage:.1f}%")
            print("  Consider using stratified sampling or class balancing techniques.")

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
                print("⚠ Cannot determine feature names after preprocessing")
                return [f'feature_{i}' for i in range(fitted_preprocessor.transform(
                    pd.DataFrame(columns=initial_feature_names)).shape[1])]
        except Exception as e:
            print(f"⚠ Error getting feature names: {e}")
            return []

    def plot_correlation_heatmap(self) -> None:
        """Plot correlation heatmap for numerical features."""
        if self.data is None or not self.numerical_features:
            print("⚠ No numerical features available for correlation analysis")
            return
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data[self.numerical_features].corr()
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Returns:
            Dictionary containing data summary statistics
        """
        if self.data is None:
            return {}
        
        summary = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict()
        }
        
        if self.target_column in self.data.columns:
            summary['target_distribution'] = self.data[self.target_column].value_counts().to_dict()
        
        return summary