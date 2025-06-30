import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataModule:
    """
    A class to handle data loading, preprocessing, splitting, and visualization.
    """
    def __init__(self, data_path, target_column, columns_to_drop=None,
                 numerical_cols=None, categorical_cols=None, text_cols=None,
                 preprocessor_settings=None, visualise=False, check_imbalance=False,
                 test_size=0.2, stratify=False, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols
        self.preprocessor_settings = preprocessor_settings if preprocessor_settings is not None else {}
        self.visualise = visualise
        self.check_imbalance = check_imbalance
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state

        self.data = None
        # MODIFIED: Initialize preprocessor as None, will be set to ColumnTransformer object
        self.preprocessor = None
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = [] # Placeholder, not actively used in current preprocessing logic

    def load_and_prepare(self):
        """
        Loads data from the specified path, drops columns, infers types,
        sets up the preprocessor, and optionally performs imbalance check and visualization.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully from {self.data_path}. Shape: {self.data.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise IOError(f"Error loading data from {self.data_path}: {e}")

        if self.columns_to_drop:
            initial_columns = set(self.data.columns)
            self.data = self.data.drop(columns=self.columns_to_drop, errors='ignore')
            dropped_actual = initial_columns - set(self.data.columns)
            if dropped_actual:
                print(f"Dropped columns: {', '.join(dropped_actual)}")
            else:
                print("No specified columns were dropped (they might not exist).")

        # Separate features (X) and target (y)
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the data.")
        
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Infer column types if not explicitly provided
        self.infer_column_types(X)

        # NEW: Call setup_preprocessor after column types are inferred
        self.setup_preprocessor()

        if self.visualise:
            self.visualize_column_distributions()
        
        if self.check_imbalance:
            print(f"Checking target class imbalance for '{self.target_column}'...")
            self.check_imbalance_and_report(y)

        return X, y

    def infer_column_types(self, X):
        """
        Infers numerical and categorical column types from the DataFrame X.
        Populates self.numerical_features and self.categorical_features.
        """
        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = X.select_dtypes(include='object').columns.tolist()
        print(f"Inferred numerical features: {self.numerical_features}")
        print(f"Inferred categorical features: {self.categorical_features}")

    def setup_preprocessor(self):
        """
        Sets up the ColumnTransformer based on inferred column types and
        preprocessor_settings. Ensures self.preprocessor is always a ColumnTransformer object.
        """
        if self.data is None: # self.data might not be loaded if DataModule is used for inference without load_and_prepare
            print("Warning: Data not loaded. Preprocessor setup might be incomplete without full data context.")
            # We can still proceed if numerical_features/categorical_features were set externally
            # or if preprocessor_settings explicitly define columns.
            
        # Ensure numerical_features and categorical_features are populated if self.data is available
        # This block is somewhat redundant if setup_preprocessor is always called after infer_column_types in load_and_prepare
        # but good for robustness if called independently.
        if self.data is not None and not self.numerical_features and not self.categorical_features:
            temp_X = self.data.drop(columns=[self.target_column], errors='ignore')
            self.infer_column_types(temp_X)

        active_transformers = []

        # Numerical Pipeline
        numerical_pipeline_steps = []
        # Check if 'numerical' key exists in preprocessor_settings
        if 'numerical' in self.preprocessor_settings:
            # Iterate through the dictionary for numerical settings
            for setting_name, setting_details in self.preprocessor_settings['numerical'].items():
                if setting_details.get('scaler') == 'standard': # Changed 'type' to 'scaler' based on config
                    numerical_pipeline_steps.append((f'{setting_name}_scaler', StandardScaler()))
                # Add other numerical transformers here if needed (e.g., MinMaxScaler, RobustScaler)
        
        if self.numerical_features and numerical_pipeline_steps:
            active_transformers.append(('num_pipeline', Pipeline(numerical_pipeline_steps), self.numerical_features))


        # Categorical Pipeline
        categorical_pipeline_steps = []
        # Check if 'categorical' key exists in preprocessor_settings
        if 'categorical' in self.preprocessor_settings:
            # Iterate through the dictionary for categorical settings
            for setting_name, setting_details in self.preprocessor_settings['categorical'].items():
                if setting_details.get('encoder') == 'onehot': # Changed 'type' to 'encoder' based on config
                    # Pass encoder_options if available
                    encoder_options = setting_details.get('encoder_options', {})
                    categorical_pipeline_steps.append((f'{setting_name}_encoder', OneHotEncoder(handle_unknown='ignore', **encoder_options)))
                elif setting_details.get('encoder') == 'ordinal': # Changed 'type' to 'encoder' based on config
                    # For OrdinalEncoder, you might need to specify categories based on your data if order matters
                    # If categories are provided in settings, use them. Otherwise, default.
                    encoder_options = setting_details.get('encoder_options', {})
                    categorical_pipeline_steps.append((f'{setting_name}_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, **encoder_options)))
                else:
                    print(f"Warning: Unsupported encoder type '{setting_details.get('encoder')}' for column '{setting_name}'. Skipping encoder for this column.")
        
        if self.categorical_features and categorical_pipeline_steps:
            active_transformers.append(('cat_pipeline', Pipeline(categorical_pipeline_steps), self.categorical_features))


        if not active_transformers:
            print("No active transformers configured for any columns. Creating a passthrough preprocessor.")
            # MODIFIED: Assign an empty ColumnTransformer for 'passthrough' behavior
            self.preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
        else:
            self.preprocessor = ColumnTransformer(transformers=active_transformers, remainder='passthrough')
            print("ColumnTransformer created with specified pipelines.")

    def create_and_fit_preprocessor(self, X):
        """
        Fits the preprocessor (ColumnTransformer) on the provided data.
        Returns the fitted preprocessor.
        Assumes self.preprocessor has already been set up in setup_preprocessor().
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been set up. Call setup_preprocessor() first.")
        
        print("Fitting preprocessor...")
        # LATEST CHANGE: Explicitly re-assigning the fitted preprocessor
        self.preprocessor = self.preprocessor.fit(X) 
        return self.preprocessor

    def transform_data(self, X):
        """
        Transforms the data using the fitted preprocessor.
        Assumes self.preprocessor has already been fitted.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been set up. Call setup_preprocessor() first.")

        print("Transforming data...")
        return self.preprocessor.transform(X)


    def perform_train_test_split(self, X, y):
        """
        Performs train-test split on the data.
        """
        if self.stratify:
            print(f"Performing stratified train-test split on '{self.target_column}'.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
        else:
            print("Performing non-stratified train-test split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def visualize_column_distributions(self):
        """
        Visualizes distributions of numerical and categorical columns.
        """
        if self.data is None:
            print("Data not loaded. Cannot visualize distributions.")
            return

        print("\n--- Generating Column Distribution Visualizations ---")

        # Numerical columns
        if self.numerical_features:
            print("Displaying distributions for numerical columns...")
            for col in self.numerical_features:
                plt.figure(figsize=(8, 6))
                sns.histplot(self.data[col], kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.show()

        # Categorical columns
        if self.categorical_features:
            print("Displaying distributions for categorical columns...")
            for col in self.categorical_features:
                plt.figure(figsize=(8, 6))
                # Change: Assign x to hue and set legend=False to address FutureWarning
                sns.countplot(data=self.data, x=col, hue=col, palette='viridis', legend=False)
                plt.title(f'Count of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.show()
        
        print("Column distribution visualizations complete.")

    def check_imbalance_and_report(self, y):
        """
        Checks for target class imbalance and prints a report.
        """
        class_counts = y.value_counts()
        total_samples = len(y)
        print(f"Class distribution for '{self.target_column}':")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")

        # You might add a threshold for "imbalance" and print a warning if exceeded
        min_class_percentage = class_counts.min() / total_samples * 100
        if min_class_percentage < 10: # Example threshold
            print(f"Warning: Potential class imbalance detected. Smallest class has {min_class_percentage:.2f}% of samples.")

    @staticmethod
    def get_feature_names_after_preprocessing(fitted_preprocessor, initial_feature_names):
        """
        Get feature names after preprocessing.
        Assumes ColumnTransformer is used and has get_feature_names_out.
        """
        # LATEST CHANGE: Special handling for a truly passthrough ColumnTransformer (no actual transformers)
        # This prevents NotFittedError on get_feature_names_out for stateless transformers.
        if isinstance(fitted_preprocessor, ColumnTransformer) and \
           not fitted_preprocessor.transformers and \
           fitted_preprocessor.remainder == 'passthrough':
            print("Preprocessor is a passthrough with no active transformers. Returning initial feature names.")
            return initial_feature_names
            
        if hasattr(fitted_preprocessor, 'get_feature_names_out'):
            return fitted_preprocessor.get_feature_names_out(initial_feature_names)
        else:
            print("Warning: Preprocessor does not have 'get_feature_names_out'. Cannot determine feature names automatically.")
            return []

    # Optional: Add methods for plotting if needed elsewhere, e.g. for correlation, scatter.
    def plot_correlation_heatmap(self):
        """Plots a correlation heatmap for numerical features."""
        if self.data is None:
            print("Data not loaded. Cannot plot correlation heatmap.")
            return
        
        if not self.numerical_features:
            print("No numerical features to plot correlation for.")
            return

        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data[self.numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()

    def plot_scatter(self, x_col, y_col, hue_col=None):
        """Plots a scatter plot for two numerical columns, optionally with a hue based on a third column."""
        if self.data is None:
            print("Data not loaded. Cannot plot scatter plot.")
            return

        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Error: '{x_col}' or '{y_col}' not found in data for scatter plot.")
            return
        
        if hue_col and hue_col not in self.data.columns:
             print(f"Warning: Hue column '{hue_col}' not found in data. Plotting without hue.")
             hue_col = None

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=hue_col)
        plt.title(f'Scatter Plot of {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if hue_col:
            plt.legend(title=hue_col)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()