"""
Data utilities for advanced data processing and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings


class FeatureEngineer:
    """Advanced feature engineering utilities."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.created_features = []
        self.feature_creation_log = []
    
    def create_polynomial_features(self, data: pd.DataFrame, columns: List[str],
                                 degree: int = 2, include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to create polynomial features for
            degree: Polynomial degree
            include_bias: Whether to include bias column
            
        Returns:
            DataFrame with polynomial features added
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        
        # Create polynomial features
        poly_features = poly.fit_transform(data[columns])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Create DataFrame with new features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
        
        # Add to original data (excluding original columns to avoid duplication)
        original_cols = set(columns)
        new_cols = [col for col in poly_df.columns if col not in original_cols]
        
        result_df = data.copy()
        result_df[new_cols] = poly_df[new_cols]
        
        self.created_features.extend(new_cols)
        self.feature_creation_log.append(f"Created {len(new_cols)} polynomial features (degree={degree})")
        
        print(f" Created {len(new_cols)} polynomial features")
        return result_df
    
    def create_interaction_features(self, data: pd.DataFrame, columns: List[str],
                                  max_combinations: int = 10) -> pd.DataFrame:
        """
        Create interaction features between specified columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to create interactions for
            max_combinations: Maximum number of combinations to create
            
        Returns:
            DataFrame with interaction features added
        """
        from itertools import combinations
        
        result_df = data.copy()
        new_features = []
        
        # Create pairwise interactions
        for col1, col2 in combinations(columns, 2):
            if len(new_features) >= max_combinations:
                break
                
            # Skip if either column is not numeric
            if not (pd.api.types.is_numeric_dtype(data[col1]) and pd.api.types.is_numeric_dtype(data[col2])):
                continue
            
            interaction_name = f"{col1}_x_{col2}"
            result_df[interaction_name] = data[col1] * data[col2]
            new_features.append(interaction_name)
        
        self.created_features.extend(new_features)
        self.feature_creation_log.append(f"Created {len(new_features)} interaction features")
        
        print(f"Created {len(new_features)} interaction features")
        return result_df
    
    def create_binning_features(self, data: pd.DataFrame, column: str, 
                              n_bins: int = 5, strategy: str = 'quantile') -> pd.DataFrame:
        """
        Create binning features for a numerical column.
        
        Args:
            data: Input DataFrame
            column: Column to bin
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            
        Returns:
            DataFrame with binning features added
        """
        from sklearn.preprocessing import KBinsDiscretizer
        
        if not pd.api.types.is_numeric_dtype(data[column]):
            print(f" Column {column} is not numeric, skipping binning")
            return data
        
        result_df = data.copy()
        
        # Create bins
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        binned_values = kbd.fit_transform(data[[column]]).flatten()
        
        # Add binned feature
        binned_column_name = f"{column}_binned_{strategy}"
        result_df[binned_column_name] = binned_values
        
        # Create one-hot encoded version
        for i in range(n_bins):
            bin_name = f"{column}_bin_{i}"
            result_df[bin_name] = (binned_values == i).astype(int)
            self.created_features.append(bin_name)
        
        self.created_features.append(binned_column_name)
        self.feature_creation_log.append(f"Created binning features for {column} ({n_bins} bins, {strategy})")
        
        print(f"Created binning features for {column}")
        return result_df
    
    def create_aggregation_features(self, data: pd.DataFrame, group_col: str,
                                  agg_cols: List[str], agg_funcs: List[str] = None) -> pd.DataFrame:
        """
        Create aggregation features based on grouping.
        
        Args:
            data: Input DataFrame
            group_col: Column to group by
            agg_cols: Columns to aggregate
            agg_funcs: Aggregation functions to apply
            
        Returns:
            DataFrame with aggregation features added
        """
        if agg_funcs is None:
            agg_funcs = ['mean', 'std', 'min', 'max', 'count']
        
        result_df = data.copy()
        new_features = []
        
        for agg_col in agg_cols:
            if not pd.api.types.is_numeric_dtype(data[agg_col]):
                continue
                
            for func in agg_funcs:
                try:
                    # Calculate aggregation
                    agg_values = data.groupby(group_col)[agg_col].transform(func)
                    
                    # Add to DataFrame
                    feature_name = f"{agg_col}_{func}_by_{group_col}"
                    result_df[feature_name] = agg_values
                    new_features.append(feature_name)
                    
                except Exception as e:
                    print(f" Could not create {func} aggregation for {agg_col}: {e}")
                    continue
        
        self.created_features.extend(new_features)
        self.feature_creation_log.append(f"Created {len(new_features)} aggregation features")
        
        print(f" Created {len(new_features)} aggregation features")
        return result_df
    
    def create_datetime_features(self, data: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """
        Extract datetime features from a datetime column.
        
        Args:
            data: Input DataFrame
            datetime_col: DateTime column name
            
        Returns:
            DataFrame with datetime features added
        """
        result_df = data.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
            try:
                result_df[datetime_col] = pd.to_datetime(data[datetime_col])
            except Exception as e:
                print(f" Could not convert {datetime_col} to datetime: {e}")
                return data
        
        dt_series = result_df[datetime_col]
        new_features = []
        
        # Extract various datetime components
        datetime_features = {
            f"{datetime_col}_year": dt_series.dt.year,
            f"{datetime_col}_month": dt_series.dt.month,
            f"{datetime_col}_day": dt_series.dt.day,
            f"{datetime_col}_dayofweek": dt_series.dt.dayofweek,
            f"{datetime_col}_hour": dt_series.dt.hour,
            f"{datetime_col}_is_weekend": (dt_series.dt.dayofweek >= 5).astype(int),
            f"{datetime_col}_quarter": dt_series.dt.quarter,
            f"{datetime_col}_is_month_start": dt_series.dt.is_month_start.astype(int),
            f"{datetime_col}_is_month_end": dt_series.dt.is_month_end.astype(int),
        }
        
        for feature_name, feature_values in datetime_features.items():
            if not feature_values.isna().all():  # Only add if not all NaN
                result_df[feature_name] = feature_values
                new_features.append(feature_name)
        
        self.created_features.extend(new_features)
        self.feature_creation_log.append(f"Created {len(new_features)} datetime features from {datetime_col}")
        
        print(f"Created {len(new_features)} datetime features")
        return result_df
    
    def get_feature_creation_summary(self) -> Dict[str, Any]:
        """Get summary of feature creation operations."""
        return {
            'total_features_created': len(self.created_features),
            'created_features': self.created_features,
            'creation_log': self.feature_creation_log
        }


class FeatureSelector:
    """Advanced feature selection utilities."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features = {}
        self.selection_scores = {}
    
    def select_k_best_features(self, X: pd.DataFrame, y: pd.Series, 
                              k: int = 10, task: str = "classification") -> List[str]:
        """
        Select k best features using statistical tests.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            task: "classification" or "regression"
            
        Returns:
            List of selected feature names
        """
        if task == "classification":
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store scores
        feature_scores = dict(zip(X.columns, selector.scores_))
        self.selection_scores['k_best'] = feature_scores
        self.selected_features['k_best'] = selected_features
        
        print(f"Selected {len(selected_features)} best features using statistical tests")
        return selected_features
    
    def select_mutual_info_features(self, X: pd.DataFrame, y: pd.Series,
                                   threshold: float = 0.01, task: str = "classification") -> List[str]:
        """
        Select features using mutual information.
        
        Args:
            X: Feature matrix
            y: Target vector
            threshold: Minimum mutual information threshold
            task: "classification" or "regression"
            
        Returns:
            List of selected feature names
        """
        if task == "classification":
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Select features above threshold
        selected_mask = mi_scores > threshold
        selected_features = X.columns[selected_mask].tolist()
        
        # Store scores
        feature_scores = dict(zip(X.columns, mi_scores))
        self.selection_scores['mutual_info'] = feature_scores
        self.selected_features['mutual_info'] = selected_features
        
        print(f" Selected {len(selected_features)} features using mutual information (threshold={threshold})")
        return selected_features
    
    def select_correlation_features(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold for removal
            
        Returns:
            List of features to keep (low correlation)
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                   if any(upper_triangle[column] > threshold)]
        
        # Features to keep
        features_to_keep = [col for col in X.columns if col not in to_drop]
        
        self.selected_features['low_correlation'] = features_to_keep
        
        print(f"Removed {len(to_drop)} highly correlated features (threshold={threshold})")
        return features_to_keep
    
    def select_variance_threshold_features(self, X: pd.DataFrame, 
                                         threshold: float = 0.0) -> List[str]:
        """
        Remove features with low variance.
        
        Args:
            X: Feature matrix
            threshold: Variance threshold
            
        Returns:
            List of features with variance above threshold
        """
        from sklearn.feature_selection import VarianceThreshold
        
        # Only consider numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print(" No numerical columns for variance threshold selection")
            return X.columns.tolist()
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X[numerical_cols])
        
        # Get selected numerical features
        selected_numerical = numerical_cols[selector.get_support()].tolist()
        
        # Include all categorical features
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        selected_features = selected_numerical + categorical_cols
        
        removed_count = len(numerical_cols) - len(selected_numerical)
        self.selected_features['variance_threshold'] = selected_features
        
        print(f"Removed {removed_count} low-variance features (threshold={threshold})")
        return selected_features
    
    def get_feature_ranking(self, method: str = 'k_best') -> pd.DataFrame:
        """
        Get feature ranking based on selection method.
        
        Args:
            method: Selection method ('k_best', 'mutual_info')
            
        Returns:
            DataFrame with feature rankings
        """
        if method not in self.selection_scores:
            raise ValueError(f"Scores for method '{method}' not available. Run selection first.")
        
        scores = self.selection_scores[method]
        ranking_df = pd.DataFrame({
            'feature': list(scores.keys()),
            'score': list(scores.values())
        }).sort_values('score', ascending=False).reset_index(drop=True)
        
        ranking_df['rank'] = ranking_df.index + 1
        return ranking_df


class DataTransformer:
    """Advanced data transformation utilities."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.transformers = {}
        self.transformation_log = []
    
    def handle_outliers(self, data: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in specified columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to handle outliers for
            method: Method to use ('iqr', 'zscore', 'clip')
            factor: Factor for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        result_df = data.copy()
        outlier_info = {}
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            original_count = (~data[col].isna()).sum()
            
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Clip outliers
                result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                mean_val = data[col].mean()
                std_val = data[col].std()
                lower_bound = mean_val - factor * std_val
                upper_bound = mean_val + factor * std_val
                
                # Clip outliers
                result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'clip':
                # Clip at percentiles
                lower_percentile = (1 - 0.99) / 2 * 100
                upper_percentile = 100 - lower_percentile
                
                lower_bound = data[col].quantile(lower_percentile / 100)
                upper_bound = data[col].quantile(upper_percentile / 100)
                
                result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Count outliers handled
            outliers_handled = (
                (data[col] < result_df[col]).sum() + 
                (data[col] > result_df[col]).sum()
            )
            
            outlier_info[col] = {
                'outliers_handled': outliers_handled,
                'percentage': (outliers_handled / original_count) * 100 if original_count > 0 else 0
            }
        
        total_outliers = sum(info['outliers_handled'] for info in outlier_info.values())
        self.transformation_log.append(f"Handled {total_outliers} outliers using {method} method")
        
        print(f"Handled outliers in {len(columns)} columns using {method} method")
        return result_df
    
    def apply_log_transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply log transformation to specified columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to transform
            
        Returns:
            DataFrame with log-transformed columns
        """
        result_df = data.copy()
        transformed_cols = []
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Check if all values are positive
            if (data[col] <= 0).any():
                print(f" Column {col} contains non-positive values, using log1p instead")
                result_df[f"{col}_log"] = np.log1p(data[col])
            else:
                result_df[f"{col}_log"] = np.log(data[col])
            
            transformed_cols.append(f"{col}_log")
        
        self.transformation_log.append(f"Applied log transformation to {len(transformed_cols)} columns")
        print(f" Applied log transformation to {len(transformed_cols)} columns")
        return result_df
    
    def apply_box_cox_transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply Box-Cox transformation to specified columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to transform
            
        Returns:
            DataFrame with Box-Cox transformed columns
        """
        from scipy.stats import boxcox
        
        result_df = data.copy()
        transformed_cols = []
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Box-Cox requires positive values
            if (data[col] <= 0).any():
                print(f"Column {col} contains non-positive values, skipping Box-Cox")
                continue
            
            try:
                transformed_data, lambda_param = boxcox(data[col].dropna())
                
                # Apply transformation to full column
                result_df[f"{col}_boxcox"] = boxcox(data[col], lmbda=lambda_param)
                transformed_cols.append(f"{col}_boxcox")
                
                # Store transformation parameters
                self.transformers[f"{col}_boxcox_lambda"] = lambda_param
                
            except Exception as e:
                print(f"Could not apply Box-Cox to {col}: {e}")
                continue
        
        self.transformation_log.append(f"Applied Box-Cox transformation to {len(transformed_cols)} columns")
        print(f" Applied Box-Cox transformation to {len(transformed_cols)} columns")
        return result_df
    
    def create_target_encoding(self, data: pd.DataFrame, categorical_col: str, 
                             target_col: str, smoothing: float = 10.0) -> pd.DataFrame:
        """
        Create target encoding for a categorical column.
        
        Args:
            data: Input DataFrame
            categorical_col: Categorical column to encode
            target_col: Target column
            smoothing: Smoothing factor
            
        Returns:
            DataFrame with target encoding added
        """
        result_df = data.copy()
        
        # Calculate global mean
        global_mean = data[target_col].mean()
        
        # Calculate category means and counts
        category_stats = data.groupby(categorical_col)[target_col].agg(['mean', 'count'])
        
        # Apply smoothing
        smoothed_means = (
            (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
            (category_stats['count'] + smoothing)
        )
        
        # Create encoded column
        encoded_col_name = f"{categorical_col}_target_encoded"
        result_df[encoded_col_name] = result_df[categorical_col].map(smoothed_means)
        
        # Fill missing values with global mean
        result_df[encoded_col_name].fillna(global_mean, inplace=True)
        
        # Store encoding mapping
        self.transformers[f"{categorical_col}_target_encoding"] = smoothed_means.to_dict()
        
        self.transformation_log.append(f"Created target encoding for {categorical_col}")
        print(f"Created target encoding for {categorical_col}")
        return result_df


class DataProfiler:
    """Comprehensive data profiling utilities."""
    
    def __init__(self):
        pass
    
    def generate_data_profile(self, data: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.
        
        Args:
            data: Input DataFrame
            target_col: Target column name (optional)
            
        Returns:
            Dictionary with comprehensive data profile
        """
        profile = {
            'overview': self._get_overview(data),
            'columns': self._get_column_profiles(data),
            'correlations': self._get_correlation_analysis(data),
            'missing_data': self._get_missing_data_analysis(data),
            'duplicates': self._get_duplicate_analysis(data),
            'data_quality': self._assess_data_quality(data)
        }
        
        if target_col and target_col in data.columns:
            profile['target_analysis'] = self._get_target_analysis(data, target_col)
        
        return profile
    
    def _get_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview of the dataset."""
        return {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtypes': data.dtypes.value_counts().to_dict(),
            'numerical_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(exclude=[np.number]).columns.tolist()
        }
    
    def _get_column_profiles(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get detailed profile for each column."""
        profiles = {}
        
        for col in data.columns:
            col_data = data[col]
            
            base_profile = {
                'dtype': str(col_data.dtype),
                'missing_count': col_data.isnull().sum(),
                'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Numerical column profile
                base_profile.update({
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q50': col_data.quantile(0.50),
                    'q75': col_data.quantile(0.75),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'zeros_count': (col_data == 0).sum(),
                    'zeros_percentage': ((col_data == 0).sum() / len(col_data)) * 100
                })
            else:
                # Categorical column profile
                value_counts = col_data.value_counts()
                base_profile.update({
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head(5).to_dict()
                })
            
            profiles[col] = base_profile
        
        return profiles
    
    def _get_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numerical columns."""
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            return {'message': 'No numerical columns for correlation analysis'}
        
        corr_matrix = numerical_data.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr_pairs,
            'max_correlation': abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).max(),
            'mean_correlation': abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).mean()
        }
    
    def _get_missing_data_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        return {
            'total_missing_cells': missing_counts.sum(),
            'missing_percentage_overall': (missing_counts.sum() / data.size) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'complete_rows': len(data) - data.isnull().any(axis=1).sum(),
            'complete_rows_percentage': ((len(data) - data.isnull().any(axis=1).sum()) / len(data)) * 100
        }
    
    def _get_duplicate_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate data."""
        duplicate_rows = data.duplicated().sum()
        
        return {
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': (duplicate_rows / len(data)) * 100,
            'unique_rows': len(data) - duplicate_rows
        }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """Assess overall data quality."""
        # Completeness
        completeness = 1 - (data.isnull().sum().sum() / data.size)
        
        # Uniqueness
        uniqueness = 1 - (data.duplicated().sum() / len(data))
        
        # Consistency (based on mixed types in object columns)
        consistency_issues = 0
        object_cols = data.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Simple check for mixed types (numbers in string columns)
            try:
                numeric_count = pd.to_numeric(data[col].dropna(), errors='coerce').notna().sum()
                total_non_null = data[col].notna().sum()
                if total_non_null > 0 and 0 < numeric_count < total_non_null:
                    consistency_issues += 1
            except:
                pass
        
        consistency = 1 - (consistency_issues / len(object_cols)) if len(object_cols) > 0 else 1.0
        
        # Overall quality score
        overall = np.mean([completeness, uniqueness, consistency])
        
        return {
            'completeness': completeness,
            'uniqueness': uniqueness,
            'consistency': consistency,
            'overall_quality': overall
        }
    
    def _get_target_analysis(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze target column specifically."""
        target_data = data[target_col]
        
        analysis = {
            'dtype': str(target_data.dtype),
            'missing_count': target_data.isnull().sum(),
            'unique_count': target_data.nunique(),
            'distribution': target_data.value_counts().to_dict()
        }
        
        if pd.api.types.is_numeric_dtype(target_data):
            analysis.update({
                'mean': target_data.mean(),
                'std': target_data.std(),
                'min': target_data.min(),
                'max': target_data.max(),
                'skewness': target_data.skew(),
                'recommended_task': 'regression' if target_data.nunique() > 20 else 'classification'
            })
        else:
            analysis['recommended_task'] = 'classification'
        
        return analysis
    
    def save_profile_report(self, profile: Dict[str, Any], output_path: str = "data_profile_report.json") -> str:
        """Save data profile to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            else:
                return convert_types(d)
        
        clean_profile = clean_dict(profile)
        
        with open(output_path, 'w') as f:
            json.dump(clean_profile, f, indent=2, default=str)
        
        print(f"Data profile report saved: {output_path}")
        return output_path