"""
Visualization utilities for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import validation_curve, learning_curve


class DataVisualizer:
    """Comprehensive data visualization utilities."""
    
    def __init__(self, output_dir: str = "plots", style: str = "whitegrid", 
                 figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Seaborn style
            figsize: Default figure size
            dpi: Plot resolution
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        plt.style.use('default')
        sns.set_style(style)
        
        # Try to set interactive backend
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup matplotlib backend for optimal display."""
        try:
            # Try interactive backends
            for backend in ['TkAgg', 'Qt5Agg', 'QtAgg']:
                try:
                    matplotlib.use(backend)
                    break
                except ImportError:
                    continue
            else:
                # Fallback to non-interactive
                matplotlib.use('Agg')
                print("ℹUsing non-interactive backend. Plots will be saved only.")
        except Exception:
            print(" Backend setup failed. Using default.")
    
    def plot_distributions(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                          target_column: Optional[str] = None, 
                          save_name: str = "distributions") -> None:
        """
        Create comprehensive distribution plots.
        
        Args:
            data: DataFrame to plot
            columns: Specific columns to plot (None for all)
            target_column: Target column for colored distributions
            save_name: Base name for saved plots
        """
        if columns is None:
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            numerical_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
            categorical_cols = [col for col in columns if col not in numerical_cols]
        
        # Remove target from feature columns
        if target_column:
            numerical_cols = [col for col in numerical_cols if col != target_column]
            categorical_cols = [col for col in categorical_cols if col != target_column]
        
        # Plot numerical distributions
        if numerical_cols:
            self._plot_numerical_distributions(data, numerical_cols, target_column, 
                                             f"{save_name}_numerical")
        
        # Plot categorical distributions  
        if categorical_cols:
            self._plot_categorical_distributions(data, categorical_cols, target_column,
                                               f"{save_name}_categorical")
        
        # Plot target distribution separately
        if target_column and target_column in data.columns:
            self._plot_target_distribution(data, target_column, f"{save_name}_target")
    
    def _plot_numerical_distributions(self, data: pd.DataFrame, columns: List[str],
                                    target_column: Optional[str], save_name: str) -> None:
        """Plot numerical feature distributions."""
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            ax = axes[i]
            
            if target_column and target_column in data.columns:
                # Colored by target
                unique_targets = data[target_column].unique()
                if len(unique_targets) <= 10:  # Only if not too many classes
                    for target_val in unique_targets:
                        subset = data[data[target_column] == target_val]
                        sns.histplot(subset[col], alpha=0.7, label=f"{target_column}={target_val}",
                                   kde=True, ax=ax)
                    ax.legend()
                else:
                    sns.histplot(data[col], kde=True, ax=ax)
            else:
                sns.histplot(data[col], kde=True, ax=ax)
            
            # Add statistics
            mean_val = data[col].mean()
            std_val = data[col].std()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'μ={mean_val:.2f}')
            ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def _plot_categorical_distributions(self, data: pd.DataFrame, columns: List[str],
                                      target_column: Optional[str], save_name: str) -> None:
        """Plot categorical feature distributions."""
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            ax = axes[i]
            
            # Get top categories to avoid overcrowding
            value_counts = data[col].value_counts().head(15)
            
            if target_column and target_column in data.columns:
                # Stacked bar plot by target
                subset_data = data[data[col].isin(value_counts.index)]
                pd.crosstab(subset_data[col], subset_data[target_column]).plot(kind='bar', 
                          stacked=True, ax=ax, rot=45)
                ax.legend(title=target_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Simple count plot
                sns.countplot(data=data[data[col].isin(value_counts.index)], 
                            x=col, order=value_counts.index, ax=ax)
                ax.tick_params(axis='x', rotation=45)
            
            # Add count annotations
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height),
                              ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add unique count info
            unique_count = data[col].nunique()
            ax.text(0.02, 0.98, f'Unique: {unique_count}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def _plot_target_distribution(self, data: pd.DataFrame, target_column: str, 
                                save_name: str) -> None:
        """Plot target variable distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        target_data = data[target_column]
        
        # Left plot: Basic distribution
        if target_data.dtype in ['object', 'category'] or target_data.nunique() <= 20:
            # Categorical target
            sns.countplot(data=data, x=target_column, ax=axes[0])
            axes[0].set_title(f'Target Distribution: {target_column}')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            total = len(target_data)
            for p in axes[0].patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                axes[0].annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                               ha='center', va='bottom')
        else:
            # Numerical target
            sns.histplot(data=data, x=target_column, kde=True, ax=axes[0])
            axes[0].set_title(f'Target Distribution: {target_column}')
            
            # Add statistics
            mean_val = target_data.mean()
            std_val = target_data.std()
            axes[0].axvline(mean_val, color='red', linestyle='--', alpha=0.7)
            axes[0].text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Right plot: Target vs top features correlation/relationship
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if numerical_cols:
            # Correlation heatmap with target
            target_corr = data[numerical_cols + [target_column]].corr()[target_column].drop(target_column)
            top_corr = target_corr.abs().nlargest(min(10, len(target_corr)))
            
            sns.barplot(x=top_corr.values, y=top_corr.index, ax=axes[1])
            axes[1].set_title(f'Top Features Correlated with {target_column}')
            axes[1].set_xlabel('Correlation Coefficient')
        else:
            axes[1].text(0.5, 0.5, 'No numerical features\nfor correlation analysis', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Correlation Analysis')
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def plot_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson',
                              save_name: str = "correlation_matrix") -> None:
        """Create correlation matrix heatmap."""
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            print(" No numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix.columns)), 
                                       max(6, len(corr_matrix.columns) * 0.8)))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(f'Feature Correlation Matrix ({method.title()})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def plot_missing_data_pattern(self, data: pd.DataFrame, 
                                save_name: str = "missing_data_pattern") -> None:
        """Visualize missing data patterns."""
        missing_data = data.isnull()
        
        if not missing_data.any().any():
            print(" No missing data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Missing data heatmap
        sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Missing Data Pattern')
        
        # 2. Missing data by column
        missing_by_col = missing_data.sum().sort_values(ascending=False)
        missing_by_col = missing_by_col[missing_by_col > 0]
        
        if not missing_by_col.empty:
            sns.barplot(x=missing_by_col.values, y=missing_by_col.index, ax=axes[0,1])
            axes[0,1].set_title('Missing Data Count by Column')
            axes[0,1].set_xlabel('Missing Count')
        
        # 3. Missing data percentage
        missing_pct = (missing_by_col / len(data)) * 100
        if not missing_pct.empty:
            sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=axes[1,0])
            axes[1,0].set_title('Missing Data Percentage by Column')
            axes[1,0].set_xlabel('Missing Percentage (%)')
        
        # 4. Missing data combinations
        missing_combinations = missing_data.value_counts().head(10)
        axes[1,1].bar(range(len(missing_combinations)), missing_combinations.values)
        axes[1,1].set_title('Top Missing Data Combinations')
        axes[1,1].set_xlabel('Pattern Index')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def plot_outliers_analysis(self, data: pd.DataFrame, method: str = 'iqr',
                              save_name: str = "outliers_analysis") -> None:
        """Analyze and visualize outliers."""
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            print(" No numerical columns for outlier analysis")
            return
        
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if len(numerical_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
        else:
            axes = axes.flatten()
        
        outlier_summary = {}
        
        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            
            # Box plot
            sns.boxplot(data=data, y=col, ax=ax)
            ax.set_title(f'Outliers in {col}')
            
            # Calculate outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(data)) * 100
            
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # Add outlier info
            ax.text(0.02, 0.98, f'Outliers: {outlier_count}\n({outlier_pct:.1f}%)', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
        
        # Print outlier summary
        print("\n OUTLIER ANALYSIS SUMMARY:")
        for col, info in outlier_summary.items():
            print(f"  {col}: {info['count']} outliers ({info['percentage']:.1f}%)")
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                               title: str = "Feature Importance", 
                               save_name: str = "feature_importance") -> None:
        """Plot feature importance scores."""
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
        
        bars = sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        
        # Add value labels on bars
        for i, bar in enumerate(bars.patches):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def _save_and_show(self, fig: plt.Figure, save_name: str) -> None:
        """Save figure and optionally display it."""
        # Save figure
        save_path = self.output_dir / f"{save_name}.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"Plot saved: {save_path}")
        
        # Try to display
        try:
            plt.show(block=False)
            print("👁️  Plot displayed")
        except Exception:
            print("Plot saved (display not available)")
        
        plt.close(fig)


class ModelVisualizer:
    """Visualization utilities for model evaluation and performance."""
    
    def __init__(self, output_dir: str = "plots/model_evaluation", 
                 figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            save_name: str = "confusion_matrix") -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       save_name: str = "roc_curves") -> None:
        """Plot ROC curves for multiclass classification."""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            
        else:
            # Multiclass
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            fig, ax = plt.subplots(figsize=self.figsize)
            colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
            
            for i, color in zip(range(n_classes), colors):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_name = class_names[i] if class_names else f'Class {i}'
                
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{class_name} (AUC = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Multi-class ROC Curves')
            ax.legend(loc="lower right")
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def plot_learning_curves(self, estimator, X: np.ndarray, y: np.ndarray,
                           cv: int = 5, scoring: str = 'accuracy',
                           save_name: str = "learning_curves") -> None:
        """Plot learning curves to diagnose bias/variance."""
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.1, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel(f'{scoring.title()} Score')
        ax.set_title('Learning Curves')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def plot_validation_curves(self, estimator, X: np.ndarray, y: np.ndarray,
                             param_name: str, param_range: np.ndarray,
                             cv: int = 5, scoring: str = 'accuracy',
                             save_name: str = "validation_curves") -> None:
        """Plot validation curves for hyperparameter analysis."""
        train_scores, val_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color='blue')
        
        ax.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                       alpha=0.1, color='red')
        
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel(f'{scoring.title()} Score')
        ax.set_title(f'Validation Curves for {param_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Use log scale if parameter values span multiple orders of magnitude
        if param_range.max() / param_range.min() > 100:
            ax.set_xscale('log')
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_name: str = "residuals") -> None:
        """Plot residuals for regression analysis."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Predicted
        axes[0,0].scatter(y_pred, residuals, alpha=0.6)
        axes[0,0].axhline(y=0, color='red', linestyle='--')
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Predicted')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Residuals histogram
        axes[0,1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Residuals')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Residuals Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Predicted vs Actual
        axes[1,1].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1,1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--')
        axes[1,1].set_xlabel('True Values')
        axes[1,1].set_ylabel('Predicted Values')
        axes[1,1].set_title('Predicted vs Actual')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_and_show(fig, save_name)
    
    def _save_and_show(self, fig: plt.Figure, save_name: str) -> None:
        """Save figure and optionally display it."""
        save_path = self.output_dir / f"{save_name}.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f" Model plot saved: {save_path}")
        
        try:
            plt.show(block=False)
            print("Model plot displayed")
        except Exception:
            print("Model plot saved (display not available)")
        
        plt.close(fig)