#!/usr/bin/env python3
"""
Training script for structured data ML models.
Supports various models, cross-validation, and MLflow logging.
Updated to work with new DataLoader and DataModule structure.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

import mlflow
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.utils.parse_config import (
    load_config, get_model_and_hyperparams, get_data_config, 
    get_logging_config, get_training_config, validate_config
)
from src.DataModule import DataModule


class ModelTrainer:
    """Handles model training, evaluation, and logging with new architecture."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Validate configuration
        validate_config(self.config)
        
        # Extract configurations
        self.model_class, self.initial_hyperparams = get_model_and_hyperparams(self.config)
        self.model_name = self.config["model"]["model_name"]
        self.task = self.config["model"]["task"]
        self.data_config = get_data_config(self.config)
        self.logging_config = get_logging_config(self.config)
        self.training_config = get_training_config(self.config)
        
        # Initialize components
        self.data_module = None
        self.label_encoder = None
        self.X = None
        self.y = None
        self.final_model = None
        
        print(f"Initializing {self.model_name} trainer for {self.task} task")

    def setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.logging_config['mlflow_tracking_uri'])
        experiment_name = f"{self.model_name}_{self.task}_experiment"
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")

    def load_data(self) -> None:
        """Load and prepare data using the new DataLoader + DataModule architecture."""
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA")
        print("="*60)
        
        # Initialize DataModule with enhanced settings
        self.data_module = DataModule(
            data_path=self.data_config['data_path'],
            target_column=self.data_config['target_column'],
            columns_to_drop=self.data_config.get('columns_to_drop', []),
            preprocessor_settings=self.data_config.get('preprocessor_settings', {}),
            visualise=self.data_config.get('visualise_data'),
            check_imbalance=self.data_config.get('check_imbalance', False),
            test_size=self.data_config.get('test_size', 0.2),
            stratify=self.data_config.get('stratify', False),
            random_state=self.data_config.get('random_state', 42),
            feature_engineering=self.data_config.get('feature_engineering', False)
        )
        
        # Load and prepare data (DataLoader handles the heavy lifting)
        self.X, self.y = self.data_module.load_and_prepare()
        
        # Encode target for classification
        if self.task == "classification":
            self.label_encoder = LabelEncoder()
            self.y = pd.Series(
                self.label_encoder.fit_transform(self.y), 
                index=self.y.index, 
                name=self.y.name
            )
            print(f"Target encoded. Classes: {list(self.label_encoder.classes_)}")
        
        # Print data summary
        summary = self.data_module.get_data_summary()
        print(f"\nFINAL DATA SUMMARY:")
        print(f"   Features shape: {self.X.shape}")
        print(f"   Target shape: {self.y.shape}")
        print(f"   Numerical features: {len(self.data_module.numerical_features)}")
        print(f"   Categorical features: {len(self.data_module.categorical_features)}")

    def load_hyperparameters(self) -> Dict[str, Any]:
        """Load hyperparameters from Optuna or use default."""
        if self.training_config.get('use_optuna', False):
            best_params_path = Path(f"models/best_params_{self.model_name}.json")
            
            if best_params_path.exists():
                with open(best_params_path, 'r') as f:
                    hyperparams = json.load(f)
                print(f"Loaded optimized hyperparameters from {best_params_path}")
                return hyperparams
            else:
                print(f"Optuna results not found at {best_params_path}")
                print("Using default hyperparameters. Run tune.py first for optimal results.")
        
        return self.initial_hyperparams

    def perform_cross_validation(self, hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """Perform cross-validation and return metrics."""
        print(f"\nCROSS-VALIDATION ({self.training_config['n_splits']}-fold)")
        print("-" * 50)
        
        # Setup cross-validation
        if self.task == "classification" and self.data_config.get('stratify', False):
            cv = StratifiedKFold(
                n_splits=self.training_config['n_splits'],
                shuffle=self.training_config['shuffle_cv'],
                random_state=self.training_config['random_state_cv']
            )
            print("Using stratified K-fold cross-validation")
        else:
            cv = KFold(
                n_splits=self.training_config['n_splits'],
                shuffle=self.training_config['shuffle_cv'],
                random_state=self.training_config['random_state_cv']
            )
            print("Using standard K-fold cross-validation")
        
        # Fit preprocessor and transform data
        fitted_preprocessor = self.data_module.create_and_fit_preprocessor(self.X)
        X_processed = fitted_preprocessor.transform(self.X)
        
        # Create model
        model = self.model_class(**hyperparams)
        print(f"Created {self.model_name} model with {len(hyperparams)} hyperparameters")
        
        # Perform cross-validation
        if self.task == "classification":
            cv_scores = cross_val_score(model, X_processed, self.y, cv=cv, scoring='accuracy', n_jobs=-1)
            metric_name = 'accuracy'
        else:
            cv_scores = cross_val_score(model, X_processed, self.y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_scores = np.sqrt(-cv_scores) 
            metric_name = 'rmse'
        
        cv_results = {
            f'cv_mean_{metric_name}': cv_scores.mean(),
            f'cv_std_{metric_name}': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        print(f"CV {metric_name.upper()}: {cv_results[f'cv_mean_{metric_name}']:.4f} ± {cv_results[f'cv_std_{metric_name}']:.4f}")
        print(f"   Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        return cv_results

    def train_final_model(self, hyperparams: Dict[str, Any]) -> None:
        """Train final model on all data."""
        print(f"\nTRAINING FINAL MODEL")
        print("-" * 30)
        
        # Fit preprocessor and transform data
        fitted_preprocessor = self.data_module.create_and_fit_preprocessor(self.X)
        X_processed = fitted_preprocessor.transform(self.X)
        
        # Train final model
        self.final_model = self.model_class(**hyperparams)
        print(f"Training {self.model_name} on {X_processed.shape[0]} samples with {X_processed.shape[1]} features")
        
        self.final_model.fit(X_processed, self.y)
        
        print("Final model training completed")
        
        # Store preprocessor with model for future use
        self.final_model._preprocessor = fitted_preprocessor
        self.final_model._feature_names = self.X.columns.tolist()

    def evaluate_holdout(self, hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model on holdout test set."""
        print(f"\nHOLDOUT EVALUATION")
        print("-" * 25)
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_module.perform_train_test_split(self.X, self.y)
        
        # Fit preprocessor on training data only
        fitted_preprocessor = self.data_module.create_and_fit_preprocessor(X_train)
        
        # Transform data
        X_train_processed = fitted_preprocessor.transform(X_train)
        X_test_processed = fitted_preprocessor.transform(X_test)
        
        # Train model
        model = self.model_class(**hyperparams)
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        if self.task == "classification":
            metrics = {
                'holdout_accuracy': accuracy_score(y_test, y_pred),
                'holdout_f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'holdout_precision_weighted': precision_score(y_test, y_pred, average='weighted'),
                'holdout_recall_weighted': recall_score(y_test, y_pred, average='weighted')
            }
            print(f"Holdout Results:")
            print(f"   Accuracy: {metrics['holdout_accuracy']:.4f}")
            print(f"   F1 (weighted): {metrics['holdout_f1_weighted']:.4f}")
            print(f"   Precision (weighted): {metrics['holdout_precision_weighted']:.4f}")
            print(f"   Recall (weighted): {metrics['holdout_recall_weighted']:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                'holdout_rmse': np.sqrt(mse),
                'holdout_mae': mean_absolute_error(y_test, y_pred),
                'holdout_r2': r2_score(y_test, y_pred)
            }
            print(f"Holdout Results:")
            print(f"   RMSE: {metrics['holdout_rmse']:.4f}")
            print(f"   MAE: {metrics['holdout_mae']:.4f}")
            print(f"   R²: {metrics['holdout_r2']:.4f}")
        
        return metrics

    def save_model(self) -> Optional[str]:
        """Save the trained model with all components."""
        if not self.logging_config.get('save_model', False) or self.final_model is None:
            return None
        
        print(f"\nSAVING MODEL")
        print("-" * 20)
        
        # Create output directory
        output_dir = Path(self.logging_config['model_output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model with comprehensive package
        model_filename = f"{self.model_name}_{self.task}_model.pkl"
        model_path = output_dir / model_filename
        
        # Create comprehensive model package
        model_package = {
            'model': self.final_model,
            'preprocessor': getattr(self.final_model, '_preprocessor', None),
            'label_encoder': self.label_encoder,
            'feature_names': getattr(self.final_model, '_feature_names', self.X.columns.tolist()),
            'numerical_features': self.data_module.numerical_features,
            'categorical_features': self.data_module.categorical_features,
            'model_name': self.model_name,
            'task': self.task,
            'target_column': self.data_config['target_column'],
            'model_config': {
                'hyperparameters': self.initial_hyperparams,
                'data_config': self.data_config,
                'training_config': self.training_config
            },
            'data_summary': self.data_module.get_data_summary()
        }
        
        joblib.dump(model_package, model_path)
        print(f"Model package saved to: {model_path}")
        print(f"   Package includes: model, preprocessor, encoders, metadata")
        
        return str(model_path)

    def run(self) -> None:
        """Run the complete training pipeline."""
        print("\n" + "="*80)
        print("STARTING ML MODEL TRAINING PIPELINE")
        print("="*80)
        
        try:
            # Setup MLflow
            self.setup_mlflow()
            
            with mlflow.start_run(run_name=f"{self.model_name}_{self.task}_training"):
                # Load data
                self.load_data()
                
                # Load hyperparameters
                hyperparams = self.load_hyperparameters()
                print(f"\nUsing hyperparameters: {hyperparams}")
                
                # Log parameters
                mlflow.log_params({
                    'model_name': self.model_name,
                    'task': self.task,
                    **hyperparams,
                    **{k: v for k, v in self.data_config.items() if not isinstance(v, dict)},
                    **self.training_config
                })
                
                # Log additional info
                if self.label_encoder is not None:
                    mlflow.log_param('target_classes', list(self.label_encoder.classes_))
                
                mlflow.log_params({
                    'final_features_count': self.X.shape[1],
                    'samples_count': self.X.shape[0],
                    'numerical_features_count': len(self.data_module.numerical_features),
                    'categorical_features_count': len(self.data_module.categorical_features)
                })
                
                all_metrics = {}
                
                # Cross-validation or holdout evaluation
                if self.training_config.get('use_kfold_cv', True):
                    cv_metrics = self.perform_cross_validation(hyperparams)
                    all_metrics.update(cv_metrics)
                    
                    # Train final model on all data
                    self.train_final_model(hyperparams)
                else:
                    # Use holdout validation
                    holdout_metrics = self.evaluate_holdout(hyperparams)
                    all_metrics.update(holdout_metrics)
                    
                    # Train final model on all data
                    self.train_final_model(hyperparams)
                
                # Log metrics
                for metric_name, metric_value in all_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Save model
                model_path = self.save_model()
                if model_path:
                    mlflow.log_artifact(model_path)
                
                print(f"\n" + "="*80)
                print("TRAINING COMPLETED SUCCESSFULLY!")
                print("="*80)
                print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
                if model_path:
                    print(f"Model saved to: {model_path}")
                print("="*80)
                
        except Exception as e:
            print(f"\nTRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            mlflow.end_run()


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML model on structured data')
    parser.add_argument(
        '--config', 
        default='/Users/ksonar/Documents/Technical/project-template/Structured_data_template/train/config/local_config.cfg',        
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Handle relative paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create and run trainer
    trainer = ModelTrainer(str(config_path))
    trainer.run()


if __name__ == "__main__":
    main()