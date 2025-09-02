"""
Hyperparameter tuning script using Optuna for structured data ML models.
Supports various models and optimization strategies.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

import mlflow
import optuna
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from src.utils.parse_config import (
    load_config, get_model_and_hyperparams, get_data_config, 
    get_logging_config, get_training_config, validate_config
)
from src.DataModule import DataModule


class HyperparameterTuner:
    """Handles hyperparameter optimization using Optuna."""
    
    def __init__(self, config_path: str):
        """Initialize tuner with configuration."""
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
        self.fitted_preprocessor = None
        
        print(f"Initializing hyperparameter tuning for {self.model_name} ({self.task})")

    def setup_mlflow(self) -> None:
        """Setup MLflow tracking for tuning."""
        mlflow.set_tracking_uri(self.logging_config['mlflow_tracking_uri'])
        experiment_name = f"{self.model_name}_{self.task}_tuning"
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")

    def load_and_prepare_data(self) -> None:
        """Load and prepare data for tuning."""
        print("\n Loading and preparing data")
        
        # Initialize DataModule
        self.data_module = DataModule(
            data_path=self.data_config['data_path'],
            target_column=self.data_config['target_column'],
            columns_to_drop=self.data_config.get('columns_to_drop', []),
            preprocessor_settings=self.data_config.get('preprocessor_settings', {}),
            visualise=False,  # Disable visualization during tuning
            check_imbalance=self.data_config.get('check_imbalance', False),
            test_size=self.data_config.get('test_size', 0.2),
            stratify=self.data_config.get('stratify', False),
            random_state=self.data_config.get('random_state', 42)
        )
        
        # Load and prepare data
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
        
        # Fit preprocessor once for all trials
        self.fitted_preprocessor = self.data_module.create_and_fit_preprocessor(self.X)
        print(" Preprocessor fitted for optimization")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial based on model type."""
        params = {}
        
        if self.model_name == "random_forest":
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.data_config.get('random_state', 42)
            })
            
        elif self.model_name == "xgboost":
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': self.data_config.get('random_state', 42)
            })
            
            # Task-specific parameters
            if self.task == "classification":
                params['objective'] = 'multi:softprob' if len(self.label_encoder.classes_) > 2 else 'binary:logistic'
                params['eval_metric'] = 'mlogloss' if len(self.label_encoder.classes_) > 2 else 'logloss'
            else:
                params['objective'] = 'reg:squarederror'
                params['eval_metric'] = 'rmse'
                
        elif self.model_name == "linear_regression":
            params.update({
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            })
            
        elif self.model_name == "logistic_regression":
            params.update({
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': self.data_config.get('random_state', 42)
            })
            
            # Handle parameter constraints
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
                if params['solver'] not in ['saga']:
                    params['solver'] = 'saga'
            elif params['penalty'] == 'l1':
                if params['solver'] not in ['liblinear', 'saga']:
                    params['solver'] = 'liblinear'
            elif params['penalty'] is None:
                if params['solver'] not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
                    params['solver'] = 'lbfgs'
        
        return params

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Get hyperparameters for this trial
        hyperparams = self.suggest_hyperparameters(trial)
        
        try:
            # Transform data
            X_transformed = self.fitted_preprocessor.transform(self.X)
            
            # Create model with suggested hyperparameters
            model = self.model_class(**hyperparams)
            
            # Setup cross-validation
            cv_folds = self.training_config.get('n_splits', 5)
            random_state = self.training_config.get('random_state_cv', 42)
            
            if self.task == "classification" and self.data_config.get('stratify', False):
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'accuracy'
            elif self.task == "classification":
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'neg_mean_squared_error'
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_transformed, self.y, cv=cv, scoring=scoring, n_jobs=-1)
            
            # Calculate metric
            if self.task == "classification":
                metric = cv_scores.mean()
            else:
                metric = np.sqrt(-cv_scores.mean())  # Convert to RMSE and negate for minimization
                
            return metric
            
        except Exception as e:
            print(f" Trial failed with error: {e}")
            # Return worst possible score for failed trials
            return 0.0 if self.task == "classification" else float('inf')

    def run_optimization(self) -> optuna.Study:
        """Run Optuna optimization."""
        print(f"\n Starting hyperparameter optimization")
        print(f"  Trials: {self.training_config['n_trials']}")
        print(f"  Timeout: {self.training_config['timeout']} seconds")
        print(f"  Direction: {self.training_config['optuna_direction']}")
        
        # Create study
        direction = self.training_config['optuna_direction']
        if self.task == "regression" and direction == "maximize":
            direction = "minimize"  # RMSE should be minimized
            
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.training_config['random_state_cv'])
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.training_config['n_trials'],
            timeout=self.training_config['timeout'],
            show_progress_bar=True
        )
        
        return study

    def save_best_parameters(self, study: optuna.Study) -> str:
        """Save best parameters to JSON file."""
        print(f"\n Saving optimization results")
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        best_params_path = models_dir / f"best_params_{self.model_name}.json"
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        print(f" Best parameters saved to: {best_params_path}")
        
        # Print results summary
        print(f"\n Optimization completed.")
        print(f"  Best value: {study.best_value:.4f}")
        print(f"  Best parameters:")
        for param, value in study.best_params.items():
            print(f"    {param}: {value}")
        
        return str(best_params_path)

    def run(self) -> None:
        """Run the complete tuning pipeline."""
        if not self.training_config.get('use_optuna', False):
            print("Optuna tuning is disabled in configuration. Exiting.")
            return
        
        try:
            # Setup MLflow
            self.setup_mlflow()
            
            with mlflow.start_run(run_name=f"{self.model_name}_{self.task}_tuning"):
                # Load and prepare data
                self.load_and_prepare_data()
                
                # Log configuration parameters
                mlflow.log_params({
                    'model_name': self.model_name,
                    'task': self.task,
                    'n_trials': self.training_config['n_trials'],
                    'timeout': self.training_config['timeout'],
                    'optuna_direction': self.training_config['optuna_direction'],
                    **{k: v for k, v in self.data_config.items() if not isinstance(v, dict)}
                })
                
                # Log additional info
                if self.label_encoder is not None:
                    mlflow.log_param('target_classes', list(self.label_encoder.classes_))
                
                # Run optimization
                study = self.run_optimization()
                
                # Log best results
                mlflow.log_metric('best_value', study.best_value)
                mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
                
                # Save results
                best_params_path = self.save_best_parameters(study)
                mlflow.log_artifact(best_params_path)
                
                # Log study statistics
                mlflow.log_metrics({
                    'n_trials_completed': len(study.trials),
                    'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'n_trials_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                })
                
                print(f"\n Hyperparameter tuning completed successfully!")
                print(f" MLflow run: {mlflow.active_run().info.run_id}")
                print(f" Run train.py next to train the model with optimized parameters")
                
        except Exception as e:
            print(f"\n Tuning failed: {str(e)}")
            raise
        finally:
            mlflow.end_run()


def main():
    """Main tuning function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune hyperparameters for ML model')
    parser.add_argument(
        '--config', 
        default='/Users/ksonar/Documents/Technical/project-template/train/config/local_config.cfg',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Handle relative paths
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = Path(__file__).parent / config_path
    
    if not Path(config_path).exists():
        print(f" Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create and run tuner
    tuner = HyperparameterTuner(str(config_path))
    tuner.run()


if __name__ == "__main__":
    main()