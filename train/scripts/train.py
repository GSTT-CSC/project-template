import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import joblib
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from project.utils.parse_config import load_config, get_model_and_hyperparams, get_data_config, get_logging_config, get_training_config
from project.DataModule import DataModule
import numpy as np
from sklearn.preprocessing import LabelEncoder
import optuna


def objective(trial, model_class, model_name, task, X_raw, y, fitted_preprocessor_global):
    """
    Optuna objective function for a single trial evaluation using a simple train-test split.
    """
    hyperparams = {}
    if model_name == "random_forest":
        hyperparams['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        hyperparams['max_depth'] = trial.suggest_int('max_depth', 5, 20, log=True)
        hyperparams['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
        hyperparams['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
    elif model_name == "xgboost":
        hyperparams['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        hyperparams['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        hyperparams['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        hyperparams['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    elif model_name == "linear_regression":
        hyperparams['fit_intercept'] = trial.suggest_categorical('fit_intercept', [True, False])
        pass

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

    X_train_processed = fitted_preprocessor_global.transform(X_train_raw)
    X_test_processed = fitted_preprocessor_global.transform(X_test_raw)
    
    try:
        processed_feature_names = DataModule.get_feature_names_after_preprocessing(fitted_preprocessor_global, X_raw.columns.tolist())
        X_train_final = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train_raw.index)
        X_test_final = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test_raw.index)
    except Exception:
        X_train_final = pd.DataFrame(X_train_processed)
        X_test_final = pd.DataFrame(X_test_processed)

    model = model_class(**hyperparams) 
    model.fit(X_train_final, y_train) 
    y_pred = model.predict(X_test_final)

    if task == "classification":
        metric = accuracy_score(y_test, y_pred)
    else:
        metric = np.sqrt(mean_squared_error(y_test, y_pred))

    return metric


def main():
    # --- 1. Load Configuration ---
    config_path = "config/local_config.cfg"
    config = load_config(config_path)

    model_class, model_hyperparams_initial = get_model_and_hyperparams(config)
    model_name = config["model"]["model_name"]
    task = config["model"]["task"]
    data_config = get_data_config(config)
    logging_config = get_logging_config(config)
    training_config = get_training_config(config)

    save_model = logging_config['save_model']
    model_output_path = logging_config['model_output_path']
    mlflow_tracking_uri = logging_config['mlflow_tracking_uri']
    use_kfold_cv = training_config['use_kfold_cv']
    n_splits = training_config['n_splits']
    shuffle_cv = training_config['shuffle_cv']
    random_state_cv = training_config['random_state_cv']
    use_optuna = training_config['use_optuna']
    n_trials = training_config['n_trials']
    timeout = training_config['timeout']
    optuna_direction = training_config['optuna_direction']

    # --- 2. MLflow Setup ---
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"{model_name}_experiment")
    with mlflow.start_run(run_name=f"{model_name}_training_run"):
        mlflow.log_params({"model_name": model_name, "task": task})
        mlflow.log_params(training_config)
        mlflow.log_params(data_config)

        categorical_settings = data_config.get('categorical_columns_settings', {})
        numerical_settings = data_config.get('numerical_columns_settings', {})
        preprocessor_settings_for_dm = {}
        if categorical_settings:
            preprocessor_settings_for_dm['categorical'] = categorical_settings
        if numerical_settings:
            preprocessor_settings_for_dm['numerical'] = numerical_settings
        
        dm_init_params = {
            'data_path': data_config['data_path'],
            'target_column': data_config['target_column'],
            'columns_to_drop': data_config.get('columns_to_drop', []),
            'visualise': data_config.get('visualise_data', False),
            'check_imbalance': data_config.get('check_imbalance', False),
            'test_size': data_config.get('test_size', 0.2),
            'stratify': data_config.get('stratify', False),
            'random_state': data_config.get('random_state', 42), 
            'preprocessor_settings': preprocessor_settings_for_dm
        }
      
        dm = DataModule(**dm_init_params)
        X_raw, y = dm.load_and_prepare()

        label_encoder = None
        if task == "classification":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            mlflow.log_param("target_classes_original", label_encoder.classes_.tolist())
       
        fitted_preprocessor_global = dm.create_and_fit_preprocessor(X_raw)

        # --- 3. Hyperparameter Tuning with Optuna or Standard Training ---
        if use_optuna:
            study = optuna.create_study(direction=optuna_direction, sampler=optuna.samplers.TPESampler(seed=random_state_cv))
            study.optimize(
                lambda trial: objective(trial, model_class, model_name, task, X_raw, y, fitted_preprocessor_global),
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True 
            )
            final_model_hyperparams = study.best_params
            mlflow.log_params({f"optuna_best_param_{k}": v for k, v in final_model_hyperparams.items()})
        else:
            final_model_hyperparams = model_hyperparams_initial
            mlflow.log_params(final_model_hyperparams)


        # --- 4. Final Model Training and Evaluation with K-Fold CV ---
        if use_kfold_cv:
            if task == "classification" and data_config.get("stratify", False):
                kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state_cv)
            else:
                kf = KFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state_cv)
            
            fold_metrics = []
            for fold, (train_index, test_index) in enumerate(kf.split(X_raw, y)):
                X_train_fold_raw, X_test_fold_raw = X_raw.iloc[train_index], X_raw.iloc[test_index]
                y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

                X_train_fold_processed = fitted_preprocessor_global.transform(X_train_fold_raw)
                X_test_fold_processed = fitted_preprocessor_global.transform(X_test_fold_raw)

                try:
                    processed_feature_names = DataModule.get_feature_names_after_preprocessing(fitted_preprocessor_global, X_raw.columns.tolist())
                    X_train_final = pd.DataFrame(X_train_fold_processed, columns=processed_feature_names, index=X_train_fold_raw.index)
                    X_test_final = pd.DataFrame(X_test_fold_processed, columns=processed_feature_names, index=test_index)
                except Exception:
                    X_train_final = pd.DataFrame(X_train_fold_processed)
                    X_test_final = pd.DataFrame(X_test_fold_processed)

                model = model_class(**final_model_hyperparams) 
                model.fit(X_train_final, y_train_fold) 
                y_pred = model.predict(X_test_final)

                if task == "classification":
                    metric = accuracy_score(y_test_fold, y_pred)
                    mlflow.log_metric(f"fold_{fold+1}_accuracy", metric)
                else: 
                    rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
                    mlflow.log_metric(f"fold_{fold+1}_rmse", rmse)
                    metric = rmse
                fold_metrics.append(metric)

            avg_metric = np.mean(fold_metrics)
            if task == "classification":
                mlflow.log_metric("avg_accuracy", avg_metric)
            else:
                mlflow.log_metric("avg_rmse", avg_metric)
            
            # Re-train the final model on the full dataset for deployment
            final_model = model_class(**final_model_hyperparams)
            X_processed_final = fitted_preprocessor_global.transform(X_raw)
            final_model.fit(X_processed_final, y)

        # --- 5. Single Train-Test Split (if K-Fold is disabled) ---
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=data_config['test_size'], stratify=data_config['stratify'], random_state=data_config['random_state'])
            
            preprocessor = fitted_preprocessor_global
            X_train_processed = preprocessor.transform(X_train_raw)
            X_test_processed = preprocessor.transform(X_test_raw)

            try:
                processed_feature_names = DataModule.get_feature_names_after_preprocessing(preprocessor, X_raw.columns.tolist())
                X_train_final = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train_raw.index)
                X_test_final = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test_raw.index)
            except Exception:
                X_train_final = pd.DataFrame(X_train_processed)
                X_test_final = pd.DataFrame(X_test_processed)

            final_model = model_class(**final_model_hyperparams)
            final_model.fit(X_train_final, y_train)
            y_pred = final_model.predict(X_test_final)

            if task == "classification":
                acc = accuracy_score(y_test, y_pred)
                mlflow.log_metric("accuracy", acc)
            else: 
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mlflow.log_metric("rmse", rmse)

        # --- 6. Save Model and End Run ---
        if save_model:
            os.makedirs(model_output_path, exist_ok=True)
            model_path = os.path.join(model_output_path, f"{model_name}_{task}.pkl")
            joblib.dump(final_model, model_path)
            mlflow.log_artifact(model_path)
        
        mlflow.end_run()

if __name__ == "__main__":
    main()
